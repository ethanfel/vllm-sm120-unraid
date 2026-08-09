#!/usr/bin/env python3
"""Wake-on-request reverse proxy and lifecycle controller for vLLM.

The proxy stays resident on the public port while the vLLM child process is
fully stopped when idle. The first OpenAI request starts vLLM, waits for its
private health endpoint, and then forwards the original request.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


LOG = logging.getLogger("vllm.on_demand")
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class ModelController:
    command: list[str]
    upstream_port: int
    idle_timeout: float
    load_timeout: float
    stop_timeout: float
    min_free_vram_mib: int = 0
    comfyui_base_url: str = ""
    comfyui_free_timeout: float = 120.0
    control_key: str = ""
    state: str = "unloaded"
    process: asyncio.subprocess.Process | None = None
    inflight: int = 0
    last_activity: float = field(default_factory=time.monotonic)
    last_error: str | None = None
    transition_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    client: httpx.AsyncClient | None = None
    watchdog_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None

    @property
    def upstream_url(self) -> str:
        return f"http://127.0.0.1:{self.upstream_port}"

    async def start(self) -> None:
        timeout = httpx.Timeout(connect=5.0, read=None, write=None, pool=10.0)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self.client = httpx.AsyncClient(timeout=timeout, limits=limits)
        self.watchdog_task = asyncio.create_task(self._idle_watchdog())
        LOG.info("Model is unloaded; waiting for the first /v1 request")

    async def close(self) -> None:
        if self.watchdog_task is not None:
            self.watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.watchdog_task
        async with self.transition_lock:
            await self._stop_process_locked("container shutdown")
        if self.monitor_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task
        if self.client is not None:
            await self.client.aclose()

    async def _set_state(self, state: str, error: str | None = None) -> None:
        async with self.state_lock:
            self.state = state
            self.last_error = error

    async def _monitor_process(self, process: asyncio.subprocess.Process) -> None:
        return_code = await process.wait()
        async with self.state_lock:
            if self.process is process:
                self.process = None
                if self.state == "stopping":
                    self.state = "unloaded"
                    self.last_error = None
                else:
                    self.state = "error"
                    self.last_error = f"vLLM exited unexpectedly with status {return_code}"
                    LOG.error(self.last_error)

    async def _wait_until_ready(self, process: asyncio.subprocess.Process) -> None:
        assert self.client is not None
        deadline = time.monotonic() + self.load_timeout
        last_detail = "health endpoint not ready"
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise ModelUnavailable(
                    f"vLLM exited during startup with status {process.returncode}"
                )
            try:
                response = await self.client.get(
                    f"{self.upstream_url}/health", timeout=3.0
                )
                if response.is_success:
                    return
                last_detail = f"health endpoint returned HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_detail = str(exc)
            await asyncio.sleep(1.0)
        raise ModelUnavailable(
            f"vLLM did not become ready within {self.load_timeout:g}s: {last_detail}"
        )

    async def _gpu_free_memory_mib(self) -> int | None:
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
            if process.returncode != 0:
                return None
            first_line = stdout.decode().strip().splitlines()[0]
            return int(float(first_line.strip()))
        except (OSError, ValueError, IndexError, asyncio.TimeoutError):
            LOG.warning("Could not query free VRAM; continuing without the preflight guard")
            return None

    async def _ask_idle_comfyui_to_release_models(self) -> None:
        assert self.client is not None
        base_url = self.comfyui_base_url.rstrip("/")
        try:
            queue_response = await self.client.get(f"{base_url}/queue", timeout=10.0)
            queue_response.raise_for_status()
            queue = queue_response.json()
            running = len(queue.get("queue_running", []))
            pending = len(queue.get("queue_pending", []))
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise ModelUnavailable(
                f"VRAM is low and the ComfyUI queue could not be checked: {exc}"
            ) from exc

        if running or pending:
            raise ModelUnavailable(
                "vLLM remains unloaded because ComfyUI has "
                f"{running} running and {pending} queued job(s)."
            )

        LOG.info("ComfyUI is idle; requesting release of its cached models")
        try:
            free_response = await self.client.post(
                f"{base_url}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=10.0,
            )
            free_response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailable(
                f"ComfyUI did not accept the model-release request: {exc}"
            ) from exc

    async def _ensure_vram_available(self) -> None:
        if self.min_free_vram_mib <= 0:
            return
        free_vram = await self._gpu_free_memory_mib()
        if free_vram is None or free_vram >= self.min_free_vram_mib:
            return

        if self.comfyui_base_url:
            await self._ask_idle_comfyui_to_release_models()
            deadline = time.monotonic() + self.comfyui_free_timeout
            while time.monotonic() < deadline:
                await asyncio.sleep(1.0)
                free_vram = await self._gpu_free_memory_mib()
                if free_vram is None or free_vram >= self.min_free_vram_mib:
                    LOG.info("ComfyUI released cached GPU models")
                    return

        message = (
            f"vLLM remains unloaded: GPU has {free_vram} MiB free but "
            f"requires at least {self.min_free_vram_mib} MiB. "
            "Finish the active ComfyUI job and unload its models, then retry."
        )
        raise ModelUnavailable(message)

    async def ensure_loaded(self) -> None:
        async with self.transition_lock:
            if self.process is not None and self.process.returncode is None:
                async with self.state_lock:
                    if self.state == "ready":
                        return

            if self.process is not None:
                await self._stop_process_locked("cleaning up a failed process")

            try:
                await self._ensure_vram_available()
            except ModelUnavailable as exc:
                await self._set_state("unloaded", str(exc))
                raise

            await self._set_state("loading")
            LOG.info("Starting vLLM: %s", " ".join(self.command))
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.command,
                    start_new_session=True,
                )
            except OSError as exc:
                message = f"Unable to start vLLM: {exc}"
                await self._set_state("error", message)
                raise ModelUnavailable(message) from exc

            async with self.state_lock:
                self.process = process
                self.last_error = None
            self.monitor_task = asyncio.create_task(self._monitor_process(process))

            try:
                await self._wait_until_ready(process)
            except ModelUnavailable as exc:
                await self._set_state("error", str(exc))
                await self._stop_process_locked("startup failure")
                raise

            await self._set_state("ready")
            LOG.info("vLLM is ready on the private port %d", self.upstream_port)

    async def begin_request(self) -> None:
        async with self.state_lock:
            self.inflight += 1
            self.last_activity = time.monotonic()
        try:
            await self.ensure_loaded()
        except Exception:
            await self.finish_request()
            raise

    async def finish_request(self) -> None:
        async with self.state_lock:
            self.inflight = max(0, self.inflight - 1)
            self.last_activity = time.monotonic()

    async def _stop_process_locked(self, reason: str) -> bool:
        process = self.process
        if process is None:
            await self._set_state("unloaded")
            return True

        await self._set_state("stopping")
        LOG.info("Stopping vLLM (%s)", reason)
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=self.stop_timeout)
            except asyncio.TimeoutError:
                LOG.warning(
                    "vLLM did not stop within %.0fs; sending SIGKILL",
                    self.stop_timeout,
                )
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()

        async with self.state_lock:
            if self.process is process:
                self.process = None
            self.state = "unloaded"
            self.last_error = None
        LOG.info("vLLM is fully unloaded")
        return True

    async def unload(self, reason: str, require_idle: bool = True) -> bool:
        async with self.transition_lock:
            async with self.state_lock:
                if require_idle and self.inflight > 0:
                    return False
            return await self._stop_process_locked(reason)

    async def _idle_watchdog(self) -> None:
        interval = min(5.0, max(1.0, self.idle_timeout / 10.0))
        while True:
            await asyncio.sleep(interval)
            async with self.state_lock:
                should_unload = (
                    self.state == "ready"
                    and self.inflight == 0
                    and time.monotonic() - self.last_activity >= self.idle_timeout
                )
            if should_unload:
                await self.unload(
                    f"idle for {self.idle_timeout:g}s", require_idle=True
                )

    async def snapshot(self) -> dict[str, object]:
        async with self.state_lock:
            process = self.process
            return {
                "state": self.state,
                "model_loaded": self.state == "ready",
                "pid": process.pid if process is not None else None,
                "inflight_requests": self.inflight,
                "idle_seconds": round(time.monotonic() - self.last_activity, 1),
                "idle_timeout_seconds": self.idle_timeout,
                "minimum_free_vram_mib": self.min_free_vram_mib,
                "comfyui_auto_release": bool(self.comfyui_base_url),
                "last_error": self.last_error,
            }


def create_app(controller: ModelController) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await controller.start()
        try:
            yield
        finally:
            await controller.close()

    app = FastAPI(
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def require_control_access(request: Request) -> None:
        if not controller.control_key:
            return
        expected = f"Bearer {controller.control_key}"
        if request.headers.get("authorization") != expected:
            raise HTTPException(status_code=401, detail="Invalid control API key")

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "vLLM on-demand controller",
            "openai_base_url": "/v1",
            **await controller.snapshot(),
        }

    @app.get("/health")
    async def health() -> dict[str, object]:
        # The controller is healthy while the model is intentionally unloaded.
        # Model startup errors are surfaced to /v1 requests and in the status.
        return await controller.snapshot()

    @app.get("/on-demand/status")
    async def status() -> dict[str, object]:
        return await controller.snapshot()

    @app.post("/on-demand/load")
    async def load(request: Request) -> dict[str, object]:
        require_control_access(request)
        async with controller.state_lock:
            controller.last_activity = time.monotonic()
        try:
            await controller.ensure_loaded()
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return await controller.snapshot()

    @app.post("/on-demand/unload")
    async def unload(request: Request) -> dict[str, object]:
        require_control_access(request)
        if not await controller.unload("manual API request", require_idle=True):
            raise HTTPException(
                status_code=409, detail="Cannot unload while requests are in flight"
            )
        return await controller.snapshot()

    async def proxy_openai_request(request: Request) -> Response:
        try:
            await controller.begin_request()
        except ModelUnavailable as exc:
            return JSONResponse(status_code=503, content={"error": {"message": str(exc)}})

        assert controller.client is not None
        response: httpx.Response | None = None
        try:
            body = await request.body()
            upstream = f"{controller.upstream_url}{request.url.path}"
            if request.url.query:
                upstream = f"{upstream}?{request.url.query}"
            headers = {
                key: value
                for key, value in request.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"host", "content-length"}
            }
            upstream_request = controller.client.build_request(
                request.method,
                upstream,
                headers=headers,
                content=body,
            )
            response = await controller.client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            if response is not None:
                await response.aclose()
            await controller.finish_request()
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"vLLM upstream request failed: {exc}"}},
            )
        except Exception:
            if response is not None:
                await response.aclose()
            await controller.finish_request()
            raise

        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }

        async def response_body() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await controller.finish_request()

        return StreamingResponse(
            response_body(),
            status_code=response.status_code,
            headers=response_headers,
        )

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    app.add_api_route("/v1", proxy_openai_request, methods=methods)
    app.add_api_route("/v1/{path:path}", proxy_openai_request, methods=methods)
    app.add_api_route("/docs", proxy_openai_request, methods=["GET"])
    app.add_api_route("/openapi.json", proxy_openai_request, methods=["GET"])
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument("--upstream-port", type=int, default=8001)
    parser.add_argument("--idle-timeout", type=float, default=600.0)
    parser.add_argument("--load-timeout", type=float, default=1200.0)
    parser.add_argument("--stop-timeout", type=float, default=120.0)
    parser.add_argument("--min-free-vram-mib", type=int, default=0)
    parser.add_argument("--comfyui-base-url", default="")
    parser.add_argument("--comfyui-free-timeout", type=float, default=120.0)
    parser.add_argument("vllm_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.vllm_command and args.vllm_command[0] == "--":
        args.vllm_command = args.vllm_command[1:]
    if not args.vllm_command:
        parser.error("a vLLM command is required after --")
    if args.idle_timeout <= 0 or args.load_timeout <= 0 or args.stop_timeout <= 0:
        parser.error("timeouts must be greater than zero")
    if args.min_free_vram_mib < 0:
        parser.error("minimum free VRAM must be zero or greater")
    if args.comfyui_free_timeout <= 0:
        parser.error("ComfyUI free timeout must be greater than zero")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=os.getenv("ON_DEMAND_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    controller = ModelController(
        command=args.vllm_command,
        upstream_port=args.upstream_port,
        idle_timeout=args.idle_timeout,
        load_timeout=args.load_timeout,
        stop_timeout=args.stop_timeout,
        min_free_vram_mib=args.min_free_vram_mib,
        comfyui_base_url=args.comfyui_base_url,
        comfyui_free_timeout=args.comfyui_free_timeout,
        control_key=os.getenv("ON_DEMAND_CONTROL_KEY", os.getenv("API_KEY", "")),
    )
    app = create_app(controller)
    uvicorn.run(
        app,
        host=args.listen_host,
        port=args.listen_port,
        log_level=os.getenv("ON_DEMAND_LOG_LEVEL", "info").lower(),
        proxy_headers=False,
    )


if __name__ == "__main__":
    main()
