from __future__ import annotations

import asyncio
import importlib.util
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import httpx


REPO_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_DIR / "docker" / "on_demand_server.py"
SPEC = importlib.util.spec_from_file_location("on_demand_server", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class OnDemandServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_comfy_queue_is_never_interrupted(self) -> None:
        controller = MODULE.ModelController(
            command=[sys.executable, "-c", "raise SystemExit(99)"],
            upstream_port=unused_port(),
            idle_timeout=10.0,
            load_timeout=5.0,
            stop_timeout=2.0,
            min_free_vram_mib=90_000,
            comfyui_base_url="http://comfy.test",
        )
        controller._gpu_free_memory_mib = AsyncMock(return_value=20_000)
        await controller.start()
        assert controller.client is not None
        await controller.client.aclose()

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.url.path, "/queue")
            return httpx.Response(
                200,
                json={"queue_running": [[1]], "queue_pending": []},
            )

        controller.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with self.assertRaisesRegex(MODULE.ModelUnavailable, "1 running"):
                await controller.ensure_loaded()
            self.assertIsNone(controller.process)
            self.assertEqual((await controller.snapshot())["state"], "unloaded")
        finally:
            await controller.close()

    async def test_insufficient_vram_does_not_start_model(self) -> None:
        controller = MODULE.ModelController(
            command=[sys.executable, "-c", "raise SystemExit(99)"],
            upstream_port=unused_port(),
            idle_timeout=10.0,
            load_timeout=5.0,
            stop_timeout=2.0,
            min_free_vram_mib=90_000,
        )
        controller._gpu_free_memory_mib = AsyncMock(return_value=20_000)
        await controller.start()
        try:
            with self.assertRaisesRegex(MODULE.ModelUnavailable, "20000 MiB free"):
                await controller.ensure_loaded()
            self.assertIsNone(controller.process)
            self.assertEqual((await controller.snapshot())["state"], "unloaded")
        finally:
            await controller.close()

    async def test_request_loads_then_idle_unloads_model(self) -> None:
        upstream_port = unused_port()
        controller = MODULE.ModelController(
            command=[
                sys.executable,
                str(REPO_DIR / "tests" / "fake_vllm.py"),
                str(upstream_port),
            ],
            upstream_port=upstream_port,
            idle_timeout=0.3,
            load_timeout=5.0,
            stop_timeout=2.0,
        )
        app = MODULE.create_app(controller)
        await controller.start()
        try:
            self.assertEqual((await controller.snapshot())["state"], "unloaded")
            self.assertIsNone(controller.process)

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://controller"
            ) as client:
                dashboard = await client.get("/")
                self.assertEqual(dashboard.status_code, 200)
                self.assertIn("vLLM On-Demand", dashboard.text)
                self.assertNotIn("__DEFAULT_MODEL_JSON__", dashboard.text)

                docs_alias = await client.get("/docs")
                self.assertEqual(docs_alias.status_code, 200)
                self.assertIn("Connection and models", docs_alias.text)
                self.assertEqual((await controller.snapshot())["state"], "unloaded")
                self.assertIsNone(controller.process)

                response = await client.get("/v1/models")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["data"][0]["id"], "fake-model")
                self.assertEqual((await controller.snapshot())["state"], "ready")

                upstream_docs = await client.get("/vllm/docs")
                self.assertEqual(upstream_docs.status_code, 200)
                self.assertIn("Fake vLLM Swagger", upstream_docs.text)

                for _ in range(30):
                    if (await controller.snapshot())["state"] == "unloaded":
                        break
                    await asyncio.sleep(0.1)

                status = (await client.get("/health")).json()
                self.assertEqual(status["state"], "unloaded")
                self.assertFalse(status["model_loaded"])
                self.assertIsNone(status["pid"])
        finally:
            await controller.close()


if __name__ == "__main__":
    unittest.main()
