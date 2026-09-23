from __future__ import annotations

import asyncio
import importlib.util
import socket
import sys
import tempfile
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
    async def test_cached_models_can_be_listed_and_only_unused_models_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory) / "hub"
            current = cache_root / "models--owner--current"
            unused = cache_root / "models--owner--unused-model"
            (current / "blobs").mkdir(parents=True)
            (unused / "blobs").mkdir(parents=True)
            (current / "blobs" / "weights").write_bytes(b"current")
            (unused / "blobs" / "weights").write_bytes(b"unused-weights")

            controller = MODULE.ModelController(
                command=[sys.executable, "-c", "raise SystemExit(99)"],
                upstream_port=unused_port(),
                idle_timeout=10.0,
                load_timeout=5.0,
                stop_timeout=2.0,
                model_cache_root=cache_root,
                configured_model_id="owner/current",
            )
            app = MODULE.create_app(controller)
            await controller.start()
            try:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://controller"
                ) as client:
                    listed = await client.get("/on-demand/models")
                    self.assertEqual(listed.status_code, 200)
                    self.assertIsNone(controller.process)
                    models = {
                        model["repo_id"]: model for model in listed.json()["models"]
                    }
                    self.assertEqual(set(models), {"owner/current", "owner/unused-model"})
                    self.assertTrue(models["owner/current"]["configured"])
                    self.assertFalse(models["owner/unused-model"]["configured"])
                    self.assertGreater(models["owner/unused-model"]["size_bytes"], 0)

                    bad_confirmation = await client.post(
                        "/on-demand/models/delete",
                        json={"repo_id": "owner/unused-model", "confirmation": "no"},
                    )
                    self.assertEqual(bad_confirmation.status_code, 400)
                    self.assertTrue(unused.is_dir())

                    traversal = await client.post(
                        "/on-demand/models/delete",
                        json={"repo_id": "../../outside", "confirmation": "../../outside"},
                    )
                    self.assertEqual(traversal.status_code, 404)
                    self.assertTrue(unused.is_dir())

                    configured = await client.post(
                        "/on-demand/models/delete",
                        json={
                            "repo_id": "owner/current",
                            "confirmation": "owner/current",
                        },
                    )
                    self.assertEqual(configured.status_code, 409)
                    self.assertTrue(current.is_dir())

                    deleted = await client.post(
                        "/on-demand/models/delete",
                        json={
                            "repo_id": "owner/unused-model",
                            "confirmation": "owner/unused-model",
                        },
                    )
                    self.assertEqual(deleted.status_code, 200)
                    self.assertTrue(deleted.json()["deleted"])
                    self.assertFalse(unused.exists())
                    self.assertTrue(current.is_dir())
                    self.assertIsNone(controller.process)
            finally:
                await controller.close()

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

    async def test_stop_mode_loads_then_idle_stops_model(self) -> None:
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
            idle_offload_mode="stop",
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

    async def test_level2_mode_offloads_and_wakes_same_process(self) -> None:
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
            idle_offload_mode="level2",
        )
        app = MODULE.create_app(controller)
        await controller.start()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://controller"
            ) as client:
                private_sleep = await client.post("/sleep?level=2")
                self.assertEqual(private_sleep.status_code, 404)
                self.assertIsNone(controller.process)

                first = await client.get("/v1/models")
                self.assertEqual(first.status_code, 200)
                first_status = await controller.snapshot()
                first_pid = first_status["pid"]
                self.assertEqual(first_status["state"], "ready")
                self.assertTrue(first_status["process_running"])

                for _ in range(30):
                    if (await controller.snapshot())["state"] == "offloaded":
                        break
                    await asyncio.sleep(0.1)

                offloaded = await controller.snapshot()
                self.assertEqual(offloaded["state"], "offloaded")
                self.assertFalse(offloaded["model_loaded"])
                self.assertTrue(offloaded["process_running"])
                self.assertEqual(offloaded["pid"], first_pid)
                self.assertEqual(offloaded["idle_offload_mode"], "level2")

                second = await client.get("/v1/models")
                self.assertEqual(second.status_code, 200)
                awake = await controller.snapshot()
                self.assertEqual(awake["state"], "ready")
                self.assertEqual(awake["pid"], first_pid)

                offload = await client.post("/on-demand/unload")
                self.assertEqual(offload.status_code, 200)
                self.assertEqual(offload.json()["state"], "offloaded")
                self.assertEqual(offload.json()["pid"], first_pid)

                stopped = await client.post("/on-demand/stop")
                self.assertEqual(stopped.status_code, 200)
                self.assertEqual(stopped.json()["state"], "unloaded")
                self.assertFalse(stopped.json()["process_running"])
                self.assertIsNone(stopped.json()["pid"])
        finally:
            await controller.close()


if __name__ == "__main__":
    unittest.main()
