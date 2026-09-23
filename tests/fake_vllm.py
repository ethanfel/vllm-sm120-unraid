#!/usr/bin/env python3
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class Handler(BaseHTTPRequestHandler):
    sleeping = False
    weights_reloaded = False

    def log_message(self, format: str, *args: object) -> None:
        pass

    def send_payload(
        self, status: int, payload: bytes, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            payload = b"OK"
            content_type = "text/plain"
        elif path == "/is_sleeping":
            payload = json.dumps({"is_sleeping": self.sleeping}).encode()
            content_type = "application/json"
        elif path == "/v1/models" and not self.sleeping:
            payload = json.dumps(
                {"object": "list", "data": [{"id": "fake-model"}]}
            ).encode()
            content_type = "application/json"
        elif path == "/docs":
            payload = b"<!doctype html><title>Fake vLLM Swagger</title>"
            content_type = "text/html"
        else:
            payload = b'{"detail":"not found"}'
            self.send_payload(404, payload)
            return

        self.send_payload(200, payload, content_type)

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/sleep" and query.get("level") == ["2"]:
            type(self).sleeping = True
            type(self).weights_reloaded = False
        elif parsed.path == "/wake_up" and query.get("tags") == ["weights"]:
            pass
        elif parsed.path == "/collective_rpc":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if body.get("method") != "reload_weights":
                self.send_payload(400, b'{"detail":"bad method"}')
                return
            type(self).weights_reloaded = True
        elif parsed.path == "/wake_up" and query.get("tags") == ["kv_cache"]:
            if not self.weights_reloaded:
                self.send_payload(409, b'{"detail":"weights not reloaded"}')
                return
            type(self).sleeping = False
        else:
            self.send_payload(404, b'{"detail":"not found"}')
            return
        self.send_payload(200, b'{"status":"ok"}')


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
    server.serve_forever()
