#!/usr/bin/env python3
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = b"OK"
            content_type = "text/plain"
        elif self.path == "/v1/models":
            payload = json.dumps(
                {"object": "list", "data": [{"id": "fake-model"}]}
            ).encode()
            content_type = "application/json"
        elif self.path == "/docs":
            payload = b"<!doctype html><title>Fake vLLM Swagger</title>"
            content_type = "text/html"
        else:
            payload = b'{"detail":"not found"}'
            content_type = "application/json"
            self.send_response(404)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
    server.serve_forever()
