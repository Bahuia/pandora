import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from models.registry import ModelRegistry


class _Handler(BaseHTTPRequestHandler):
    authorization = None
    model = None

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).authorization = self.headers.get("Authorization")
        type(self).model = payload["model"]
        response = json.dumps({"choices": [{"message": {"content": "ready"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *_args):
        return


def test_openai_compatible_endpoint_round_trip():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = ModelRegistry.create(
            "local-model",
            provider="openai-compatible",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="test-token",
        )
        assert client.generate("hello") == "ready"
        assert _Handler.authorization == "Bearer test-token"
        assert _Handler.model == "local-model"
    finally:
        server.shutdown()
        server.server_close()


def test_openai_compatible_endpoint_requires_base_url():
    with pytest.raises(ValueError, match="base-url"):
        ModelRegistry.create("local-model", provider="openai-compatible")
