"""Dependency-free, same-origin web shell for the controlled local UI."""

from __future__ import annotations

import json
import hmac
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .api import API_PATH_PREFIX, API_VERSION, ApiResponse, ReadOnlyTrendApi
from .canonical import canonical_bytes


WEB_UI_VERSION = "trend_v2_korean_controlled_strategy_ui_v1"
_ASSET_ROOT = Path(__file__).with_name("ui_assets")
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
}
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


@dataclass(frozen=True)
class WebResponse:
    status_code: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


class TrendWebApplication:
    """Serve fixed packaged assets and delegate bounded API operations."""

    def __init__(self, api: ReadOnlyTrendApi, *, asset_root: Path | None = None) -> None:
        self.api = api
        self.asset_root = asset_root or _ASSET_ROOT

    @staticmethod
    def _api_response(response: ApiResponse) -> WebResponse:
        return WebResponse(
            response.status_code,
            canonical_bytes(response.body),
            {**response.headers, **_SECURITY_HEADERS},
        )

    @staticmethod
    def _error(status: int, message_ko: str) -> WebResponse:
        payload = canonical_bytes(
            {
                "error": {
                    "code": "web_resource_not_found" if status == 404 else "method_not_allowed",
                    "message_ko": message_ko,
                }
            }
        )
        return WebResponse(
            status,
            payload,
            {"Content-Type": "application/json; charset=utf-8", **_SECURITY_HEADERS},
        )

    def dispatch(
        self,
        method: str,
        target: str,
        *,
        headers: Mapping[str, str] | None = None,
        body: bytes | Mapping[str, Any] | None = None,
    ) -> WebResponse:
        parsed = urlsplit(target)
        path = unquote(unquote(parsed.path))
        if path == API_PATH_PREFIX or path.startswith(f"{API_PATH_PREFIX}/"):
            return self._api_response(self.api.dispatch(method, target, headers=headers, body=body))
        if method.upper() not in {"GET", "HEAD"}:
            return self._error(405, "이 로컬 화면은 저장된 결과 읽기만 지원합니다.")
        if "\\" in path or "\x00" in path or any(
            part in {".", ".."} or ".." in part for part in path.split("/")
        ):
            return self._error(404, "요청한 화면 리소스를 찾을 수 없습니다.")
        asset = _ASSETS.get(path)
        if asset is None:
            return self._error(404, "요청한 화면 리소스를 찾을 수 없습니다.")
        filename, content_type = asset
        try:
            payload = (self.asset_root / filename).read_bytes()
        except OSError:
            return self._error(404, "화면 리소스를 읽을 수 없습니다.")
        if method.upper() == "HEAD":
            payload = b""
        return WebResponse(
            200,
            payload,
            {"Content-Type": content_type, **_SECURITY_HEADERS},
        )


def build_web_server(
    application: TrendWebApplication,
    *,
    launcher_instance_id: str | None = None,
    launcher_shutdown_token: str | None = None,
) -> ThreadingHTTPServer:
    """Create the loopback server used by the API and packaged web interface."""

    api = application.api

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrendV2LocalWeb/1"

        def _send_launcher_control(self, method: str) -> bool:
            if launcher_instance_id is None or launcher_shutdown_token is None:
                return False
            path = urlsplit(self.path).path
            if path == "/__trend_v2_launcher__/identity" and method == "GET":
                body = canonical_bytes(
                    {
                        "schema_version": "trend_v2_windows_launcher_identity_v1",
                        "application": API_VERSION,
                        "instance_id": launcher_instance_id,
                        "pid": os.getpid(),
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                for key, value in _SECURITY_HEADERS.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return True
            if path == "/__trend_v2_launcher__/shutdown" and method == "POST":
                supplied = self.headers.get("X-Trend-V2-Shutdown-Token", "")
                if not hmac.compare_digest(supplied, launcher_shutdown_token):
                    self.send_error(403)
                    return True
                body = canonical_bytes({"status": "shutdown_requested"})
                self.send_response(202)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                for key, value in _SECURITY_HEADERS.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return True
            return False

        def _send(self, method: str) -> None:
            if self._send_launcher_control(method):
                return
            payload = b""
            if method == "POST":
                try:
                    declared = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    declared = -1
                maximum = (
                    api.controlled_execution_service.policy.maximum_json_body_bytes
                    if api.controlled_execution_service is not None
                    else int(api.robustness_execution_service.policy.document["maximum_json_body_bytes"])
                    if api.robustness_execution_service is not None
                    else 65_536
                )
                payload = self.rfile.read(min(max(declared, 0), maximum + 1))
                if declared < 0 or declared > maximum:
                    payload = b"x" * (maximum + 1)
            response = application.dispatch(
                method,
                self.path,
                headers=dict(self.headers.items()),
                body=payload,
            )
            self.send_response(response.status_code)
            for key, value in response.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response.body)))
            self.end_headers()
            if method != "HEAD":
                self.wfile.write(response.body)

        def do_GET(self) -> None:  # noqa: N802
            self._send("GET")

        def do_HEAD(self) -> None:  # noqa: N802
            self._send("HEAD")

        def do_POST(self) -> None:  # noqa: N802
            self._send("POST")

        def do_PUT(self) -> None:  # noqa: N802
            self._send("PUT")

        def do_DELETE(self) -> None:  # noqa: N802
            self._send("DELETE")

        def do_PATCH(self) -> None:  # noqa: N802
            self._send("PATCH")

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return ThreadingHTTPServer(
        (api.server_config.host, api.server_config.port),
        Handler,
    )


def load_retention_policy(store_root: str | Path) -> Mapping[str, Any]:
    """Read the persisted policy needed to open an existing local store."""

    source = Path(store_root) / "retention_policy.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("retention_policy.json does not contain a policy mapping")
    return policy
