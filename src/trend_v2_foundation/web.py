"""Dependency-free, same-origin web shell for the Foundation 4 read-only UI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit

from .api import API_PATH_PREFIX, ApiResponse, ReadOnlyTrendApi
from .canonical import canonical_bytes


WEB_UI_VERSION = "trend_v2_korean_saved_run_ui_v1"
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
    """Serve fixed packaged assets and delegate API reads to Foundation 3."""

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
    ) -> WebResponse:
        parsed = urlsplit(target)
        path = unquote(unquote(parsed.path))
        if path == API_PATH_PREFIX or path.startswith(f"{API_PATH_PREFIX}/"):
            return self._api_response(self.api.dispatch(method, target, headers=headers))
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


def build_web_server(application: TrendWebApplication) -> ThreadingHTTPServer:
    """Create the loopback server used by the API and packaged web interface."""

    api = application.api

    class Handler(BaseHTTPRequestHandler):
        server_version = "TrendV2LocalWeb/1"

        def _send(self, method: str) -> None:
            response = application.dispatch(method, self.path, headers=dict(self.headers.items()))
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
