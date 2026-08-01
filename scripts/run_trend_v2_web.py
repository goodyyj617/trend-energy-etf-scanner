"""Start the loopback-only Trend Strategy v2 saved-result UI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trend_v2_foundation import (  # noqa: E402
    ApiServerConfig,
    ArtifactRetentionPolicy,
    LocalResultStore,
    ReadOnlyTrendApi,
    TrendWebApplication,
    build_web_server,
    load_retention_policy,
    load_terminology_source,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 Trend Strategy v2 결과 UI")
    parser.add_argument("--store", required=True, type=Path, help="기존 로컬 ResultStore 디렉터리")
    parser.add_argument("--port", type=int, default=8765, help="루프백 포트 (기본 8765)")
    args = parser.parse_args()

    store_root = args.store.resolve()
    policy = ArtifactRetentionPolicy.from_dict(load_retention_policy(store_root))
    store = LocalResultStore(store_root, policy)
    terminology = load_terminology_source(ROOT / "config" / "trend_v2" / "terminology_ko.json")
    api = ReadOnlyTrendApi(
        store,
        terminology_source=terminology,
        server_config=ApiServerConfig(port=args.port),
    )
    server = build_web_server(TrendWebApplication(api))
    address = f"http://{server.server_address[0]}:{server.server_address[1]}/"
    print(f"Trend Strategy v2 저장 결과 UI: {address}")
    print("읽기 전용입니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
