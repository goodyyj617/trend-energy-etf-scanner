from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


RERUN_GUIDANCE = (
    "Rerun the workflow from the latest main; stale generated outputs must not be "
    "published."
)


class PublishBaseError(RuntimeError):
    pass


def _run_git(
    arguments: Sequence[str],
    *,
    cwd: str | Path | None = None,
) -> str:
    command = ["git", *arguments]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        message = (
            f"{' '.join(command)} failed with exit code {exc.returncode}"
            + (f": {detail}" if detail else "")
        )
        raise PublishBaseError(f"{message}. {RERUN_GUIDANCE}") from exc
    return completed.stdout.strip()


def verify_data_publish_base(
    source_sha: str,
    *,
    remote: str = "origin",
    branch: str = "main",
    cwd: str | Path | None = None,
) -> tuple[str, str]:
    expected_sha = source_sha.strip()
    if not expected_sha:
        raise PublishBaseError(f"source SHA must not be empty. {RERUN_GUIDANCE}")

    print(f"generated_data_source_sha={expected_sha}")
    local_head = _run_git(["rev-parse", "HEAD"], cwd=cwd)
    if local_head != expected_sha:
        raise PublishBaseError(
            "Local HEAD no longer matches the calculation source: "
            f"expected source SHA {expected_sha}, observed local HEAD {local_head}. "
            f"{RERUN_GUIDANCE}"
        )

    _run_git(["fetch", "--no-tags", remote, branch], cwd=cwd)
    fetched_remote_sha = _run_git(["rev-parse", "FETCH_HEAD"], cwd=cwd)
    print(f"fetched_remote_{branch}_sha={fetched_remote_sha}")
    if fetched_remote_sha != expected_sha:
        raise PublishBaseError(
            f"Remote {remote}/{branch} no longer matches the calculation source: "
            f"expected source SHA {expected_sha}, observed remote SHA "
            f"{fetched_remote_sha}. {RERUN_GUIDANCE}"
        )

    return local_head, fetched_remote_sha


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless generated data still targets its source commit."
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        verify_data_publish_base(
            args.source_sha,
            remote=args.remote,
            branch=args.branch,
        )
    except PublishBaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
