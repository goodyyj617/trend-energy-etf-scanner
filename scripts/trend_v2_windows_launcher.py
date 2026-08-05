"""Windows one-click lifecycle wrapper for the canonical Trend Strategy v2 server."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Mapping


IDENTITY_SCHEMA = "trend_v2_windows_launcher_identity_v1"
RUNTIME_SCHEMA = "trend_v2_windows_launcher_runtime_v1"
API_VERSION = "trend_v2_local_read_api_v1"
DEFAULT_PORT = 8765
READY_TIMEOUT_SECONDS = 60.0
STOP_TIMEOUT_SECONDS = 20.0


class LauncherError(RuntimeError):
    """A concise user-facing launcher failure."""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return dict(value) if isinstance(value, Mapping) else None
    except (OSError, ValueError, TypeError):
        return None


def _process_start_marker(pid: int) -> str | None:
    if pid <= 0:
        return None
    if os.name == "nt":
        process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return None
        try:
            exit_code = ctypes.wintypes.DWORD()
            if not ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code)) or exit_code.value != 259:
                return None
            creation = ctypes.wintypes.FILETIME()
            exit_time = ctypes.wintypes.FILETIME()
            kernel = ctypes.wintypes.FILETIME()
            user = ctypes.wintypes.FILETIME()
            if not ctypes.windll.kernel32.GetProcessTimes(
                process,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
        finally:
            ctypes.windll.kernel32.CloseHandle(process)
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        try:
            os.kill(pid, 0)
            return f"alive:{pid}"
        except OSError:
            return None


def _request_json(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    timeout: float = 1.0,
) -> dict[str, Any] | None:
    try:
        request = urllib.request.Request(url, method=method, headers=dict(headers or {}), data=b"" if method == "POST" else None)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
            return dict(value) if isinstance(value, Mapping) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _expected_service(url: str) -> bool:
    health = _request_json(f"{url.rstrip('/')}/api/v1/health")
    return bool(
        health
        and health.get("status") == "ok"
        and health.get("api_version") == API_VERSION
        and health.get("controlled_local_writes") is True
    )


def _launcher_identity(url: str) -> dict[str, Any] | None:
    value = _request_json(f"{url.rstrip('/')}/__trend_v2_launcher__/identity")
    if not value or value.get("schema_version") != IDENTITY_SCHEMA or value.get("application") != API_VERSION:
        return None
    return value


def _port_occupied(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.settimeout(0.3)
        return candidate.connect_ex(("127.0.0.1", port)) == 0


def _paths(root: Path, store: Path) -> dict[str, Path]:
    launcher = store / "launcher"
    return {
        "launcher": launcher,
        "log": launcher / "launcher.log",
        "runtime": launcher / "runtime.json",
        "token": launcher / "shutdown.token",
        "lock": launcher / "start.lock",
    }


def discover_python(root: Path, override: str | None = None) -> Path:
    candidate = Path(override).expanduser() if override else root / ".venv" / "Scripts" / "python.exe"
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise LauncherError(f"저장소 가상환경을 찾을 수 없습니다: {candidate}")
    completed = subprocess.run(
        [str(candidate), "-c", "import sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 2)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise LauncherError(f"저장소 가상환경의 Python을 사용할 수 없습니다: {candidate}")
    return candidate


def build_server_command(root: Path, python: Path, store: Path, port: int, instance_id: str, token_file: Path) -> list[str]:
    return [
        str(python),
        str(root / "scripts" / "run_trend_v2_web.py"),
        "start",
        "--store",
        str(store),
        "--port",
        str(port),
        "--launcher-instance-id",
        instance_id,
        "--launcher-token-file",
        str(token_file),
    ]


def _run_logged(command: list[str], root: Path, log: Path) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    completed = subprocess.run(command, cwd=root, env=environment, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command[:3])} ...\n{completed.stdout}{completed.stderr}")
    return completed


def _cleanup_runtime(paths: Mapping[str, Path]) -> None:
    for key in ("runtime", "token"):
        try:
            paths[key].unlink()
        except FileNotFoundError:
            pass


def _terminate_failed_start(process: subprocess.Popen[bytes], state: Mapping[str, Any], paths: Mapping[str, Path]) -> bool:
    """Terminate only the freshly launched, creation-marker-verified process tree."""
    if _process_start_marker(process.pid) != state.get("process_start_marker"):
        return False
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    else:
        process.terminate()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _process_start_marker(process.pid) is None:
            _cleanup_runtime(paths)
            return True
        time.sleep(0.05)
    return False


def _validated_runtime(paths: Mapping[str, Path], root: Path, store: Path) -> tuple[dict[str, Any] | None, bool]:
    state = _read_json(paths["runtime"])
    if state is None:
        return None, False
    try:
        valid_scope = (
            state.get("schema_version") == RUNTIME_SCHEMA
            and Path(state["repository_root"]).resolve() == root.resolve()
            and Path(state["store_root"]).resolve() == store.resolve()
        )
        marker_matches = _process_start_marker(int(state["pid"])) == state.get("process_start_marker")
    except (KeyError, TypeError, ValueError, OSError):
        valid_scope = marker_matches = False
    return state, bool(valid_scope and marker_matches)


def _start_owned(
    root: Path,
    store: Path,
    port: int,
    python: Path,
    *,
    browser_open: Callable[[str], Any] = webbrowser.open,
    ready_timeout: float = READY_TIMEOUT_SECONDS,
) -> int:
    paths = _paths(root, store)
    paths["launcher"].mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{port}/"
    if _expected_service(url):
        browser_open(url)
        print(f"[완료] 실행 중인 Trend Strategy v2를 재사용했습니다: {url}")
        return 0
    if _port_occupied(port):
        raise LauncherError(f"포트 {port}을(를) 다른 프로그램이 사용 중입니다. 해당 프로그램은 종료하지 않았습니다.")
    state, verified_process = _validated_runtime(paths, root, store)
    if state is not None and not verified_process:
        _cleanup_runtime(paths)

    if not (store / "retention_policy.json").is_file():
        initialized = _run_logged(
            [str(python), str(root / "scripts" / "run_trend_v2_web.py"), "init", "--store", str(store)],
            root,
            paths["log"],
        )
        if initialized.returncode:
            raise LauncherError("ResultStore 자동 초기화에 실패했습니다.")

    preflight = _run_logged(
        [str(python), str(root / "scripts" / "run_trend_v2_web.py"), "preflight", "--store", str(store), "--port", str(port)],
        root,
        paths["log"],
    )
    if preflight.returncode:
        if preflight.stdout.strip():
            print(preflight.stdout.strip())
        raise LauncherError("사전 점검의 차단 항목을 해결한 뒤 다시 시작하세요.")

    instance_id = secrets.token_urlsafe(24)
    token = secrets.token_urlsafe(48)
    _atomic_write(paths["token"], token.encode("utf-8"))
    command = build_server_command(root, python, store, port, instance_id, paths["token"])
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    log_handle = paths["log"].open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
    finally:
        log_handle.close()
    marker = _process_start_marker(process.pid)
    if marker is None:
        process.terminate()
        _cleanup_runtime(paths)
        raise LauncherError("시작한 서버 프로세스의 소유권을 확인할 수 없습니다.")
    state = {
        "schema_version": RUNTIME_SCHEMA,
        "repository_root": str(root.resolve()),
        "store_root": str(store.resolve()),
        "port": port,
        "url": url,
        "pid": process.pid,
        "process_start_marker": marker,
        "bootstrap_pid": process.pid,
        "bootstrap_process_start_marker": marker,
        "instance_id": instance_id,
        "status": "starting",
    }
    _write_json(paths["runtime"], state)
    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _cleanup_runtime(paths)
            raise LauncherError("서버 프로세스가 준비되기 전에 종료되었습니다.")
        identity = _launcher_identity(url)
        if identity and identity.get("instance_id") == instance_id:
            try:
                server_pid = int(identity["pid"])
            except (KeyError, TypeError, ValueError):
                time.sleep(0.1)
                continue
            server_marker = _process_start_marker(server_pid)
            if server_marker is None:
                time.sleep(0.1)
                continue
            state["pid"] = server_pid
            state["process_start_marker"] = server_marker
            state["status"] = "running"
            _write_json(paths["runtime"], state)
            browser_open(url)
            print(f"[완료] Trend Strategy v2를 시작하고 브라우저를 열었습니다: {url}")
            return 0
        time.sleep(0.1)
    if _terminate_failed_start(process, state, paths):
        raise LauncherError("서버 준비 확인 시간이 초과되어 확인된 시작 프로세스를 정리했습니다.")
    state["status"] = "startup_failed_process_alive"
    _write_json(paths["runtime"], state)
    raise LauncherError("서버 준비 확인 시간이 초과되었고 프로세스 정리를 확인할 수 없습니다. 런처 상태와 로그를 보존했습니다.")


def start(
    root: Path,
    store: Path,
    port: int,
    python: Path,
    *,
    browser_open: Callable[[str], Any] = webbrowser.open,
    ready_timeout: float = READY_TIMEOUT_SECONDS,
) -> int:
    paths = _paths(root, store)
    paths["launcher"].mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        deadline = time.monotonic() + ready_timeout
        url = f"http://127.0.0.1:{port}/"
        while time.monotonic() < deadline:
            if _expected_service(url):
                browser_open(url)
                print(f"[완료] 시작 중이던 Trend Strategy v2를 재사용했습니다: {url}")
                return 0
            if not paths["lock"].exists():
                return start(root, store, port, python, browser_open=browser_open, ready_timeout=max(1.0, deadline - time.monotonic()))
            time.sleep(0.1)
        try:
            stale = time.time() - paths["lock"].stat().st_mtime > READY_TIMEOUT_SECONDS + 10
        except OSError:
            stale = False
        if stale:
            paths["lock"].unlink(missing_ok=True)
            return start(root, store, port, python, browser_open=browser_open, ready_timeout=ready_timeout)
        raise LauncherError("다른 시작 요청이 서버 준비를 기다리는 중입니다. 잠시 후 다시 실행하세요.")
    else:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    try:
        return _start_owned(root, store, port, python, browser_open=browser_open, ready_timeout=ready_timeout)
    finally:
        paths["lock"].unlink(missing_ok=True)


def stop(root: Path, store: Path, *, stop_timeout: float = STOP_TIMEOUT_SECONDS) -> int:
    paths = _paths(root, store)
    state, verified_process = _validated_runtime(paths, root, store)
    if state is None:
        print("[완료] 이 런처가 시작한 실행 중 서버가 없습니다.")
        return 0
    if not verified_process:
        _cleanup_runtime(paths)
        print("[완료] 오래된 런처 상태를 정리했습니다. 경제 작업 완료로 간주하지 않습니다.")
        return 0
    url = str(state.get("url", ""))
    identity = _launcher_identity(url)
    if not identity or identity.get("instance_id") != state.get("instance_id") or identity.get("pid") != state.get("pid"):
        raise LauncherError("기록된 프로세스가 Trend Strategy v2 런처 소유 서버인지 확인할 수 없어 종료하지 않았습니다.")
    try:
        token = paths["token"].read_text(encoding="utf-8").strip()
    except OSError as error:
        raise LauncherError("안전한 종료 토큰을 읽을 수 없어 서버를 종료하지 않았습니다.") from error
    response = _request_json(
        f"{url.rstrip('/')}/__trend_v2_launcher__/shutdown",
        method="POST",
        headers={"X-Trend-V2-Shutdown-Token": token},
    )
    if not response or response.get("status") != "shutdown_requested":
        raise LauncherError("서버가 안전한 종료 요청을 수락하지 않았습니다.")
    deadline = time.monotonic() + stop_timeout
    process_identities = {
        (int(state["pid"]), str(state["process_start_marker"])),
        (
            int(state.get("bootstrap_pid", state["pid"])),
            str(state.get("bootstrap_process_start_marker", state["process_start_marker"])),
        ),
    }
    while time.monotonic() < deadline:
        if all(_process_start_marker(pid) != marker for pid, marker in process_identities):
            _cleanup_runtime(paths)
            print("[완료] Trend Strategy v2 서버를 정상 종료했습니다.")
            return 0
        time.sleep(0.1)
    raise LauncherError("서버 종료 대기 시간이 초과되었습니다.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trend Strategy v2 Windows 브라우저 런처")
    parser.add_argument("command", choices=("start", "stop"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--store", type=Path)
    parser.add_argument("--port", type=int, default=int(os.environ.get("TREND_V2_PORT", DEFAULT_PORT)))
    parser.add_argument("--python", dest="python_override", default=os.environ.get("TREND_V2_PYTHON"))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    store_value = args.store or Path(os.environ.get("TREND_V2_STORE", ".trend_v2_store"))
    store = (root / store_value).resolve() if not store_value.is_absolute() else store_value.resolve()
    paths = _paths(root, store)
    paths["launcher"].mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "start":
            python = discover_python(root, args.python_override)
            return start(root, store, args.port, python)
        return stop(root, store)
    except (LauncherError, OSError, subprocess.SubprocessError) as error:
        with paths["log"].open("a", encoding="utf-8") as handle:
            handle.write(f"\n[런처 오류] {error}\n")
        print(f"[오류] {error}")
        print(f"상세 로그: {paths['log']}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
