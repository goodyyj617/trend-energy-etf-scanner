"""Focused deterministic coverage for the Windows one-click launcher."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.trend_v2_foundation.local_operability import initialize_result_store
from src.trend_v2_foundation.web import build_web_server


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("trend_v2_windows_launcher", ROOT / "scripts" / "trend_v2_windows_launcher.py")
assert SPEC and SPEC.loader
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class _Process:
    pid = 4321

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False

    def poll(self): return self.returncode
    def terminate(self): self.terminated = True
    def wait(self, timeout=None): return 0


class WindowsLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "한국어 경로 (테스트)"
        self.root.mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "run_trend_v2_web.py").write_text("", encoding="utf-8")
        self.store = self.root / ".trend_v2_store"
        self.python = self.root / ".venv" / "Scripts" / "python.exe"
        self.python.parent.mkdir(parents=True)
        self.python.write_bytes(b"python")

    def tearDown(self): self.temporary.cleanup()

    def test_command_preserves_spaces_korean_and_parentheses(self):
        command = launcher.build_server_command(self.root, self.python, self.store, 9876, "instance", self.store / "launcher" / "shutdown.token")
        self.assertEqual(command[0], str(self.python))
        self.assertEqual(command[1], str(self.root / "scripts" / "run_trend_v2_web.py"))
        self.assertEqual(command[command.index("--store") + 1], str(self.store))
        self.assertNotIn('"', command[0])

    def test_virtual_environment_discovery_and_missing_environment(self):
        completed = subprocess.CompletedProcess([], 0)
        with patch.object(launcher.subprocess, "run", return_value=completed) as run:
            self.assertEqual(launcher.discover_python(self.root), self.python.resolve())
        self.assertIn("sys.prefix", run.call_args.args[0][2])
        self.python.unlink()
        with self.assertRaisesRegex(launcher.LauncherError, "가상환경"):
            launcher.discover_python(self.root)

    def test_launcher_operational_log_does_not_block_first_init(self):
        log = self.store / "launcher" / "launcher.log"
        log.parent.mkdir(parents=True)
        log.write_text("missing venv", encoding="utf-8")
        self.assertTrue(initialize_result_store(self.store))
        self.assertTrue((self.store / "retention_policy.json").is_file())

    def test_preflight_block_prevents_server_and_browser(self):
        blocked = subprocess.CompletedProcess([], 1, stdout="[차단] 점검 실패\n", stderr="")
        browser = Mock()
        with patch.object(launcher, "_expected_service", return_value=False), patch.object(launcher, "_port_occupied", return_value=False), patch.object(launcher, "_run_logged", side_effect=[subprocess.CompletedProcess([], 0, "", ""), blocked]), patch.object(launcher.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(launcher.LauncherError, "사전 점검"):
                launcher._start_owned(self.root, self.store, 9876, self.python, browser_open=browser)
        popen.assert_not_called()
        browser.assert_not_called()

    def test_readiness_is_confirmed_before_browser_open(self):
        (self.store / "retention_policy.json").parent.mkdir(parents=True)
        (self.store / "retention_policy.json").write_text("{}", encoding="utf-8")
        process = _Process()
        browser = Mock()
        identity = {"schema_version": launcher.IDENTITY_SCHEMA, "application": launcher.API_VERSION, "instance_id": "ignored", "pid": 9876}
        with patch.object(launcher, "_expected_service", return_value=False), patch.object(launcher, "_port_occupied", return_value=False), patch.object(launcher, "_run_logged", return_value=subprocess.CompletedProcess([], 0, "", "")), patch.object(launcher.subprocess, "Popen", return_value=process), patch.object(launcher, "_process_start_marker", return_value="marker"), patch.object(launcher.secrets, "token_urlsafe", side_effect=["ignored", "secret-token-with-more-than-thirty-two-characters"]), patch.object(launcher, "_launcher_identity", return_value=identity):
            self.assertEqual(launcher._start_owned(self.root, self.store, 9876, self.python, browser_open=browser), 0)
        browser.assert_called_once_with("http://127.0.0.1:9876/")
        state = json.loads((self.store / "launcher" / "runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "running")
        self.assertEqual(state["pid"], 9876)
        self.assertEqual(state["bootstrap_pid"], process.pid)

    def test_browser_does_not_open_when_child_exits(self):
        (self.store / "retention_policy.json").parent.mkdir(parents=True)
        (self.store / "retention_policy.json").write_text("{}", encoding="utf-8")
        browser = Mock()
        with patch.object(launcher, "_expected_service", return_value=False), patch.object(launcher, "_port_occupied", return_value=False), patch.object(launcher, "_run_logged", return_value=subprocess.CompletedProcess([], 0, "", "")), patch.object(launcher.subprocess, "Popen", return_value=_Process(1)), patch.object(launcher, "_process_start_marker", return_value="marker"):
            with self.assertRaisesRegex(launcher.LauncherError, "준비되기 전에"):
                launcher._start_owned(self.root, self.store, 9876, self.python, browser_open=browser)
        browser.assert_not_called()
        self.assertFalse((self.store / "launcher" / "runtime.json").exists())

    def test_expected_service_reused_and_unrelated_port_refused(self):
        browser = Mock()
        with patch.object(launcher, "_expected_service", return_value=True):
            self.assertEqual(launcher._start_owned(self.root, self.store, 9876, self.python, browser_open=browser), 0)
        browser.assert_called_once()
        with patch.object(launcher, "_expected_service", return_value=False), patch.object(launcher, "_port_occupied", return_value=True):
            with self.assertRaisesRegex(launcher.LauncherError, "다른 프로그램"):
                launcher._start_owned(self.root, self.store, 9876, self.python)

    def test_duplicate_start_waits_for_and_reuses_service(self):
        paths = launcher._paths(self.root, self.store)
        paths["launcher"].mkdir(parents=True)
        paths["lock"].write_text("other", encoding="utf-8")
        browser = Mock()
        with patch.object(launcher, "_expected_service", return_value=True):
            self.assertEqual(launcher.start(self.root, self.store, 9876, self.python, browser_open=browser), 0)
        browser.assert_called_once()

    def test_runtime_state_is_atomic(self):
        target = self.store / "launcher" / "runtime.json"
        with patch.object(launcher.os, "replace", wraps=launcher.os.replace) as replace:
            launcher._write_json(target, {"schema_version": launcher.RUNTIME_SCHEMA})
        replace.assert_called_once()
        self.assertEqual(launcher._read_json(target)["schema_version"], launcher.RUNTIME_SCHEMA)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_stop_without_owned_server_and_stale_state_are_safe(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(launcher.stop(self.root, self.store), 0)
        self.assertIn("실행 중 서버가 없습니다", output.getvalue())
        paths = launcher._paths(self.root, self.store)
        launcher._write_json(paths["runtime"], {"schema_version": launcher.RUNTIME_SCHEMA, "repository_root": str(self.root), "store_root": str(self.store), "pid": 1, "process_start_marker": "old"})
        paths["token"].write_text("secret", encoding="utf-8")
        with patch.object(launcher, "_process_start_marker", return_value=None), redirect_stdout(output):
            self.assertEqual(launcher.stop(self.root, self.store), 0)
        self.assertFalse(paths["runtime"].exists())
        self.assertIn("경제 작업 완료로 간주하지 않습니다", output.getvalue())

    def test_stop_refuses_unverified_process(self):
        paths = launcher._paths(self.root, self.store)
        launcher._write_json(paths["runtime"], {"schema_version": launcher.RUNTIME_SCHEMA, "repository_root": str(self.root), "store_root": str(self.store), "pid": 4321, "process_start_marker": "marker", "url": "http://127.0.0.1:9876/", "instance_id": "mine"})
        with patch.object(launcher, "_process_start_marker", return_value="marker"), patch.object(launcher, "_launcher_identity", return_value=None):
            with self.assertRaisesRegex(launcher.LauncherError, "확인할 수 없어"):
                launcher.stop(self.root, self.store)
        self.assertTrue(paths["runtime"].exists())

    def test_verified_stop_uses_authenticated_graceful_endpoint(self):
        paths = launcher._paths(self.root, self.store)
        launcher._write_json(paths["runtime"], {"schema_version": launcher.RUNTIME_SCHEMA, "repository_root": str(self.root), "store_root": str(self.store), "pid": 4321, "process_start_marker": "marker", "url": "http://127.0.0.1:9876/", "instance_id": "mine"})
        paths["token"].write_text("secret-token", encoding="utf-8")
        identity = {"instance_id": "mine", "pid": 4321}
        with patch.object(launcher, "_process_start_marker", side_effect=["marker", None]), patch.object(launcher, "_launcher_identity", return_value=identity), patch.object(launcher, "_request_json", return_value={"status": "shutdown_requested"}) as request:
            self.assertEqual(launcher.stop(self.root, self.store), 0)
        self.assertEqual(request.call_args.kwargs["headers"]["X-Trend-V2-Shutdown-Token"], "secret-token")
        self.assertFalse(paths["runtime"].exists())

    def test_error_reports_deterministic_log_path(self):
        output = io.StringIO()
        self.python.unlink()
        with redirect_stdout(output):
            self.assertEqual(launcher.main(["start", "--root", str(self.root)]), 1)
        expected = self.store / "launcher" / "launcher.log"
        self.assertIn(str(expected), output.getvalue())
        self.assertTrue(expected.is_file())

    def test_cmd_files_quote_root_and_python(self):
        for name in ("Trend Strategy V2 시작.cmd", "Trend Strategy V2 종료.cmd"):
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn('set "ROOT=%~dp0"', source)
            self.assertIn('"%TREND_V2_PYTHON%" "%ROOT%scripts\\trend_v2_windows_launcher.py"', source)
            self.assertIn('set "EXIT_CODE=%ERRORLEVEL%"', source)


class LauncherControlServerTests(unittest.TestCase):
    def test_authenticated_shutdown_stops_server(self):
        api = SimpleNamespace(server_config=SimpleNamespace(host="127.0.0.1", port=0))
        application = SimpleNamespace(api=api)
        server = build_web_server(application, launcher_instance_id="instance", launcher_shutdown_token="secret-token-value-with-at-least-32-characters")
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            identity = json.loads(urllib.request.urlopen(base + "/__trend_v2_launcher__/identity", timeout=2).read())
            self.assertEqual(identity["instance_id"], "instance")
            request = urllib.request.Request(base + "/__trend_v2_launcher__/shutdown", method="POST", data=b"", headers={"X-Trend-V2-Shutdown-Token": "secret-token-value-with-at-least-32-characters"})
            self.assertEqual(urllib.request.urlopen(request, timeout=2).status, 202)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
