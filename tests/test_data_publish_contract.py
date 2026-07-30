import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import verify_data_publish_base as publish_guard


ROOT = Path(__file__).parents[1]


class TemporaryGitRepository:
    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)
        self.remote = self.root / "remote.git"
        self.repository = self.root / "repository"

        self.run(
            ["git", "init", "--bare", "--initial-branch=main", str(self.remote)],
            cwd=self.root,
        )
        self.run(["git", "init", "-b", "main", str(self.repository)], cwd=self.root)
        self.configure_identity(self.repository)
        (self.repository / "source.txt").write_text("source\n", encoding="utf-8")
        self.run(["git", "add", "source.txt"], cwd=self.repository)
        self.run(["git", "commit", "-m", "Source revision"], cwd=self.repository)
        self.run(
            ["git", "remote", "add", "origin", str(self.remote)],
            cwd=self.repository,
        )
        self.run(["git", "push", "-u", "origin", "main"], cwd=self.repository)
        self.source_sha = self.git_output(["rev-parse", "HEAD"], cwd=self.repository)

    def close(self) -> None:
        self._temporary_directory.cleanup()

    @staticmethod
    def run(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def git_output(cls, arguments: list[str], *, cwd: Path) -> str:
        return cls.run(["git", *arguments], cwd=cwd).stdout.strip()

    @classmethod
    def configure_identity(cls, repository: Path) -> None:
        cls.run(["git", "config", "user.name", "fixture"], cwd=repository)
        cls.run(
            ["git", "config", "user.email", "fixture@example.com"],
            cwd=repository,
        )

    def advance_remote(self) -> str:
        updater = self.root / "updater"
        self.run(["git", "clone", str(self.remote), str(updater)], cwd=self.root)
        self.configure_identity(updater)
        (updater / "remote.txt").write_text("new remote revision\n", encoding="utf-8")
        self.run(["git", "add", "remote.txt"], cwd=updater)
        self.run(["git", "commit", "-m", "Advance remote"], cwd=updater)
        self.run(["git", "push", "origin", "main"], cwd=updater)
        return self.git_output(["rev-parse", "HEAD"], cwd=updater)


class PublishBaseGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = TemporaryGitRepository()

    def tearDown(self) -> None:
        self.fixture.close()

    def test_matching_head_source_and_remote_succeeds(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            local_sha, remote_sha = publish_guard.verify_data_publish_base(
                self.fixture.source_sha,
                cwd=self.fixture.repository,
            )

        self.assertEqual(local_sha, self.fixture.source_sha)
        self.assertEqual(remote_sha, self.fixture.source_sha)
        self.assertIn(
            f"generated_data_source_sha={self.fixture.source_sha}",
            output.getvalue(),
        )
        self.assertIn(
            f"fetched_remote_main_sha={self.fixture.source_sha}",
            output.getvalue(),
        )

    def test_local_head_mismatch_fails_with_expected_and_observed_sha(self) -> None:
        (self.fixture.repository / "local.txt").write_text(
            "local revision\n",
            encoding="utf-8",
        )
        self.fixture.run(["git", "add", "local.txt"], cwd=self.fixture.repository)
        self.fixture.run(
            ["git", "commit", "-m", "Advance local"],
            cwd=self.fixture.repository,
        )
        observed_sha = self.fixture.git_output(
            ["rev-parse", "HEAD"],
            cwd=self.fixture.repository,
        )

        with self.assertRaises(publish_guard.PublishBaseError) as raised:
            publish_guard.verify_data_publish_base(
                self.fixture.source_sha,
                cwd=self.fixture.repository,
            )

        message = str(raised.exception)
        self.assertIn(self.fixture.source_sha, message)
        self.assertIn(observed_sha, message)
        self.assertIn("Rerun the workflow from the latest main", message)

    def test_remote_mismatch_fails_with_expected_and_observed_sha(self) -> None:
        observed_sha = self.fixture.advance_remote()

        with self.assertRaises(publish_guard.PublishBaseError) as raised:
            publish_guard.verify_data_publish_base(
                self.fixture.source_sha,
                cwd=self.fixture.repository,
            )

        message = str(raised.exception)
        self.assertIn(self.fixture.source_sha, message)
        self.assertIn(observed_sha, message)
        self.assertIn("Rerun the workflow from the latest main", message)

    def test_staged_generated_files_do_not_prevent_verification(self) -> None:
        generated = self.fixture.repository / "generated.csv"
        generated.write_text("value\n1\n", encoding="utf-8")
        self.fixture.run(["git", "add", "generated.csv"], cwd=self.fixture.repository)

        publish_guard.verify_data_publish_base(
            self.fixture.source_sha,
            cwd=self.fixture.repository,
        )

        staged = self.fixture.git_output(
            ["diff", "--cached", "--name-only"],
            cwd=self.fixture.repository,
        )
        self.assertEqual(staged, "generated.csv")

    def test_helper_issues_only_read_and_fetch_git_commands(self) -> None:
        commands: list[list[str]] = []
        responses = iter([self.fixture.source_sha, "", self.fixture.source_sha])

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            self.assertTrue(kwargs["check"])
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            self.assertFalse(kwargs.get("shell", False))
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=next(responses) + "\n",
                stderr="",
            )

        with patch.object(publish_guard.subprocess, "run", side_effect=fake_run):
            publish_guard.verify_data_publish_base(self.fixture.source_sha)

        self.assertEqual(
            commands,
            [
                ["git", "rev-parse", "HEAD"],
                ["git", "fetch", "--no-tags", "origin", "main"],
                ["git", "rev-parse", "FETCH_HEAD"],
            ],
        )
        forbidden = {"pull", "merge", "rebase", "reset", "checkout", "push"}
        self.assertFalse(
            any(command[1] in forbidden for command in commands),
            commands,
        )

    def test_subprocess_failure_propagates_as_publish_base_error(self) -> None:
        failure = subprocess.CalledProcessError(
            returncode=2,
            cmd=["git", "rev-parse", "HEAD"],
            stderr="fixture failure",
        )
        with (
            patch.object(publish_guard.subprocess, "run", side_effect=failure),
            self.assertRaises(publish_guard.PublishBaseError) as raised,
        ):
            publish_guard.verify_data_publish_base(self.fixture.source_sha)

        self.assertIn("exit code 2", str(raised.exception))
        self.assertIn("fixture failure", str(raised.exception))
        self.assertIn("Rerun the workflow from the latest main", str(raised.exception))

    def test_cli_returns_nonzero_and_prints_failure_guidance(self) -> None:
        error = publish_guard.PublishBaseError(
            f"fixture mismatch. {publish_guard.RERUN_GUIDANCE}"
        )
        stderr = io.StringIO()
        with (
            patch.object(
                publish_guard,
                "verify_data_publish_base",
                side_effect=error,
            ),
            contextlib.redirect_stderr(stderr),
        ):
            exit_code = publish_guard.main(
                ["--source-sha", self.fixture.source_sha]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("ERROR: fixture mismatch", stderr.getvalue())
        self.assertIn("Rerun the workflow from the latest main", stderr.getvalue())


class DataPublishWorkflowContractTest(unittest.TestCase):
    EXPECTED_STAGING_LINES = {
        "Backtest Only": [
            'git add -u -- "$raw_output"',
            "git add docs/data/backtest_summary.json docs/data/backtest_strategy_summary.csv docs/data/backtest_strategy_year_summary.csv",
            "git add docs/data/backtest_recent_trades.csv docs/data/signal_diagnostics_summary.csv docs/data/backtest_skipped_summary.csv",
            "git add docs/data/backtest_portfolio_strategy_summary.csv docs/data/backtest_portfolio_curve_manifest.json",
            "git add docs/data/backtest_benchmark_spy.json docs/data/backtest_portfolio_daily_returns.csv.gz",
            "git add -A docs/data/backtest_portfolio_curves",
        ],
        "Daily ETF Scan": [
            "git add config/aum.csv",
            "git add docs/data/latest.json docs/data/latest.csv docs/data/universe_current.csv docs/data/excluded_etfs_summary.csv",
            "git add docs/data/history || true",
        ],
    }
    CALCULATION_STEP = {
        "Backtest Only": "Run backtest only",
        "Daily ETF Scan": "Run ETF scan",
    }
    COMMIT_SUBJECT = {
        "Backtest Only": "Update backtest data",
        "Daily ETF Scan": "Update ETF scan data",
    }

    @classmethod
    def workflows_by_name(cls) -> dict[str, str]:
        workflows: dict[str, str] = {}
        for path in (ROOT / ".github" / "workflows").glob("*.yml"):
            text = path.read_text(encoding="utf-8")
            first_line = next(
                line for line in text.splitlines() if line.startswith("name:")
            )
            workflows[first_line.partition(":")[2].strip()] = text
        return workflows

    @staticmethod
    def step_names(workflow: str) -> list[str]:
        return [
            line.strip().partition(":")[2].strip()
            for line in workflow.splitlines()
            if line.strip().startswith("- name:")
        ]

    def test_source_sha_capture_is_immediately_after_checkout(self) -> None:
        workflows = self.workflows_by_name()
        for name in self.EXPECTED_STAGING_LINES:
            with self.subTest(workflow=name):
                workflow = workflows[name]
                steps = self.step_names(workflow)
                checkout_index = steps.index("Checkout")
                self.assertEqual(
                    steps[checkout_index + 1],
                    "Capture generated-data source revision",
                )
                capture_index = workflow.index(
                    "- name: Capture generated-data source revision"
                )
                calculation_index = workflow.index(
                    f"- name: {self.CALCULATION_STEP[name]}"
                )
                self.assertLess(capture_index, calculation_index)
                self.assertIn('SOURCE_SHA="$(git rev-parse HEAD)"', workflow)
                self.assertIn('>> "${GITHUB_ENV}"', workflow)
                self.assertIn(
                    'echo "generated_data_source_sha=${SOURCE_SHA}"',
                    workflow,
                )

    def test_guard_commit_parent_and_push_order(self) -> None:
        workflows = self.workflows_by_name()
        for name in self.EXPECTED_STAGING_LINES:
            with self.subTest(workflow=name):
                workflow = workflows[name]
                calculation_index = workflow.index(
                    f"- name: {self.CALCULATION_STEP[name]}"
                )
                guard_index = workflow.index(
                    "python scripts/verify_data_publish_base.py "
                    '--source-sha "${SOURCE_SHA}"'
                )
                commit_line = (
                    f'git commit -m "{self.COMMIT_SUBJECT[name]}" '
                    '-m "Generated-From: ${SOURCE_SHA}"'
                )
                commit_index = workflow.index(commit_line)
                parent_index = workflow.index(
                    'GENERATED_PARENT_SHA="$(git rev-parse HEAD^)"'
                )
                push_index = workflow.index("git push origin HEAD:main")
                self.assertLess(calculation_index, guard_index)
                if name == "Backtest Only":
                    self.assertLess(
                        workflow.index("- name: Preflight size check"),
                        guard_index,
                    )
                self.assertLess(guard_index, commit_index)
                self.assertLess(commit_index, parent_index)
                self.assertLess(parent_index, push_index)
                self.assertIn(
                    'if [ "${GENERATED_PARENT_SHA}" != "${SOURCE_SHA}" ]; then',
                    workflow,
                )
                self.assertIn("expected ${SOURCE_SHA}", workflow)
                self.assertIn("observed ${GENERATED_PARENT_SHA}", workflow)

    def test_no_change_path_and_owned_staging_paths_are_preserved(self) -> None:
        workflows = self.workflows_by_name()
        for name, expected_lines in self.EXPECTED_STAGING_LINES.items():
            with self.subTest(workflow=name):
                workflow = workflows[name]
                staging_lines = [
                    line.strip()
                    for line in workflow.splitlines()
                    if line.strip().startswith("git add ")
                ]
                self.assertEqual(staging_lines, expected_lines)
                no_change_index = workflow.index("if git diff --cached --quiet; then")
                guard_index = workflow.index(
                    "python scripts/verify_data_publish_base.py"
                )
                self.assertLess(no_change_index, guard_index)
                self.assertIn("exit 0", workflow[no_change_index:guard_index])

    def test_forbidden_publication_commands_and_error_swallowing_are_absent(self) -> None:
        workflows = self.workflows_by_name()
        forbidden_fragments = [
            "git pull",
            "git rebase",
            "git merge",
            "--autostash",
            "git push --force",
            "--force-with-lease",
        ]
        for name in self.EXPECTED_STAGING_LINES:
            with self.subTest(workflow=name):
                workflow = workflows[name]
                for fragment in forbidden_fragments:
                    self.assertNotIn(fragment, workflow)
                commit_lines = [
                    line.strip()
                    for line in workflow.splitlines()
                    if line.strip().startswith("git commit ")
                ]
                push_lines = [
                    line.strip()
                    for line in workflow.splitlines()
                    if line.strip().startswith("git push ")
                ]
                self.assertEqual(len(commit_lines), 1)
                self.assertEqual(push_lines, ["git push origin HEAD:main"])
                self.assertNotIn("||", commit_lines[0])
                self.assertNotIn("||", push_lines[0])

    def test_shared_concurrency_contract_is_unchanged(self) -> None:
        workflows = self.workflows_by_name()
        for name in self.EXPECTED_STAGING_LINES:
            with self.subTest(workflow=name):
                workflow = workflows[name]
                self.assertIn("group: data-publish-main", workflow)
                self.assertIn("cancel-in-progress: false", workflow)


if __name__ == "__main__":
    unittest.main()
