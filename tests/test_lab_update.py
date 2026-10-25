"""Launcher auto-update (scripts/lab_update.py): bring merged code from GitHub's main into the lab folder.

Every test builds its own throw-away "GitHub" (a bare repository) and "user folder" (a clone) under tmp_path,
with git's global/system config switched off so the machine's own git settings never leak in.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import lab_update

ROOT = Path(__file__).resolve().parents[1]
ID = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]


@pytest.fixture(autouse=True)
def _isolated_git(tmp_path, monkeypatch):
    empty = tmp_path / "empty_gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(k, raising=False)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *ID, *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def commit(repo: Path, name: str, text: str, msg: str) -> None:
    (repo / name).write_text(text, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", msg)


@pytest.fixture
def setup(tmp_path):
    """(github, user folder, a second clone that 'merges PRs' into main)."""
    hub = tmp_path / "hub.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(hub)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(hub), str(seed)], check=True, capture_output=True)
    git(seed, "checkout", "-q", "-b", "main")
    commit(seed, "app.py", "page = 1\n", "first")
    commit(seed, "requirements.txt", "pandas\n", "requirements")
    git(seed, "push", "-q", "origin", "main")
    user = tmp_path / "user"
    subprocess.run(["git", "clone", "-q", str(hub), str(user)], check=True, capture_output=True)
    return hub, user, seed


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD")


def merge_pr(seed: Path, name: str = "search.py", text: str = "new page\n", msg: str = "PR #77 new page") -> None:
    commit(seed, name, text, msg)
    git(seed, "push", "-q", "origin", "main")


def test_brings_a_merged_pr_into_the_folder(setup):
    _, user, seed = setup
    merge_pr(seed)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "updated"
    assert res.new_commits == 1
    assert (user / "search.py").read_text(encoding="utf-8") == "new page\n"
    assert any("PR #77 new page" in line for line in res.lines)


def test_second_run_is_already_current(setup):
    _, user, seed = setup
    merge_pr(seed)
    lab_update.update(user, install=lambda root: None)
    before = head(user)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "current"
    assert head(user) == before


def test_stays_on_its_own_branch_and_keeps_its_commits(setup):
    # the laptop folder sits on a work branch with commits not on main: merge main in, never switch branches
    _, user, seed = setup
    git(user, "checkout", "-q", "-b", "work")
    commit(user, "local.py", "mine\n", "local work")
    merge_pr(seed)
    res = lab_update.update(user, install=lambda root: None)  # no git identity configured at all here
    assert res.status == "updated"
    assert git(user, "rev-parse", "--abbrev-ref", "HEAD") == "work"
    assert (user / "local.py").exists() and (user / "search.py").exists()
    assert len(git(user, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3  # a merge commit


def test_conflict_changes_nothing(setup):
    _, user, seed = setup
    git(user, "checkout", "-q", "-b", "work")
    commit(user, "app.py", "page = 2\n", "local edit")
    merge_pr(seed, "app.py", "page = 3\n", "main edit")
    before = head(user)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "kept"
    assert head(user) == before
    assert not (user / ".git" / "MERGE_HEAD").exists()
    assert git(user, "status", "--porcelain") == ""
    assert (user / "app.py").read_text(encoding="utf-8") == "page = 2\n"


def test_unsaved_edit_in_a_file_main_changes_is_kept(setup):
    _, user, seed = setup
    (user / "app.py").write_text("page = 99  # not committed\n", encoding="utf-8")
    merge_pr(seed, "app.py", "page = 3\n", "main edit")
    before = head(user)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "kept"
    assert head(user) == before
    assert (user / "app.py").read_text(encoding="utf-8") == "page = 99  # not committed\n"


def test_unsaved_edit_elsewhere_does_not_block_the_update(setup):
    _, user, seed = setup
    (user / "app.py").write_text("page = 99  # not committed\n", encoding="utf-8")
    merge_pr(seed)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "updated"
    assert (user / "search.py").exists()
    assert (user / "app.py").read_text(encoding="utf-8") == "page = 99  # not committed\n"


def test_offline_changes_nothing(setup, tmp_path):
    _, user, _ = setup
    git(user, "remote", "set-url", "origin", str(tmp_path / "no_such_repo.git"))
    before = head(user)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "offline"
    assert head(user) == before


def test_not_a_git_folder_is_skipped(tmp_path):
    folder = tmp_path / "plain"
    folder.mkdir()
    assert lab_update.update(folder, install=lambda root: None).status == "skip"


def test_unfinished_merge_is_left_alone(setup):
    _, user, seed = setup
    git(user, "checkout", "-q", "-b", "work")
    commit(user, "app.py", "page = 2\n", "local edit")
    merge_pr(seed, "app.py", "page = 3\n", "main edit")
    git(user, "fetch", "-q", "origin", "main")
    subprocess.run(["git", *ID, "merge", "origin/main"], cwd=user, capture_output=True)  # leaves a conflict behind
    assert (user / ".git" / "MERGE_HEAD").exists()
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "skip"
    assert (user / ".git" / "MERGE_HEAD").exists()  # the user's own merge is not aborted


def test_packages_are_installed_only_when_requirements_change(setup):
    _, user, seed = setup
    calls = []
    merge_pr(seed)
    lab_update.update(user, install=calls.append)
    assert calls == []
    merge_pr(seed, "requirements.txt", "pandas\nplotly\n", "add plotly")
    res = lab_update.update(user, install=calls.append)
    assert res.status == "updated"
    assert calls == [user]


def test_main_never_fails(tmp_path, capsys):
    folder = tmp_path / "plain"
    folder.mkdir()
    assert lab_update.main([str(folder)]) == 0
    assert capsys.readouterr().out.strip() != ""


def test_launchers_update_before_starting():
    # A running .cmd is read from disk line by line, so a launcher that the update itself rewrites must have
    # parsed everything after the update already: the update and the start sit in one ( ... ) block ending in exit.
    for name in ("자산배분 연구실 시작.cmd", "추세추종 연구실 시작.cmd"):
        text = (ROOT / name).read_text(encoding="utf-8")
        block = text[text.index("\n(") :]
        assert block.index("scripts\\lab_update.py") < block.index("streamlit run")
        assert "exit /b" in block[block.index("streamlit run") :]
        assert all(ord(ch) < 128 for line in text.splitlines() if line.strip().lower().startswith("echo") for ch in line)
    for name in ("aa_link.py", "lab_link.py"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        start = text.index("def open_link")
        assert text.index("lab_update", start) < text.index("start_lab()", start)


def test_a_broken_signing_setup_does_not_block_the_update(setup):
    # the merge commit is local; a commit-signing setup that fails must not stop every future update
    _, user, seed = setup
    git(user, "checkout", "-q", "-b", "work")
    commit(user, "local.py", "mine\n", "local work")
    git(user, "config", "commit.gpgsign", "true")
    git(user, "config", "gpg.program", "false")
    merge_pr(seed)
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "updated"
    assert (user / "search.py").exists()


def test_a_failed_merge_says_why(setup):
    _, user, seed = setup
    git(user, "checkout", "-q", "-b", "work")
    commit(user, "app.py", "page = 2\n", "local edit")
    merge_pr(seed, "app.py", "page = 3\n", "main edit")
    res = lab_update.update(user, install=lambda root: None)
    assert res.status == "kept"
    assert any("app.py" in line for line in res.lines)  # git's own reason names the file
