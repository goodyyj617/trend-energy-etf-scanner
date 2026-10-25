"""Bring the newest merged lab code from GitHub (origin/main) into this folder before a lab starts.

    python scripts/lab_update.py [FOLDER]     # FOLDER defaults to the repository root

The two lab launchers (.cmd) and the two bookmark handlers (aalab://, trendlab://) run this before they start
Streamlit, so a merged pull request shows up the next time the lab is opened.

Rules:
- Stay on the folder's current branch. main is merged into it (`git fetch origin main` + `git merge`), never
  checked out, so a work branch with its own commits (the laptop's Remote Control branch) keeps them.
- Never lose or block anything: if the merge cannot finish (conflict, an uncommitted edit in a file main also
  changed, no network, no git, a merge already in progress) the folder is left exactly as it was and the lab
  starts on the code it already has. The script always exits 0.
- When requirements*.txt changed, the packages are installed again.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
REMOTE, BRANCH = "origin", "main"
REQUIREMENTS = ("requirements.txt", "requirements-lab.txt")
# used only for the local merge commit when this computer has no git name/e-mail set
FALLBACK_ID = ["-c", "user.name=lab-launcher", "-c", "user.email=lab-launcher@localhost"]
SHOW_COMMITS = 8


@dataclass
class Result:
    status: str  # updated | current | kept | offline | skip
    lines: list[str] = field(default_factory=list)
    new_commits: int = 0


def _git(root: Path, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    # never wait for a password prompt (terminal or Git Credential Manager window)
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash under pythonw (bookmarks)
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, timeout=timeout, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(["git", *args], 1, "", str(e))


def _out(root: Path, *args: str) -> str:
    return _git(root, *args).stdout.strip()


def pip_install(root: Path) -> bool:
    files = [a for f in REQUIREMENTS if (root / f).exists() for a in ("-r", f)]
    if not files:
        return True
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        done = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *files], cwd=root, creationflags=flags)
    except OSError:
        return False
    return done.returncode == 0


def update(root: Path = ROOT, *, install: Callable[[Path], object] | None = pip_install) -> Result:
    root = Path(root)
    if shutil.which("git") is None:
        return Result("skip", ["git이 없어 업데이트를 건너뜁니다. 지금 있는 코드로 켭니다."])
    if _out(root, "rev-parse", "--is-inside-work-tree") != "true":
        return Result("skip", ["git 폴더가 아니어서 업데이트를 건너뜁니다. 지금 있는 코드로 켭니다."])
    git_dir = Path(_out(root, "rev-parse", "--absolute-git-dir"))
    if any((git_dir / p).exists() for p in ("MERGE_HEAD", "rebase-merge", "rebase-apply", "CHERRY_PICK_HEAD")):
        return Result("skip", ["이 폴더에서 진행 중인 병합 작업이 있어 업데이트를 건너뜁니다. 지금 있는 코드로 켭니다."])

    if _git(root, "fetch", "-q", REMOTE, BRANCH, timeout=45).returncode != 0:  # the only step with a time limit
        return Result("offline", ["GitHub에 연결하지 못해 업데이트를 건너뜁니다. 지금 있는 코드로 켭니다."])
    new = _out(root, "rev-parse", "FETCH_HEAD")
    old = _out(root, "rev-parse", "HEAD")
    if _git(root, "merge-base", "--is-ancestor", new, "HEAD").returncode == 0:
        return Result("current", ["이미 최신 코드입니다."])

    ident = [] if _out(root, "config", "user.email") and _out(root, "config", "user.name") else FALLBACK_ID
    # a local merge commit: no signing (a broken signing setup must not block every update)
    merged = _git(root, *ident, "merge", "--no-edit", "--no-gpg-sign",
                  "-m", f"Merge {REMOTE}/{BRANCH} (lab launcher update)", new)
    if merged.returncode != 0:
        if (git_dir / "MERGE_HEAD").exists():
            _git(root, "merge", "--abort")
        if _out(root, "rev-parse", "HEAD") != old:  # should not happen; put the branch back where it was
            _git(root, "reset", "-q", "--keep", old)
        why = [ln.strip() for ln in (merged.stdout + "\n" + merged.stderr).splitlines()
               if ln.strip() and not ln.startswith("hint:") and ("CONFLICT" in ln or "error" in ln or ln[:1] == "\t")]
        return Result("kept", [
            "새 코드를 자동으로 합치지 못했습니다. 아무것도 바꾸지 않고 지금 있는 코드로 켭니다.",
            *[f"  git: {ln}" for ln in why[:6]],
            "Claude에게 이 창의 내용을 보여 주며 '연구실 업데이트가 안 된다'고 알려 주세요.",
        ])

    count = int(_out(root, "rev-list", "--count", "--no-merges", f"{old}..HEAD") or 0)
    subjects = _out(root, "log", "--no-merges", f"-n{SHOW_COMMITS}", "--format=%s", f"{old}..HEAD").splitlines()
    lines = [f"GitHub에서 새 변경 {count}개를 받았습니다."] + [f"  · {s}" for s in subjects]
    if count > len(subjects):
        lines.append(f"  · … 외 {count - len(subjects)}개")
    if _git(root, "diff", "--quiet", old, "HEAD", "--", *REQUIREMENTS).returncode != 0 and install is not None:
        ok = install(root) is not False
        lines.append("필요한 패키지 목록이 바뀌어 다시 설치했습니다." if ok else
                     "필요한 패키지를 설치하지 못했습니다. 인터넷 연결을 확인한 뒤 다시 켜 보세요.")
    return Result("updated", lines, count)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else ROOT
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        res = update(root)
    except Exception as e:  # the lab must start whatever happens here
        res = Result("skip", [f"업데이트 중 오류가 나 건너뜁니다({type(e).__name__}). 지금 있는 코드로 켭니다."])
    for line in ["[업데이트] " + res.lines[0], *res.lines[1:]]:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
