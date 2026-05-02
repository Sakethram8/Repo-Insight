"""Sandbox module for isolated AI-driven edits on temporary repository clones."""

import difflib
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SANDBOX_ROOT = Path.home() / ".repo-insight" / "sandboxes"
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class ApplyBackResult:
    files_applied: list[str]
    files_failed: list[str]
    success: bool
    error: str | None = None


class SandboxManager:
    def __init__(self, original_path: str | Path) -> None:
        self.original_path = Path(original_path).resolve()
        if not self.original_path.exists():
            raise ValueError(f"Not a valid path: {self.original_path}")
        self.sandbox_id: str | None = None
        self.sandbox_path: Path | None = None

    def create(self) -> Path:
        self.sandbox_id = str(uuid.uuid4())[:8]
        self.sandbox_path = SANDBOX_ROOT / self.sandbox_id

        if (self.original_path / ".git").is_dir():
            subprocess.run(
                ["git", "clone", "--local", str(self.original_path), str(self.sandbox_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
        else:
            shutil.copytree(self.original_path, self.sandbox_path)

        return self.sandbox_path

    @property
    def is_ready(self) -> bool:
        return self.sandbox_path is not None and self.sandbox_path.is_dir()

    def get_diff(self) -> str:
        if (self.original_path / ".git").is_dir():
            result = subprocess.run(
                ["git", "diff", "--no-index", "--", str(self.original_path), str(self.sandbox_path)],
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode not in (0, 1):
                if result.returncode != 0:
                    logger.warning("git diff stderr: %s", result.stderr)
            return result.stdout
        else:
            all_py = []
            for root, _, files in os.walk(self.original_path):
                for f in files:
                    if f.endswith(".py"):
                        all_py.append(Path(root) / f)
            for root, _, files in os.walk(self.sandbox_path):
                for f in files:
                    if f.endswith(".py"):
                        all_py.append(Path(root) / f)
            all_py = sorted(set(all_py))
            diff_parts = []
            for py_path in all_py:
                rel = py_path.relative_to(self.original_path if self.original_path in py_path.parents else self.sandbox_path if self.sandbox_path in py_path.parents else Path("."))
                orig = py_path if self.original_path in py_path.parents else None
                sand = py_path if self.sandbox_path in py_path.parents else None
                if orig is not None and orig.exists():
                    orig_content = orig.read_text()
                else:
                    orig_content = ""
                if sand is not None and sand.exists():
                    sand_content = sand.read_text()
                else:
                    sand_content = ""
                orig_rel = str(orig.relative_to(self.original_path)) if orig else str(rel)
                sand_rel = str(sand.relative_to(self.sandbox_path)) if sand else str(rel)
                diff_lines = list(difflib.unified_diff(
                    orig_content.splitlines(keepends=True),
                    sand_content.splitlines(keepends=True),
                    fromfile=f"a/{orig_rel}",
                    tofile=f"b/{sand_rel}",
                ))
                if diff_lines:
                    diff_parts.append("".join(diff_lines))
            return "".join(diff_parts)

    def get_changed_files(self) -> list[dict]:
        diff_text = self.get_diff()
        changed_files = []
        current_file = None
        added = 0
        removed = 0

        for line in diff_text.splitlines():
            if line.startswith("--- a/"):
                current_file = None
                path_part = line[6:].strip()
                added = 0
                removed = 0
            elif line.startswith("+++ b/"):
                if current_file is not None:
                    changed_files.append({
                        "path": current_file,
                        "status": "M",
                        "lines_added": added,
                        "lines_removed": removed,
                    })
                current_file = line[6:].strip()
                added = 0
                removed = 0
            elif current_file is not None and line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif current_file is not None and line.startswith("-") and not line.startswith("---"):
                removed += 1

        if current_file is not None:
            changed_files.append({
                "path": current_file,
                "status": "M",
                "lines_added": added,
                "lines_removed": removed,
            })

        return changed_files

    def apply_to_original(self) -> ApplyBackResult:
        changed_files = self.get_changed_files()
        files_applied = []
        files_failed = []

        for file_info in changed_files:
            rel_path = file_info["path"]
            if rel_path.startswith("a/"):
                rel_path = rel_path[2:]
            elif rel_path.startswith("b/"):
                rel_path = rel_path[2:]

            sandbox_file = self.sandbox_path / rel_path
            original_file = self.original_path / rel_path

            try:
                if not sandbox_file.exists():
                    files_failed.append(rel_path)
                    continue
                original_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sandbox_file, original_file)
                files_applied.append(rel_path)
            except Exception:
                files_failed.append(rel_path)

        return ApplyBackResult(
            files_applied=files_applied,
            files_failed=files_failed,
            success=len(files_failed) == 0,
        )

    def discard(self) -> None:
        if self.sandbox_path is not None and self.sandbox_path.exists():
            shutil.rmtree(self.sandbox_path, ignore_errors=True)
        self.sandbox_path = None

    @staticmethod
    def cleanup_old_sandboxes(max_age_hours: int = 24) -> int:
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        deleted = 0
        if not SANDBOX_ROOT.exists():
            return 0
        for entry in SANDBOX_ROOT.iterdir():
            if entry.is_dir():
                try:
                    mtime = os.stat(entry).st_mtime
                    if mtime < cutoff:
                        shutil.rmtree(entry, ignore_errors=True)
                        deleted += 1
                except OSError:
                    pass
        return deleted


if __name__ == "__main__":
    m = SandboxManager(".")
    p = m.create()
    print(f"Sandbox: {m.sandbox_id} at {p}")
    diff = m.get_diff()
    print(f"Diff length: {len(diff)} chars")
    m.discard()
    print("Discarded.")
