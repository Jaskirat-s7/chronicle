"""Generate labeled eval questions from real git history (deterministic, seeded)."""

from __future__ import annotations

import random
from pathlib import Path

from .. import git_ops
from ..logging_config import get_logger
from .schema import EvalQuestion

log = get_logger("chronicle.groundtruth")

# Text/code files we will sample lines from. Keeps blame answers meaningful and
# avoids binary/vendored noise.
ALLOWED_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".rb", ".php", ".cs", ".swift",
    ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml",
    ".json", ".sh",
}
_SKIP_DIR_PARTS = {"vendor", "node_modules", "third_party", "dist", "build"}


class GroundTruthGenerator:
    def __init__(self, repo: Path | str, repo_name: str, *, seed: int = 7) -> None:
        self.repo = repo
        self.repo_name = repo_name
        self.rng = random.Random(seed)
        self.eval_sha = git_ops.head_sha(repo)
        self.short = git_ops.short(self.eval_sha)

    # -- candidate selection -------------------------------------------------

    def _candidate_files(self) -> list[str]:
        files = git_ops.list_files(self.repo, self.eval_sha)
        out = []
        for f in files:
            p = Path(f)
            if p.suffix.lower() not in ALLOWED_EXT:
                continue
            if _SKIP_DIR_PARTS & set(p.parts):
                continue
            out.append(f)
        self.rng.shuffle(out)
        return out

    def _pick_nonblank_line(self, path: str) -> int | None:
        content = git_ops.show_file(self.repo, self.eval_sha, path).splitlines()
        idxs = [i for i, ln in enumerate(content, start=1) if ln.strip()]
        if not idxs:
            return None
        return self.rng.choice(idxs)

    # -- per-template builders ----------------------------------------------

    def _blame_questions(self, path: str, line: int) -> list[EvalQuestion]:
        bl = git_ops.blame_line(self.repo, self.eval_sha, path, line)
        ev = {"path": path, "line": line, "rev": self.eval_sha}
        qs = [
            EvalQuestion(
                id=f"blame_commit::{path}::{line}",
                template="blame_commit",
                question=(
                    f"Which commit (full 40-char SHA) last modified line {line} "
                    f"of `{path}`, as of commit {self.short}?"
                ),
                answer=bl.sha,
                answer_kind="commit_sha",
                temporal=True,
                repo=self.repo_name,
                eval_sha=self.eval_sha,
                evidence=ev,
            ),
            EvalQuestion(
                id=f"commit_date::{path}::{line}",
                template="commit_date",
                question=(
                    f"On what date (YYYY-MM-DD, UTC) was line {line} of `{path}` "
                    f"last changed, as of commit {self.short}?"
                ),
                answer=bl.author_time.strftime("%Y-%m-%d"),
                answer_kind="date",
                temporal=True,
                repo=self.repo_name,
                eval_sha=self.eval_sha,
                evidence=ev,
            ),
            EvalQuestion(
                id=f"blame_author::{path}::{line}",
                template="blame_author",
                question=(
                    f"Who authored the change that last modified line {line} of "
                    f"`{path}`, as of commit {self.short}?"
                ),
                answer=bl.author_name,
                answer_kind="author",
                temporal=False,
                repo=self.repo_name,
                eval_sha=self.eval_sha,
                evidence=ev,
            ),
        ]
        return qs

    def _file_added_question(self, path: str) -> EvalQuestion:
        sha = git_ops.first_add_commit(self.repo, path)
        return EvalQuestion(
            id=f"file_added::{path}",
            template="file_added",
            question=f"In which commit (full 40-char SHA) was the file `{path}` first added to the repository?",
            answer=sha,
            answer_kind="commit_sha",
            temporal=True,
            repo=self.repo_name,
            eval_sha=self.eval_sha,
            evidence={"path": path},
        )

    def _pr_question(self, meta: git_ops.CommitMeta) -> EvalQuestion:
        assert meta.pr_number is not None
        return EvalQuestion(
            id=f"pr_for_commit::{meta.sha}",
            template="pr_for_commit",
            question=(
                f"Which pull request number introduced commit {git_ops.short(meta.sha)} "
                f"(\"{meta.subject}\")?"
            ),
            answer=str(meta.pr_number),
            answer_kind="pr_number",
            temporal=False,
            repo=self.repo_name,
            eval_sha=self.eval_sha,
            evidence={"sha": meta.sha},
        )

    # -- orchestration -------------------------------------------------------

    def generate(self, target: int) -> list[EvalQuestion]:
        """Produce ~`target` questions balanced across templates."""
        files = self._candidate_files()
        if not files:
            raise RuntimeError("No candidate text/code files found in repo.")

        questions: list[EvalQuestion] = []
        seen_ids: set[str] = set()

        def add(q: EvalQuestion) -> None:
            if q.id not in seen_ids:
                seen_ids.add(q.id)
                questions.append(q)

        # Blame family (3 questions per sampled line) + file_added.
        # Sample enough lines/files to comfortably exceed the target.
        n_lines = max(target, 40)
        for path in files:
            if len([q for q in questions if q.template != "pr_for_commit"]) >= n_lines * 3:
                break
            line = self._pick_nonblank_line(path)
            if line is None:
                continue
            for q in self._blame_questions(path, line):
                add(q)
            add(self._file_added_question(path))

        # PR-link family from commits that reference a PR.
        commits = git_ops.list_commits(self.repo, max_count=2000)
        pr_commits = [c for c in commits if c.pr_number is not None]
        self.rng.shuffle(pr_commits)
        for meta in pr_commits[: max(target // 4, 10)]:
            add(self._pr_question(meta))

        self.rng.shuffle(questions)
        log.info("Generated %d candidate questions (target %d).", len(questions), target)
        return questions[:target] if len(questions) >= target else questions

    def stratified_subset(
        self, questions: list[EvalQuestion], size: int
    ) -> list[EvalQuestion]:
        """A smaller dev set that still covers every template present."""
        by_template: dict[str, list[EvalQuestion]] = {}
        for q in questions:
            by_template.setdefault(q.template, []).append(q)

        chosen: list[EvalQuestion] = []
        templates = list(by_template)
        # Round-robin across templates for even coverage.
        idx = 0
        while len(chosen) < size and any(by_template.values()):
            t = templates[idx % len(templates)]
            bucket = by_template[t]
            if bucket:
                chosen.append(bucket.pop())
            idx += 1
        return chosen[:size]
