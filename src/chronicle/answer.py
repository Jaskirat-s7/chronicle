"""Answer generation: retrieve -> build cited context -> generate via Gemini.

Citations carry commit + date + file:line. In the PR2 snapshot baseline every
chunk is tagged with HEAD, so temporal/"when" questions are expected to do
poorly — that is the measured floor PR3 improves on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .git_ops import short
from .models.base import LLMResponse, ModelClient
from .retrieval import HybridRetriever, RetrievedChunk

_SYSTEM = (
    "You are chronicle, a tool that answers questions about a git repository's "
    "history. Answer ONLY from the provided context. Cite the commit SHA, date, "
    "and file:line you relied on. If the context is insufficient to answer, say "
    "so explicitly rather than guessing."
)


@dataclass
class AnswerResult:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    llm: LLMResponse | None = None
    contexts: list[str] = field(default_factory=list)


def format_context(chunks: list[RetrievedChunk]) -> list[str]:
    blocks: list[str] = []
    for c in chunks:
        date = c.commit_date.strftime("%Y-%m-%d")
        header = f"[{c.file_path}:{c.line_start}-{c.line_end} @ {short(c.commit_sha)} {date}]"
        blocks.append(f"{header}\n{c.content}")
    return blocks


def build_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)"
    return f"Context:\n{joined}\n\nQuestion: {question}\n\nAnswer with citations:"


def answer_question(
    question: str,
    retriever: HybridRetriever,
    generator: ModelClient,
    *,
    repo: str | None = None,
    top_k: int = 8,
    rerank: bool = True,
) -> AnswerResult:
    retrieved = retriever.retrieve(question, repo=repo, top_k=top_k, rerank=rerank)
    contexts = format_context(retrieved)
    prompt = build_prompt(question, contexts)
    llm = generator.complete(prompt, system=_SYSTEM)
    return AnswerResult(
        question=question,
        answer=llm.text,
        retrieved=retrieved,
        llm=llm,
        contexts=contexts,
    )
