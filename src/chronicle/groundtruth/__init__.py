"""Auto ground-truth: labeled eval Q/A derived only from real git history.

No model is involved and nothing is mocked. Every answer is re-derivable from
git, and the generator self-checks each question with the verifier before it is
written — so "git-verifiable" is asserted, not merely claimed.
"""

from .schema import EvalQuestion, load_jsonl, write_jsonl
from .generator import GroundTruthGenerator
from .verifier import VerificationError, verify_all, verify_question

__all__ = [
    "EvalQuestion",
    "GroundTruthGenerator",
    "VerificationError",
    "verify_all",
    "verify_question",
    "load_jsonl",
    "write_jsonl",
]
