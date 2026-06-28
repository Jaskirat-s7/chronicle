# War stories

A running log of subtle failures and gotchas hit while building chronicle.
Append as we hit them — these are interview gold.

---

## PR1 — date answers are timezone-sensitive

**Symptom.** An external cross-check of a `commit_date` ground-truth answer
disagreed by one day: raw `git log --date=format:%Y-%m-%d` said `2021-02-08`,
our generated answer said `2021-02-09`.

**Cause.** Not a bug. The commit was authored at `2021-02-08T18:17:24-08:00`,
which is `2021-02-09 02:17 UTC`. `git`'s default `%ad` renders the author's
*local* timezone; our generator normalizes the author-time epoch to **UTC**.

**Resolution / how we avoid it.** Date answers explicitly say "(YYYY-MM-DD,
UTC)" in the question text, and both the generator and the independent verifier
derive the date from the epoch in UTC (`datetime.fromtimestamp(t, tz=utc)`).
The answer is well-defined and self-consistent. Lesson: any temporal label must
pin a timezone, or "what day did this change" is ambiguous — which matters a lot
for a tool whose whole premise is time as a first-class dimension.
