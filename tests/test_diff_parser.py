"""Strict diff parser: correct structure on valid input, crash on malformed."""

import pytest

from chronicle.diff_parser import DiffParseError, parse_patch

VALID = """\
diff --git a/foo.py b/foo.py
index 1111111..2222222 100644
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@ def f():
 a
-b
+B
+c
 d
diff --git a/old.txt b/new.txt
similarity index 100%
rename from old.txt
rename to new.txt
"""


def test_parse_valid_multifile():
    files = parse_patch(VALID)
    assert len(files) == 2

    foo = files[0]
    assert foo.new_path == "foo.py"
    assert len(foo.hunks) == 1
    h = foo.hunks[0]
    assert (h.old_start, h.old_count, h.new_start, h.new_count) == (1, 3, 1, 4)
    assert h.new_span == (1, 4)
    assert h.old_span == (1, 3)
    assert h.added_lines == [2, 3]
    assert h.removed_lines == [2]
    assert foo.added == 2 and foo.removed == 1
    assert h.section == "def f():"

    rename = files[1]
    assert rename.is_rename
    assert rename.old_path == "old.txt"
    assert rename.new_path == "new.txt"
    assert rename.hunks == []


def test_binary_file():
    patch = (
        "diff --git a/img.png b/img.png\n"
        "index aaa..bbb 100644\n"
        "Binary files a/img.png and b/img.png differ\n"
    )
    files = parse_patch(patch)
    assert files[0].is_binary
    assert files[0].hunks == []


def test_new_and_deleted_flags():
    patch = (
        "diff --git a/new.py b/new.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/new.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+line1\n"
        "+line2\n"
    )
    fd = parse_patch(patch)[0]
    assert fd.is_new
    assert fd.hunks[0].added_lines == [1, 2]
    assert fd.hunks[0].removed_lines == []


def test_malformed_hunk_header_raises():
    bad = (
        "diff --git a/x b/x\n"
        "--- a/x\n"
        "+++ b/x\n"
        "@@ -1 +nonsense @@\n"
        " a\n"
    )
    with pytest.raises(DiffParseError, match="hunk header"):
        parse_patch(bad)


def test_unexpected_line_prefix_raises():
    bad = (
        "diff --git a/x b/x\n"
        "--- a/x\n"
        "+++ b/x\n"
        "@@ -1,1 +1,1 @@\n"
        "?corrupt\n"
    )
    with pytest.raises(DiffParseError, match="Unexpected line prefix"):
        parse_patch(bad)


def test_truncated_hunk_raises():
    bad = (
        "diff --git a/x b/x\n"
        "--- a/x\n"
        "+++ b/x\n"
        "@@ -1,3 +1,3 @@\n"
        " only one line\n"
    )
    with pytest.raises(DiffParseError, match="truncated"):
        parse_patch(bad)
