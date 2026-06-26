"""Smoke: the package imports and the CLI wires up."""

from typer.testing import CliRunner

import chronicle
from chronicle.cli import app

runner = CliRunner()


def test_version_string():
    assert isinstance(chronicle.__version__, str)
    assert chronicle.__version__


def test_help_lists_core_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "ask", "doctor", "migrate"):
        assert command in result.stdout


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert chronicle.__version__ in result.stdout


def test_ask_help():
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--as-of" in result.stdout
