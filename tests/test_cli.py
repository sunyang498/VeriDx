"""CLI 入口冒烟测试（P0.1 验收：`veridx --version` 有输出）。"""

from __future__ import annotations

import json

import pytest

from veridx.cli import main


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("veridx ")


def test_info_prints_json_without_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "db_path" in payload and "llm_mode" in payload
    assert "api_key" not in json.dumps(payload).lower()


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: veridx" in capsys.readouterr().out
