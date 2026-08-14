"""Unit tests for get_task_template — regression test for the cwd-relative sc.json bug."""
from __future__ import annotations

import os

from psychscanner import get_task_template


def test_default_template_has_expected_keys():
    template = get_task_template()
    assert template["tasktype"] == "sc"
    assert "items" in template


def test_sc_template_loads_regardless_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    template = get_task_template("sc")
    assert template["tasktype"] == "sc"
    assert template["items"] == [{}]


def test_unknown_ttype_falls_back_to_default(capsys):
    template = get_task_template("nonexistent")
    assert template["tasktype"] == "sc"
    assert template["taskname"] == ""
    captured = capsys.readouterr()
    assert "not found" in captured.out.lower()
