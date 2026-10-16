"""LongTermMemory tests: persistence, corrupt-file handling, diffs."""
import json

from wealth_assistant.memory import LongTermMemory


def _summary(**overrides):
    base = {"risk_score": 10, "anomalies": [], "recommendations": [],
            "total_balance": 100.0}
    base.update(overrides)
    return base


def test_save_and_get_roundtrip(tmp_path):
    mem = LongTermMemory(str(tmp_path / "memory.json"))
    mem.save_summary("C1", _summary(risk_score=42))

    got = mem.get_last_summary("C1")
    assert got["risk_score"] == 42
    assert "timestamp" in got
    # persisted as valid JSON
    raw = json.loads((tmp_path / "memory.json").read_text())
    assert raw["C1"]["risk_score"] == 42


def test_missing_file_starts_empty(tmp_path):
    mem = LongTermMemory(str(tmp_path / "memory.json"))
    assert mem.get_last_summary("C1") is None


def test_corrupt_file_starts_empty_with_warning(tmp_path, caplog):
    path = tmp_path / "memory.json"
    path.write_text("{oops", encoding="utf-8")
    with caplog.at_level("WARNING"):
        mem = LongTermMemory(str(path))
    assert mem.get_last_summary("C1") is None
    assert any("corrupt" in r.message for r in caplog.records)


def test_diff_new_resolved_worsened(tmp_path):
    mem = LongTermMemory(str(tmp_path / "memory.json"))
    mem.save_summary("C1", _summary(anomalies=[
        {"type": "t", "id": "a1", "severity": "low"},
        {"type": "t", "id": "a2", "severity": "medium"},
    ]))
    diff = mem.diff_with_last("C1", {"anomalies": [
        {"type": "t", "id": "a2", "severity": "high"},
        {"type": "t", "id": "a3", "severity": "low"},
    ]})
    assert diff == {"new": ["a3"], "resolved": ["a1"], "worsened": ["a2"]}


def test_diff_first_run_counts_everything_new(tmp_path):
    mem = LongTermMemory(str(tmp_path / "memory.json"))
    diff = mem.diff_with_last("C1", {"anomalies": [
        {"type": "t", "id": "a1", "severity": "low"},
    ]})
    assert diff == {"new": ["a1"], "resolved": [], "worsened": []}
