import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from data_extraction import load_data


def test_missing_file_returns_none(tmp_path):
    missing = tmp_path / "no_such_file.csv"
    result = load_data(str(missing))
    assert result is None

def test_load_valid_csv(tmp_path):
    p = tmp_path / "valid.csv"
    p.write_text("a,b\n1,2\n3,4", encoding="utf-8")
    df = load_data(str(p))
    assert df is not None
    assert tuple(df.shape) == (2, 2)
    assert list(df.columns) == ["a", "b"]

def test_empty_file_returns_none(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    df = load_data(str(p))
    assert df is None

def test_semicolon_delimiter_detection(tmp_path):
    p = tmp_path / "semi.csv"
    p.write_text("col1;col2\nx;y\nz;w", encoding="utf-8")
    df = load_data(str(p))
    assert df is not None
    assert tuple(df.shape) == (2, 2)
    assert list(df.columns) == ["col1", "col2"]

def test_encoding_fallback(monkeypatch, tmp_path):
    p = tmp_path / "latin1.csv"
    p.write_bytes("café\nå".encode("latin1"))

    original_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if kwargs.get("encoding") is None:
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")
        return pd.DataFrame({"cafe": ["café"]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    try:
        df = load_data(str(p))
        assert df is not None
        assert list(df.columns) == ["cafe"]
    finally:
        monkeypatch.setattr(pd, "read_csv", original_read_csv)

def test_parser_error_fallback(monkeypatch, tmp_path):
    p = tmp_path / "weird_delim.csv"
    p.write_text("a|b\n1|2\n3|4", encoding="utf-8")

    call = {"n": 0}

    def fake_read_csv(path, *args, **kwargs):
        if call["n"] == 0:
            call["n"] += 1
            raise pd.errors.ParserError("simulated parser error")
        return pd.DataFrame({"a": [1, 3], "b": [2, 4]})

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)
    try:
        df = load_data(str(p))
        assert df is not None
        assert tuple(df.shape) == (2, 2)
        assert list(df.columns) == ["a", "b"]
    finally:
        monkeypatch.setattr(pd, "read_csv", pd.read_csv)
