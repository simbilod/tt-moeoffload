"""prepare.py's model step without the network: files already on disk are checked against their pins, then linked."""
import hashlib

import pytest

from tt_moeoffload import prepare as P


def pins(tmp_path, body: bytes):
    return {"models": {"m": {"repo": "x/y", "revision": "r" * 40, "files": {
        "sub/a.gguf": {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body), "to": "m-dir/a.gguf"}}}}}


def test_from_links_and_verifies(tmp_path, monkeypatch):
    src = tmp_path / "have"
    src.mkdir()
    (src / "a.gguf").write_bytes(b"weights")
    monkeypatch.setattr(P, "PINS", pins(tmp_path, b"weights"))
    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    first = P.model("m", src)
    assert first == tmp_path / "data" / "models" / "m-dir" / "a.gguf"
    assert (tmp_path / "data" / "models" / "m-dir").resolve() == src.resolve()
    assert "m-dir/a.gguf" in (tmp_path / "data" / "models" / ".verified.json").read_text()
    P.model("m", None)                                  # a rerun: verified by the book, nothing fetched


def test_wrong_file_refused(tmp_path, monkeypatch):
    src = tmp_path / "have"
    src.mkdir()
    (src / "a.gguf").write_bytes(b"WEIGHTS")            # same size, other bytes
    monkeypatch.setattr(P, "PINS", pins(tmp_path, b"weights"))
    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    with pytest.raises(SystemExit, match="not the pinned file"):
        P.model("m", src)


def test_from_takes_the_hub_layout(tmp_path, monkeypatch):
    src = tmp_path / "repo"                             # the Hub's layout: the pinned path sub/a.gguf under DIR
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.gguf").write_bytes(b"weights")
    monkeypatch.setattr(P, "PINS", pins(tmp_path, b"weights"))
    monkeypatch.setattr(P, "DATA", tmp_path / "data")
    P.model("m", src)
    assert (tmp_path / "data" / "models" / "m-dir").resolve() == (src / "sub").resolve()


def test_a_whole_part_is_not_fetched_again(tmp_path, monkeypatch):
    dest = tmp_path / "a.gguf"
    (tmp_path / "a.gguf.part").write_bytes(b"weights")
    monkeypatch.setattr(P.urllib.request, "urlopen", lambda *a, **k: pytest.fail("no request for a whole .part"))
    monkeypatch.setattr(P, "need_disk", lambda *a: None)
    P.download("http://x", dest, 7, hashlib.sha256(b"weights").hexdigest())
    assert dest.read_bytes() == b"weights"
