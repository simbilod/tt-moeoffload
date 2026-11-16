import pytest

from tt.cpulane import lane_cpus


def sysfs(tmp_path, cores: int, smt: int):
    """A host with `cores` physical cores, CPU c's siblings c, c + cores, ... (AMD's and Intel's numbering)."""
    for c in range(cores * smt):
        d = tmp_path / f"cpu{c}/topology"
        d.mkdir(parents=True)
        d.joinpath("thread_siblings_list").write_text(",".join(str(c % cores + k * cores) for k in range(smt)) + "\n")
    return str(tmp_path)


def test_a_two_way_smt_host(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_MOEOFFLOAD_LANE_CPUS", raising=False)
    root = sysfs(tmp_path, 128, 2)
    assert lane_cpus(range(256), root) == list(range(8, 128))      # what Lane(min(120, len - 8), cpus[8:]) picked
    assert lane_cpus(range(128), root) == list(range(8, 128))      # --cpus all
    assert lane_cpus(range(128, 256), root) == list(range(136, 256))  # only the second siblings: one per core
    assert lane_cpus(range(32), root) == list(range(8, 32))


def test_override_and_refusals(tmp_path, monkeypatch):
    root = sysfs(tmp_path, 16, 2)
    monkeypatch.setenv("TT_MOEOFFLOAD_LANE_CPUS", "4-7,10")
    assert lane_cpus(range(32), root) == [4, 5, 6, 7, 10]
    with pytest.raises(ValueError):
        lane_cpus(range(8), root)                                    # 10 is not ours
    monkeypatch.delenv("TT_MOEOFFLOAD_LANE_CPUS")
    with pytest.raises(ValueError):
        lane_cpus([0, 1, 16, 17], root)                              # two cores
    assert lane_cpus(range(12), str(tmp_path / "none")) == [8, 9, 10, 11]   # no topology: each CPU a core
