import os

from tt_moeoffload import preflight


def host(tmp_path, kmd="2.11.0", fw="19.16.0.0", kind="p150a", iommu="DMA-FQ", pages="4",
         flags="fpu avx512f avx512bw avx512_vnni"):
    r = tmp_path
    card = r / "sys/class/tenstorrent/tenstorrent!0"
    group = r / "sys/kernel/iommu_groups/49"
    (r / "sys/devices/card").mkdir(parents=True)
    card.parent.mkdir(parents=True)
    card.mkdir()
    (card / "tt_card_type").write_text(kind + "\n")
    (card / "tt_fw_bundle_ver").write_text(fw + "\n")
    os.symlink(r / "sys/devices/card", card / "device")
    if iommu:
        group.mkdir(parents=True)
        (group / "type").write_text(iommu + "\n")
        os.symlink(group, r / "sys/devices/card/iommu_group")
    (r / "sys/module/tenstorrent").mkdir(parents=True)
    (r / "sys/module/tenstorrent/version").write_text(kmd + "\n")
    hp = r / "sys/kernel/mm/hugepages/hugepages-1048576kB"
    hp.mkdir(parents=True)
    (hp / "nr_hugepages").write_text(pages + "\n")
    (r / "dev/tenstorrent").mkdir(parents=True)
    (r / "dev/tenstorrent/0").write_text("")
    (r / "dev/hugepages-1G").mkdir()
    (r / "proc").mkdir()
    (r / "proc/cpuinfo").write_text(f"processor\t: 0\nflags\t\t: {flags}\n")
    return str(r)


def test_a_good_host_passes(tmp_path):
    root = host(tmp_path)
    assert preflight.check_cards(root) == [] and preflight.check_cpu(root) == []


def test_identity_iommu_is_refused(tmp_path):
    (p,) = preflight.check_cards(host(tmp_path, iommu="identity"))
    assert "IOMMU group 49 is identity" in p and "DMA-FQ" in p and "tools/iommu" in p


def test_iommu_off_is_refused(tmp_path):
    (p,) = preflight.check_cards(host(tmp_path, iommu=None))
    assert "IOMMU is off" in p and "intel_iommu=on" in p


def test_old_versions_wormhole_and_no_hugepages(tmp_path):
    found = preflight.check_cards(host(tmp_path, kmd="2.9.1", fw="18.10.0.0", kind="n300", pages="0"))
    assert len(found) == 4
    assert any("tt-kmd 2.9.1" in p for p in found) and any("n300" in p for p in found)
    assert any("firmware 18.10.0.0" in p for p in found) and any("hugepages" in p for p in found)


def test_no_card(tmp_path):
    assert "no Tenstorrent card" in preflight.check_cards(str(tmp_path))[0]


def test_cpu_without_vnni(tmp_path):
    (p,) = preflight.check_cpu(host(tmp_path, flags="avx512f avx512bw"))
    assert "avx512_vnni" in p


def test_only_the_p150(tmp_path):
    for kind in ("p100a", "p300"):
        (p,) = preflight.check_cards(host(tmp_path / kind, kind=kind))
        assert kind in p and "p150" in p
    assert preflight.check_cards(host(tmp_path / "b", kind="p150b")) == []


def test_ram_need_per_model(tmp_path):
    (tmp_path / "proc").mkdir()
    (tmp_path / "proc/meminfo").write_text("MemTotal: 135000000 kB\nMemAvailable: 120000000 kB\n")
    (p,) = preflight.check_ram("flashnext", str(tmp_path))
    assert "123 GB of RAM available" in p and "flashnext needs about 173 GB" in p
    (tmp_path / "proc/meminfo").write_text("MemAvailable: 700000000 kB\n")
    assert preflight.check_ram("flashnext", str(tmp_path)) == []
