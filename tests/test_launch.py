import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tt_moeoffload import launch

STUB = Path(__file__).with_name("stub_engine.py")



@pytest.fixture(autouse=True)
def own_data_root(tmp_path, monkeypatch):
    """the launcher makes <data root>/home: a temporary root, not the user's default one"""
    monkeypatch.setenv("TT_MOEOFFLOAD_DATA", str(tmp_path / "data-root"))


def test_container_argv_fake_leaves_the_card_out(tmp_path):
    argv = launch.container_argv(STUB, ["--serve"], fake=True, name="t", data=tmp_path, cwd=Path("/"), img="IMG")
    assert not set(launch.CARD_ARGS) & set(argv)
    assert f"--workdir={launch.REPO}" in argv                 # a cwd outside every mount falls back to the repo
    assert f"--env=HOME={tmp_path.resolve() / 'home'}" in argv and f"--env=TT_MOEOFFLOAD_DATA={tmp_path.resolve()}" in argv
    assert "--network=host" not in argv and not any(a == f"--volume={Path.home()}:{Path.home()}" for a in argv)
    assert argv[-3:] == ["IMG", "-c", f"source {launch.TTENV} && exec python {STUB} --serve"]


def test_container_argv_card_mounts_and_limits(tmp_path):
    models = tmp_path / "models"
    argv = launch.container_argv(STUB, [], fake=False, name="t", cpus="0-7", memory_max=1 << 30, mounts=[models],
                                 env={"A": "1"}, data=tmp_path / "d", cwd=models, img="IMG")
    assert set(launch.CARD_ARGS) <= set(argv)
    assert f"--volume={models}:{models}" in argv and f"--workdir={models}" in argv
    assert "--cpuset-cpus=0-7" in argv and f"--memory={1 << 30}" in argv and "--env=A" in argv
    assert not any(x.startswith("--env=A=") for x in argv)            # values stay out of the argv (and the log)


def test_env_passes_prefixed_vars(monkeypatch):
    for k in ("STRATA_X", "TTC_ROWS", "W51_EXPERTS", "TT_CTX", "OTHER", "PATH_X"):
        monkeypatch.setenv(k, "1")
    env = launch.passed_env(["K=a=b"])
    assert {"STRATA_X", "TTC_ROWS", "W51_EXPERTS", "TT_CTX", "K"} <= set(env) and env["K"] == "a=b"
    monkeypatch.setenv("STRATA_API_KEY", "s3cret")
    assert "STRATA_API_KEY" not in launch.passed_env([])                 # a server's secrets are not forwarded
    assert "OTHER" not in env and "PATH_X" not in env and "HOME" not in env


def test_engine_args_split(monkeypatch):
    seen = {}
    monkeypatch.setattr(launch, "container_argv", lambda engine, args, **kw: seen.update(args=args) or ["true"])
    monkeypatch.setattr(launch, "running", lambda name: False)
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: None)
    assert launch.main(["--serve", "--fake", "--cpus", "0-3", "--", "--name", "x"]) == 0
    assert seen["args"] == ["--serve", "--name", "x", "--fake"]   # after --: the engine's, even --name


def test_refuses_when_preflight_fails(monkeypatch, capsys):
    monkeypatch.setattr(launch.preflight, "problems", lambda model=None: ["no card"])
    assert launch.main(["--serve"]) == 2
    assert "no card" in capsys.readouterr().err


def _image_present():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", launch.image()], capture_output=True).returncode == 0


docker = pytest.mark.skipif(not _image_present(), reason="docker or the pinned tt-metal image is missing")


def _start(name):
    return subprocess.Popen([sys.executable, str(launch.REPO / "tt_moeoffload" / "launch.py"), "--serve", "--fake",
                             "--engine", str(STUB), "--name", name, "--grace", "3"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _container_gone(name):
    r = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^{name}$"], capture_output=True, text=True)
    return r.stdout.strip() == ""


@docker
def test_protocol_round_trip_in_the_container():
    p = _start("tt-moeoffload-test-rt")
    try:
        assert p.stdout.readline().startswith("INFO stub=1 args=--serve,--fake")
        assert p.stdout.readline().strip() == "READY 4096 stop"
        p.stdin.write("GEN 2 1,2,3\n")
        p.stdin.flush()
        assert [p.stdout.readline().strip() for _ in range(3)] == ["T 1", "T 2", "DONE 2 3 0 0 length 0 0 0 0 0 0 0 0 3"]
        p.stdin.write("QUIT\n")
        p.stdin.close()
        assert p.wait(timeout=30) == 0
    finally:
        p.kill()
    assert _container_gone("tt-moeoffload-test-rt")


@docker
def test_sigterm_reaches_the_engine_and_ends_the_container():
    p = _start("tt-moeoffload-test-term")
    try:
        assert p.stdout.readline().startswith("INFO")
        p.stdout.readline()
        t0 = time.time()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=30) == 0 and time.time() - t0 < 10
        assert "BYE" in p.stderr.read()
    finally:
        p.kill()
    assert _container_gone("tt-moeoffload-test-term")


card = pytest.mark.skipif(os.environ.get("TT_MOEOFFLOAD_CARD") != "1", reason="TT_MOEOFFLOAD_CARD=1 runs the card tests")


def _start_card(name):
    return subprocess.Popen([sys.executable, str(launch.REPO / "tt_moeoffload" / "launch.py"), "--serve", "--engine",
                             str(STUB.with_name("card_engine.py")), "--name", name, "--grace", "15"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


@card
def test_card_path_opens_the_card_and_serves():
    p = _start_card("tt-moeoffload-test-card")
    try:
        assert p.stdout.readline().startswith("INFO card=ok matmul=ok sockets=ok")
        assert p.stdout.readline().strip() == "READY 4096 stop"
        p.stdin.write("GEN 1 9,8\nQUIT\n")
        p.stdin.close()
        assert [p.stdout.readline().strip() for _ in range(2)] == ["T 9", "DONE 1 2 0 0 length 0 0 0 0 0 0 0 0 2"]
        assert p.wait(timeout=60) == 0
        assert "CLOSED" in p.stderr.read()
    finally:
        p.kill()
    assert _container_gone("tt-moeoffload-test-card")


@card
def test_card_closes_on_sigterm():
    p = _start_card("tt-moeoffload-test-card-term")
    try:
        assert p.stdout.readline().startswith("INFO card=ok")
        p.stdout.readline()
        p.send_signal(signal.SIGTERM)
        assert p.wait(timeout=60) == 0
        assert "CLOSED" in p.stderr.read()
    finally:
        p.kill()
    assert _container_gone("tt-moeoffload-test-card-term")


def test_engine_paths_are_mounted_and_nested_ones_once(tmp_path):
    (tmp_path / "m" / "sub").mkdir(parents=True)
    ids = tmp_path / "p" / "ids.i32"
    argv = launch.container_argv(STUB, ["--model-dir", str(tmp_path / "m" / "sub"), "--ids", str(ids)], fake=True,
                                 name="t", mounts=[tmp_path / "m"], data=tmp_path / "d", cwd=tmp_path, img="IMG")
    vols = [a for a in argv if a.startswith("--volume=") and not a.startswith("--volume=/etc")]
    assert f"--volume={tmp_path / 'm'}:{tmp_path / 'm'}" in vols and f"--volume={tmp_path / 'p'}:{tmp_path / 'p'}" in vols
    assert not any("/m/sub:" in v for v in vols)              # inside a mounted folder: already there


def test_a_running_container_of_the_same_name_refuses(monkeypatch, capsys):
    monkeypatch.setattr(launch, "running", lambda name: name == "busy")
    assert launch.main(["--fake", "--name", "busy"]) == 2
    assert "busy is running" in capsys.readouterr().err


def test_links_in_the_data_root_are_mounted(tmp_path):
    (tmp_path / "elsewhere" / "census").mkdir(parents=True)
    (tmp_path / "d" / "census").mkdir(parents=True)
    (tmp_path / "d" / "census" / "m").symlink_to(tmp_path / "elsewhere" / "census")
    argv = launch.container_argv(STUB, [], fake=True, name="t", data=tmp_path / "d", cwd=tmp_path, img="IMG")
    assert f"--volume={tmp_path / 'elsewhere' / 'census'}:{tmp_path / 'elsewhere' / 'census'}:ro" in argv   # read only


def test_the_engines_model_reaches_the_ram_check(monkeypatch):
    seen = []
    monkeypatch.setattr(launch.preflight, "problems", lambda model=None: seen.append(model) or ["stop"])
    launch.main(["--serve", "--", "--model", "glm"])
    launch.main(["--serve", "--", "--max-context", "8192"])
    launch.main(["--engine", str(STUB)])
    assert seen == ["glm", "flashnext", None]


def test_refuses_to_mount_system_folders(tmp_path):
    with pytest.raises(SystemExit):
        launch.container_argv(STUB, ["--out", "/x.f32"], fake=True, name="t", data=tmp_path, img="IMG")
    with pytest.raises(SystemExit):
        launch.container_argv(STUB, ["--ids", "/etc/passwd"], fake=True, name="t", data=tmp_path, img="IMG")
