#!/usr/bin/env python3
# Part B's session format (SYSTEM, TOOLS, SESSIONS, session_messages) is derived from Strata's
# tools/agent_turns_bench.py, Copyright (c) 2026 Niko1221 and the Strata contributors, MIT License (LICENSES/).
"""corpus.py - the routing census's corpus, rendered with the model's own chat template as the API server renders it.

Two parts, both apart from what the engine is later measured on, so the expert ranking is not fitted to it:
  A  general: Strata's bench/profile_corpus.json - 45 short prompts over code, maths, chat, writing, science, tools
  B  agentic: coding-agent sessions over pinned public repositories (third_party/: project-maya, llama.cpp, Strata):
     a system prompt, tools, a task, then tool calls whose results are the files at their pinned commits (<= 2,000
     lines per read, as agents' read tools return them). Only each session's final prompt is kept: it contains every
     earlier turn, and the census reads each prompt from an empty cache.

Every input is pinned (pins.json's commits, the GGUF's template), so the output is the same bytes on any host; prepare
checks its sha256 against pins.json's "census_corpus". Rendering is the server's own (Strata's serve/frontend.py
ChatTemplate: jinja2 as transformers renders it) with the template read from the model's GGUF.

    python3 tools/llama/corpus.py <model .gguf (the first shard)> <out prompts.jsonl>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TP = REPO / "third_party"
sys.path.insert(0, str(TP / "llama.cpp" / "gguf-py"))
sys.path.insert(0, str(TP / "strata" / "serve"))

SYSTEM = ("You are an autonomous coding agent working in the user's repository. Use the tools to read files and search "
          "the code before you answer; quote the lines you rely on; keep changes minimal and explain them.")
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search the repository for a regex.",
                                      "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}},
                                                     "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run", "description": "Run a shell command in the repository.",
                                      "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                                     "required": ["cmd"]}}},
]

# (name, checkout under third_party/, task, [(tool, argument, what the agent says before the call)])
SESSIONS = [
    ("maya-dashboard", "maya",
     "The dashboard's top bar sometimes shows the previous model's name after a model switch. Find where the label "
     "comes from and propose a fix.",
     [("search", "model_name", "Let me find where the model name is set."),
      ("read_file", "serve/server.py", "The server owns the engine handle; reading it."),
      ("read_file", "serve/frontend.py", "Now the frontend that renders the top bar."),
      ("read_file", "serve/telemetry.py", "Telemetry may cache the label; checking.")]),
    ("maya-calibrate", "maya",
     "Review how `./maya.sh --calibrate` picks the CPU lane's thread count and whether it handles a single NUMA node "
     "with 128 cores well.",
     [("search", "CPU_LANE", "Searching for the CPU lane setting."),
      ("read_file", "maya.py", "Reading the launcher's calibration code.")]),
    ("llama-offload", "llama.cpp",
     "Explain exactly when the CUDA backend offloads a CPU-resident matmul to the GPU, and how the batch threshold "
     "is chosen.",
     [("search", "op_offload_min_batch", "Finding the threshold."),
      ("read_file", "ggml/src/ggml-cuda/ggml-cuda.cu", "Reading the CUDA backend around it."),
      ("read_file", "ggml/src/ggml-backend.cpp", "The scheduler decides the split; reading it.")]),
    ("llama-ds41", "llama.cpp",
     "Walk through the deepseek41 graph: where the Engram table is looked up and where the routed experts run. "
     "I want to intercept the MoE path.",
     [("read_file", "src/models/deepseek41.cpp", "Reading the model graph."),
      ("search", "build_moe_ffn", "Finding the shared MoE builder."),
      ("read_file", "src/llama-graph.cpp", "Reading build_moe_ffn.")]),
    ("llama-server-slots", "llama.cpp",
     "Add a server option that pins prompt-cache slots for a given conversation id. Where would it go?",
     [("read_file", "tools/server/server-context.cpp", "Reading how slots are chosen."),
      ("read_file", "common/arg.cpp", "Arguments are declared here; reading.")]),
    # the one prompt that fits the MTP layer's dense attention (g6_mtp.py's LONG) and the served check's prefill prompt
    ("strata-drafter", "strata",
     "The suffix drafter stops proposing drafts after a long repeated span. Find where the match length is capped "
     "and how the controller's tests exercise the drafter.",
     [("search", "max_match", "Finding the match-length cap."),
      ("read_file", "include/strata/spec/suffix_drafter.hpp", "Reading the drafter's interface."),
      ("read_file", "src/spec/suffix_drafter.cpp", "Now its implementation."),
      ("read_file", "src/spec/controller_test.cpp", "And the controller's tests, which drive it."),
      ("run", "ctest --test-dir build -R controller", "Running those tests first.")]),
]
RUN_RESULT = "100% tests passed, 0 tests failed out of 1"   # `run`: a fixed result keeps the corpus deterministic


def tool_result(root: Path, tool: str, arg: str) -> str:
    if tool == "read_file":
        p = root / arg
        if not p.is_file():
            raise SystemExit(f"{p} is missing: is third_party/{root.name} at its pins.json commit? (tt/env/setup.sh)")
        lines = p.read_text(errors="replace").splitlines()
        return "\n".join(f"{i + 1}\t{ln}" for i, ln in enumerate(lines[:2000]))
    if tool == "search":   # git grep: path order, the same on every filesystem
        out = subprocess.run(["git", "grep", "-n", "-E", arg, "--", "*.py", "*.cpp", "*.cu", "*.h", "*.hpp", "*.md"],
                             cwd=root, capture_output=True, text=True)
        if out.returncode > 1:
            raise SystemExit(f"git grep in {root}: {out.stderr.strip()}")
        return "\n".join(out.stdout.splitlines()[:200]) or "no matches"
    return RUN_RESULT


def session_messages(name: str, root: Path, task: str, steps) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
    for i, (tool, arg, before) in enumerate(steps):
        cid = f"call_{name}_{i}"
        key = {"read_file": "path", "search": "pattern", "run": "cmd"}[tool]
        msgs.append({"role": "assistant", "content": before, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": tool, "arguments": {key: arg}}}]})
        msgs.append({"role": "tool", "tool_call_id": cid, "content": tool_result(root, tool, arg)})
    return msgs


def template_of(gguf_path: str) -> str:
    from gguf import GGUFReader
    f = GGUFReader(gguf_path).fields["tokenizer.chat_template"]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1].strip())
    from frontend import ChatTemplate
    out = Path(sys.argv[2])
    tpl_file = out.with_name("chat_template.jinja")
    out.parent.mkdir(parents=True, exist_ok=True)
    tpl_file.write_text(template_of(sys.argv[1]), encoding="utf-8")
    tpl = ChatTemplate(tpl_file)
    rows = []
    corpus = json.loads((TP / "strata" / "bench" / "profile_corpus.json").read_text())
    for i, item in enumerate(corpus):
        rows.append({"id": f"A.{item['d']}.{i}", "part": "A", "prompt": tpl.render([{"role": "user", "content": item["p"]}])})
    for name, repo, task, steps in SESSIONS:
        rows.append({"id": f"B.{name}", "part": "B",
                     "prompt": tpl.render(session_messages(name, TP / repo, task, steps), tools=TOOLS)})
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out)
    sizes = {r["id"]: len(r["prompt"]) for r in rows if r["part"] == "B"}
    print(f"{len(rows)} prompts -> {out}; part B chars: {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
