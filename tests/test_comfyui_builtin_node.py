"""ComfyUI as a BUILT-IN queue citizen: one node_job = one whole ComfyUI render.

Any project can enqueue a gpu node_job with ``node_module="comfyui"`` and NO host glue:
the engine resolves the module from ``queue_workflows.builtin_nodes``, the gpu no-model
guard exempts it by default (builtin pool module), and its single ``run()`` performs the
entire render — take the box for ComfyUI (evict rivals + start + ready), POST the graph,
poll, fetch the outputs into the job's out dir. The queue therefore accounts the whole
ComfyUI usage as ONE job: claimed, leased, retried, observed like any other node_job.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import queue_workflows
from queue_workflows import config
from queue_workflows.builtin_nodes import comfyui as builtin_comfyui


@pytest.fixture(autouse=True)
def _reset():
    cfg = config.get_config()
    saved_pkg = cfg.node_module_package
    saved_pool = cfg.vlm_pool_node_modules
    cfg.comfyui_start_fn = None
    yield
    cfg.node_module_package = saved_pkg
    cfg.vlm_pool_node_modules = saved_pool
    cfg.comfyui_start_fn = None


# ── the builtin module resolves with ZERO host config ────────────────────────


def test_resolve_node_module_falls_back_to_builtin_comfyui():
    cfg = config.get_config()
    cfg.node_module_package = "some_host_pkg.nodes"     # host package has no 'comfyui'
    mod = cfg.resolve_node_module("comfyui")
    assert mod is builtin_comfyui


def test_host_package_module_still_wins_over_builtin():
    """A host that ships its OWN nodes keeps them — builtin is a FALLBACK, not an override."""
    cfg = config.get_config()
    cfg.node_module_package = "queue_workflows"          # any importable package
    # 'comfyui' resolves to queue_workflows.comfyui (the host package hit), not builtin_nodes
    mod = cfg.resolve_node_module("comfyui")
    import queue_workflows.comfyui as qc
    assert mod is qc


# ── the gpu no-model guard exempts the builtin by default ────────────────────


def test_builtin_comfyui_is_a_default_pool_module():
    cfg = config.get_config()
    assert "comfyui" in cfg.effective_vlm_pool_node_modules


def test_host_pool_modules_union_with_builtin_not_replace():
    queue_workflows.configure(vlm_pool_node_modules={"my_vlm_node"})
    cfg = config.get_config()
    assert {"comfyui", "my_vlm_node"} <= set(cfg.effective_vlm_pool_node_modules)


def _wf_with_gpu_node(module: str):
    """A minimal workflow + pipeline schema pair holding ONE gpu no-model node."""
    wf = {"name": "t", "mode": "node",
          "steps": [{"id": "p", "kind": "pipeline", "pipeline": "t", "inputs": {}}]}
    schema = {"name": "t",
              "nodes": [{"id": "r", "node": module, "depends_on": [], "gpu": True}]}
    queue_workflows.set_workflow_provider(lambda n: wf, lambda n: schema)
    return wf


def test_dispatcher_guard_accepts_a_no_model_gpu_comfyui_node():
    from queue_workflows import dispatcher
    wf = _wf_with_gpu_node("comfyui")
    dispatcher._assert_gpu_nodes_declare_model(wf)      # must not raise


def test_dispatcher_guard_still_rejects_unknown_no_model_gpu_nodes():
    from queue_workflows import dispatcher
    wf = _wf_with_gpu_node("mystery_gpu_thing")
    with pytest.raises(ValueError, match="mystery_gpu_thing"):
        dispatcher._assert_gpu_nodes_declare_model(wf)


# ── one node_job = the WHOLE render ──────────────────────────────────────────


def _graph():
    return {"52": {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}}


def test_run_acquires_box_submits_and_fetches_in_one_job(tmp_path):
    order = []

    def acquire_fn(**kw):
        order.append("acquire")
        return []

    def submit_fn(url, graph, **kw):
        order.append("submit")
        assert graph == _graph()
        return [{"filename": "out_00001_.mp4", "subfolder": "", "type": "output"}]

    def fetch_fn(url, f, dest):
        order.append("fetch")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"MP4")
        return Path(dest)

    res = builtin_comfyui.run(
        inputs={"graph": _graph(), "comfyui_url": "http://box-a:8188"},
        out=str(tmp_path),
        acquire_fn=acquire_fn, submit_fn=submit_fn, fetch_fn=fetch_fn,
    )
    assert order == ["acquire", "submit", "fetch"]       # whole render inside the one job
    files = res["summary"]["files"]
    assert files and files[0].endswith("out_00001_.mp4")
    assert (tmp_path / "out_00001_.mp4").read_bytes() == b"MP4"


def test_run_reads_graph_from_file_when_given_a_path(tmp_path):
    gpath = tmp_path / "g.json"
    gpath.write_text(json.dumps(_graph()))
    seen = {}

    res = builtin_comfyui.run(
        inputs={"graph_path": str(gpath), "comfyui_url": "http://box-a:8188"},
        out=str(tmp_path / "o"),
        acquire_fn=lambda **k: [],
        submit_fn=lambda u, g, **k: seen.setdefault("g", g) and [] or
            [{"filename": "f.png", "subfolder": "", "type": "output"}],
        fetch_fn=lambda u, f, d: (Path(d).parent.mkdir(parents=True, exist_ok=True), Path(d).write_bytes(b"X"), Path(d))[-1],
    )
    assert seen["g"] == _graph()
    assert res["summary"]["files"]


def test_run_fails_loud_without_url_or_graph(tmp_path):
    with pytest.raises(RuntimeError, match="comfyui_url"):
        builtin_comfyui.run(inputs={"graph": _graph()}, out=str(tmp_path),
                            acquire_fn=lambda **k: [], submit_fn=lambda u, g, **k: [],
                            fetch_fn=lambda u, f, d: d)
    with pytest.raises(RuntimeError, match="graph"):
        builtin_comfyui.run(inputs={"comfyui_url": "http://x:8188"}, out=str(tmp_path),
                            acquire_fn=lambda **k: [], submit_fn=lambda u, g, **k: [],
                            fetch_fn=lambda u, f, d: d)


def test_run_propagates_render_failure_so_the_job_goes_red(tmp_path):
    from queue_workflows.comfyui import ComfyUIError

    def submit_fn(url, graph, **kw):
        raise ComfyUIError("run errored")

    with pytest.raises(ComfyUIError):
        builtin_comfyui.run(
            inputs={"graph": _graph(), "comfyui_url": "http://box-a:8188"},
            out=str(tmp_path),
            acquire_fn=lambda **k: [], submit_fn=submit_fn, fetch_fn=lambda u, f, d: d,
        )
