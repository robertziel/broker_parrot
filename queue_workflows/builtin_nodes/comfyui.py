"""Built-in node: ONE node_job = ONE whole ComfyUI render.

Enqueue a gpu node_job with ``node_module="comfyui"`` (no host glue, no ``model:`` —
the builtin is a default pool-lane module) and this single job performs the entire
render lifecycle inside the queue's accounting:

  1. take the box for ComfyUI — :func:`queue_workflows.comfyui.ensure_comfyui_box`
     evicts every rival serving kind off the card (ollama / vLLM / native warm model),
     starts ComfyUI via the host lifecycle lever, waits until it answers;
  2. POST the graph (``inputs.graph`` inline, or ``inputs.graph_path`` a JSON file)
     via :func:`queue_workflows.comfyui.submit_workflow`;
  3. fetch every output file over ``/view`` into the job's ``out`` dir.

Failures raise → the node_job goes red and broker's lease/retry machinery owns the
re-run; a success returns the fetched filenames. So ComfyUI usage is always queued,
claimed, leased, retried and observed as one first-class node_job.

Lane routing: on a host with NO configured ``vlm_pool_node_modules`` the pool lane
claims every no-model gpu row, comfyui included. On a host WITH a configured set,
a ``comfyui`` row not in that set is claimed by the INLINE (concurrency-1) lane —
also correct for a GPU renderer, since ComfyUI serializes prompts anyway; add
"comfyui" to your set if you want it in the PAR pool instead.

inputs:
  comfyui_url   base URL of the box's ComfyUI (falls back to QUEUE_WORKFLOWS_COMFYUI_URL).
  graph         API-format prompt graph (dict), OR
  graph_path    path to a JSON file holding it.
  timeout_s     optional render deadline (default 1800).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from queue_workflows import comfyui as _comfyui


def _default_fetch(base_url: str, f: dict, dest: Path) -> Path:
    q = urllib.parse.urlencode(
        {"filename": f.get("filename", ""), "subfolder": f.get("subfolder", ""),
         "type": f.get("type", "output")}
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/view?{q}", timeout=120) as resp:  # noqa: S310
        dest.write_bytes(resp.read())
    return dest


def run(
    *,
    inputs=None,
    out=None,
    status_callback=None,
    acquire_fn: Callable[..., list] | None = None,
    submit_fn: Callable[..., list[dict]] | None = None,
    fetch_fn: Callable[[str, dict, Path], Path] | None = None,
    **_kw,
) -> dict[str, Any]:
    inputs = inputs or {}
    url = (inputs.get("comfyui_url") or "").strip()
    if not url:
        from queue_workflows.config import get_config
        from queue_workflows.envcompat import env_get
        url = (env_get("QUEUE_WORKFLOWS_COMFYUI_URL") or get_config().comfyui_url or "").strip()
    if not url:
        raise RuntimeError(
            "builtin comfyui node: no comfyui_url (inputs.comfyui_url or "
            "QUEUE_WORKFLOWS_COMFYUI_URL) — the box must declare its ComfyUI"
        )

    graph = inputs.get("graph")
    if graph is None and inputs.get("graph_path"):
        graph = json.loads(Path(inputs["graph_path"]).read_text())
    if not isinstance(graph, dict) or not graph:
        raise RuntimeError(
            "builtin comfyui node: no graph (inputs.graph dict or inputs.graph_path JSON file)"
        )

    acquire = acquire_fn or _comfyui.ensure_comfyui_box
    submit = submit_fn or _comfyui.submit_workflow
    fetch = fetch_fn or _default_fetch

    # 1. the box is ComfyUI's for this job — rivals evicted, server up.
    acquire(comfyui_url=url, label="builtin-comfyui")

    # 2. the render itself.
    timeout_s = float(inputs.get("timeout_s") or 1800.0)
    outputs = submit(url, graph, timeout_s=timeout_s) or []

    # 3. land every output in the job's out dir (the queue's artifact channel).
    dest_dir = Path(out) if out is not None else Path(".")
    fetched: list[str] = []
    for f in outputs:
        name = f.get("filename") or "output.bin"
        fetch(url, f, dest_dir / name)
        fetched.append(name)

    payload = {"comfyui_url": url, "files": fetched, "n_outputs": len(fetched)}
    return {"summary": payload,
            "primary_file": fetched[-1] if fetched else None}
