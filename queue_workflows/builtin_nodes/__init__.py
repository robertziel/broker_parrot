"""Built-in node modules shipped WITH the engine.

A stored ``node_module`` name resolves against the host's ``node_module_package`` first
(host nodes always win); when that import fails and the name matches a module here, the
engine falls back to the builtin — so capabilities like the ComfyUI render job work for
every project with zero host glue. See ``EngineConfig.resolve_node_module``.
"""
