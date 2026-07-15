# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""``run_async_in_sync`` must carry the caller's contextvars into the worker
thread it spawns for the async→sync-from-async case.

When ``run_async_in_sync`` is called from a running event loop
(``AsyncContext.ASYNC``) it runs the coroutine on a brand-new worker thread with
its own event loop. A fresh thread starts with an EMPTY contextvars context, so
without copying the caller's context, request-scoped state the coroutine reads —
tenant/user identity, the OTEL trace context — is silently lost. This is exactly
what dropped the per-request headers of an MCP client loaded synchronously from
an async request handler.
"""

import contextvars

import pytest

from pyagentspec.adapters.langgraph.mcp_utils import (
    AsyncContext,
    get_execution_context,
    run_async_in_sync,
)

_probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="")


@pytest.mark.anyio
async def test_run_async_in_sync_propagates_contextvars_across_worker_thread() -> None:
    # Being inside the event loop, this is the ASYNC case — the one that spawns
    # the worker thread; the propagation gap only exists there.
    assert get_execution_context() is AsyncContext.ASYNC

    async def read_probe() -> str:
        return _probe.get()

    token = _probe.set("azaaza")
    try:
        assert run_async_in_sync(read_probe) == "azaaza"
    finally:
        _probe.reset(token)


@pytest.mark.anyio
async def test_run_async_in_sync_when_async_library_marker_is_set() -> None:
    # Simulate an anyio-managed caller: the sniffio async-library marker is set,
    # so copy_context() carries it into the worker thread. A naive anyio.run()
    # there raises "Already running <lib> in this thread"; the fix must clear the
    # inherited marker AND still propagate the caller's contextvars.
    from sniffio import current_async_library_cvar

    async def read_probe() -> str:
        return _probe.get()

    marker = current_async_library_cvar.set("asyncio")
    probe = _probe.set("azaaza")
    try:
        assert run_async_in_sync(read_probe) == "azaaza"
    finally:
        _probe.reset(probe)
        current_async_library_cvar.reset(marker)


def test_run_async_in_sync_runs_in_plain_sync_context() -> None:
    # Sanity: the synchronous case (no loop) already shares the caller's context,
    # so this passed before the fix too — it guards against a regression that
    # would break the common path.
    async def read_probe() -> str:
        return _probe.get()

    token = _probe.set("sync-tenant")
    try:
        assert run_async_in_sync(read_probe) == "sync-tenant"
    finally:
        _probe.reset(token)
