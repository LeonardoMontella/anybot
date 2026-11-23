"""End-to-end: MCP client → MCP adapter → studio HTTP API → simulation, all in-process."""

import httpx
import pytest
from mcp.client import Client

from anybot import server, studio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def wired():
    """Point the MCP adapter at the studio app without opening sockets."""
    server._http["client"] = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=studio.app), base_url="http://studio", timeout=120
    )
    yield
    server._http["client"] = None
    studio.host.close()
    studio.host.sim = None


@pytest.mark.anyio
async def test_mcp_drives_the_studio_robot(wired):
    async with Client(server.mcp) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"load_robot", "move_end_effector", "gripper", "snapshot", "drive"} <= names

        res = await client.call_tool("load_robot", {"robot": "panda"})
        assert not res.is_error, res.content
        assert '"category":"arm"' in res.content[0].text.replace(" ", "")

        res = await client.call_tool("spawn_object", {"x": 0.5, "y": 0.0, "color": "blue"})
        assert not res.is_error

        res = await client.call_tool("snapshot", {})
        assert res.content[0].type == "image"

        res = await client.call_tool("drive", {"forward_speed": 0.2})
        assert res.is_error and "no wheels" in res.content[0].text

    # Everything Claude did is in the studio's activity log, tagged with its source.
    tools = [(e["tool"], e["source"], e["status"]) for e in studio.host.events]
    assert ("load_robot", "claude", "ok") in tools
    assert ("drive", "claude", "error") in tools


@pytest.mark.anyio
async def test_studio_state_and_ui(wired):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=studio.app), base_url="http://studio") as c:
        assert (await c.get("/")).text.startswith("<!doctype html>")
        assert (await c.get("/api/state")).json()["robot"] is None
        r = await c.post("/api/tools/load_robot", json={"robot": "go2"})
        assert r.json()["ok"]
        s = (await c.get("/api/state")).json()
        assert s["robot"]["category"] == "quadruped" and len(s["joints"]) == 12
        r = await c.post("/api/tools/gripper", json={"action": "open"})
        assert r.status_code == 400 and "no arm" in r.json()["error"]
        await c.post("/api/camera", json={"reset": True})
        cam = (await c.post("/api/camera", json={"d_azimuth": 10, "zoom": 1.1})).json()
        assert cam["azimuth"] == 145.0
