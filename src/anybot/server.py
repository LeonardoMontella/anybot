"""anybot MCP server: an adapter that lets any MCP client (Claude Desktop, Claude Code, ...) drive
the robot running in anybot studio. The studio is started automatically and opens in the browser,
so you watch the robot do what you ask.

Run:  anybot-mcp            (used by MCP clients over stdio)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import Field

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

STUDIO_URL = os.environ.get("ANYBOT_STUDIO_URL", f"http://127.0.0.1:{os.environ.get('ANYBOT_PORT', '8765')}")

INSTRUCTIONS = """\
anybot simulates robots in MuJoCo (realistic rigid-body physics, real time): one robot, or a team.
The user watches the robot live in anybot studio (a browser tab), so act step by step.
Workflow: list_robots → load_robot → act → snapshot to check the result.
load_robot puts the robot in a warehouse by default and returns its zones, objects and tasks:
use run_task for whole jobs (sort, stack, patrol, carry_to_b) and go_to_zone / pick_object / place_object for steps.
pick_object / place_object work for every arm: parallel grippers grasp by friction, multi-finger hands
close their fingers on an assisted grasp, fingerless arms hold suction-style; walking and wheeled robots
first walk or drive up to the object or the table. Robots: arms, AMRs (amr, tidybot with arm),
quadrupeds, the G1 humanoid (walks, carries boxes with both hands).
Conventions: world frame, metres, z up, the robot base is near the origin; joint angles in radians.
Only the skills listed by load_robot apply to the loaded robot. The G1 humanoid balances and walks
with a learned policy. stop() halts any running task at once.
World 'apartment' is a digital twin built from a real 3D scan (import_scene makes more from a phone video or scan): navigate_to / go_to_zone find a path
around walls and furniture, so a point in another room works. Snapshots there look best from above (elevation -60).
Sensors: get_scan (2D lidar) and camera_image (the robot's own camera) show what the robot perceives;
navigation plans on the world's collision map plus what the lidar sees, so it goes around walls, through doors
and around boxes or other robots. The user can watch lidar, camera, map and path live in the studio's Sensors view.
Teams: load_team('amr_line' or 'mixed_line') puts several robots in the warehouse around a conveyor line
(loader → conveyor → sorter → tote on a carrier → dock). run_task('production') runs the whole line with all
robots working at the same time; to command one robot pass robot='loader' / 'sorter' / 'carrier' to any robot
tool (or select_robot once). Zones belt_in / belt_out / tote are the line's stations.
For arms with a gripper prefer pick_object / place_object over hand-made gripper sequences. Look at a snapshot before and after
manipulation to verify what happened instead of assuming success.
"""

mcp = MCPServer("anybot", instructions=INSTRUCTIONS)
_http: dict = {"client": None}


def _studio_up() -> bool:
    try:
        return httpx.get(f"{STUDIO_URL}/api/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_studio() -> None:
    """Start anybot studio in the background (it outlives this process) unless it already runs."""
    if _studio_up():
        return
    log = Path.home() / ".anybot" / "studio.log"
    log.parent.mkdir(exist_ok=True)
    subprocess.Popen(
        [sys.executable, "-m", "anybot.studio"],
        stdout=log.open("a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
    )
    for _ in range(100):
        if _studio_up():
            return
        time.sleep(0.2)
    raise RuntimeError(f"anybot studio did not start; see {log}")


def _client() -> httpx.AsyncClient:
    if _http["client"] is None:
        _http["client"] = httpx.AsyncClient(base_url=STUDIO_URL, timeout=180.0)
    return _http["client"]


async def _call(name: str, args: dict | None = None, timeout: float | None = None):
    args = {k: v for k, v in (args or {}).items() if v is not None}
    kw = {"timeout": timeout} if timeout else {}
    try:
        r = await _client().post(f"/api/tools/{name}", json=args, headers={"x-anybot-source": "claude"}, **kw)
    except httpx.ConnectError:
        ensure_studio()
        r = await _client().post(f"/api/tools/{name}", json=args, headers={"x-anybot-source": "claude"}, **kw)
    if r.headers.get("content-type", "").startswith("image/"):
        return r.content
    body = r.json()
    if not body.get("ok"):
        raise ToolError(body.get("error", "unknown error"))
    return body["result"]


@mcp.tool()
async def list_robots(query: Annotated[str, Field(description="Optional substring filter, e.g. 'arm', 'go', 'hand'")] = "") -> list[str]:
    """List the robots bundled with anybot (MuJoCo Menagerie). Any .xml/.urdf path also works in load_robot."""
    return await _call("list_robots", {"query": query})


@mcp.tool()
async def load_robot(
    robot: Annotated[str, Field(description="Bundled robot name (e.g. 'panda', 'go2', 'g1') or absolute path to a .xml/.urdf")],
    pin_base: Annotated[bool, Field(description="Fix a legged robot's base in place (use for humanoid arm tasks)")] = False,
    floating: Annotated[bool, Field(description="URDF only: give the root a free joint (legged/mobile robots)")] = False,
    world: Annotated[str, Field(description="'warehouse': racks, zones, work tables; 'lab': bright lab with tables and 3D-scanned real objects; 'apartment': digital twin of a real scanned apartment (rooms, furniture; robots plan paths around walls, spots spot_1..4); 'bench': a fixed manipulation cell (used automatically for pinned humanoids); 'empty'; or the name of a place imported with import_scene")] = "warehouse",
) -> dict:
    """Load a robot into a fresh scene, replacing the current one.

    Returns what anybot understood about it, plus the world's zones, objects and ready-made tasks.
    """
    return await _call("load_robot", {"robot": robot, "pin_base": pin_base, "floating": floating, "world": world})


@mcp.tool()
async def load_team(
    team: Annotated[Literal["amr_line", "mixed_line"], Field(description="'amr_line': Tidybot loader, Panda sorter, AMR carrier; 'mixed_line': G1 humanoid loader, UR5e sorter, Go2 quadruped carrier")] = "amr_line",
) -> dict:
    """Load several robots into one warehouse around a small production line, replacing the current scene.

    Returns the team members (use their names as the robot argument), zones, objects and tasks."""
    return await _call("load_team", {"team": team})


@mcp.tool()
async def select_robot(name: Annotated[str, Field(description="Team member, e.g. loader, sorter, carrier")]) -> dict:
    """Make a team member the default target of robot tools (and the one the studio frames)."""
    return await _call("select_robot", {"name": name})


@mcp.tool()
async def list_tasks() -> list[dict]:
    """Ready-made jobs for the loaded robot in this world (e.g. sort, stack, patrol, go_zone_a)."""
    return await _call("list_tasks")


@mcp.tool()
async def run_task(task: Annotated[str, Field(description="A task id from list_tasks / load_robot")], robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """Run a ready-made task from start to finish; every step shows live in anybot studio."""
    return await _call("run_task", {"task": task, "robot": robot})


@mcp.tool()
async def go_to_zone(zone: Annotated[str, Field(description="Zone name from load_robot's 'zones', e.g. zone_a, dock, pad_b")], robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """Walk/drive to a named floor zone. For an arm: put the held object on that pad (or hover above it)."""
    return await _call("go_to_zone", {"zone": zone, "robot": robot})


@mcp.tool()
async def describe_robot(robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """Full profile of the loaded robot: every joint with limits and actuator type, limbs, cameras, poses."""
    return await _call("describe_robot", {"robot": robot})


@mcp.tool()
async def get_state(robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """Current joint angles (rad), base pose (if free), end-effector positions and object positions (m)."""
    return await _call("get_state", {"robot": robot})


@mcp.tool()
async def set_joint_positions(
    positions: Annotated[dict[str, float], Field(description="Joint name → target angle (rad) or position (m)")],
    duration: Annotated[float, Field(ge=0.1, le=10)] = 1.5,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Move some joints to target positions with a smooth trajectory; other joints hold. Returns all joint angles."""
    return await _call("set_joint_positions", {"positions": positions, "duration": duration, "robot": robot})


@mcp.tool()
async def move_end_effector(
    x: float, y: float, z: float,
    arm: Annotated[str | None, Field(description="Substring of the arm name, e.g. 'left'; default first arm")] = None,
    point_down: Annotated[bool, Field(description="Also orient the gripper to approach from above (for grasping)")] = False,
    duration: Annotated[float, Field(ge=0.2, le=10)] = 2.0,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Move an arm's tool centre point (between the fingers) to world position (x, y, z) in metres using IK."""
    return await _call("move_end_effector", {"x": x, "y": y, "z": z, "arm": arm, "point_down": point_down, "duration": duration, "robot": robot})


@mcp.tool()
async def gripper(
    action: Literal["open", "close"],
    arm: Annotated[str | None, Field(description="Substring of the arm name; default first arm")] = None,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Open or close a gripper (direction is auto-calibrated per robot)."""
    return await _call("gripper", {"action": action, "arm": arm, "robot": robot})


@mcp.tool()
async def go_to_pose(
    name: Annotated[str, Field(description="A pose from load_robot's 'poses' (e.g. 'home'), or 'zero'")],
    duration: Annotated[float, Field(ge=0.2, le=10)] = 2.0,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Move all joints to a named pose (keyframe) smoothly. For quadrupeds 'home' is standing."""
    return await _call("go_to_pose", {"name": name, "duration": duration, "robot": robot})


@mcp.tool()
async def drive(
    forward_speed: Annotated[float, Field(description="m/s, negative = backwards")],
    turn_rate: Annotated[float, Field(description="rad/s, positive = counter-clockwise")] = 0.0,
    seconds: Annotated[float, Field(ge=0.1, le=20)] = 2.0,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Move the base for some seconds, then stop: wheels (differential drive) or the trot gait for quadrupeds."""
    return await _call("drive", {"forward_speed": forward_speed, "turn_rate": turn_rate, "seconds": seconds, "robot": robot})


@mcp.tool()
async def navigate_to(
    x: Annotated[float, Field(description="Goal x on the floor (m)")],
    y: Annotated[float, Field(description="Goal y on the floor (m)")],
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Walk (quadrupeds: trot gait; G1 humanoid: Unitree walking policy) or drive (wheeled robots) to a point, closed loop."""
    return await _call("navigate_to", {"x": x, "y": y, "robot": robot})


@mcp.tool()
async def go_to_point(x: float, y: float, z: float = 0.0, robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """Do what "go there" means for this robot: walk/drive to (x, y), or hover the gripper above (x, y, z)."""
    return await _call("go_to_point", {"x": x, "y": y, "z": z, "robot": robot})


@mcp.tool()
async def pick_object(
    name: Annotated[str, Field(description="Object name, as in get_state()['objects']")],
    arm: Annotated[str | None, Field(description="Substring of the arm name; default the nearest free arm")] = None,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Grasp an object from above and lift it (verified: fails if the object does not rise)."""
    return await _call("pick_object", {"name": name, "arm": arm, "robot": robot})


@mcp.tool()
async def place_object(
    x: float = 0.0, y: float = 0.0,
    on: Annotated[str | None, Field(description="Stack on this object instead of placing at (x, y) on the floor")] = None,
    arm: Annotated[str | None, Field(description="Arm holding the object; default the nearest one")] = None,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> dict:
    """Put the held object down at (x, y), or on top of another object."""
    return await _call("place_object", {"x": x, "y": y, "on": on, "arm": arm, "robot": robot})


@mcp.tool()
async def spawn_object(
    shape: Literal["box", "sphere", "cylinder", "capsule"] = "box",
    size: Annotated[float, Field(gt=0, le=1, description="Half-size / radius in metres")] = 0.025,
    x: float = 0.5, y: float = 0.0, z: Annotated[float | None, Field(description="Default: resting on the floor")] = None,
    color: Literal["red", "green", "blue", "yellow", "orange", "white", "black", "gray", "purple"] = "red",
    name: str | None = None,
    fixed: Annotated[bool, Field(description="Static object (e.g. a table/obstacle) instead of a free body")] = False,
    mass: Annotated[float, Field(gt=0, le=50)] = 0.1,
) -> dict:
    """Add an object to the scene. Use fixed=true with a larger box to make a table."""
    return await _call("spawn_object", {"shape": shape, "size": size, "x": x, "y": y, "z": z, "color": color,
                                        "name": name, "fixed": fixed, "mass": mass})


@mcp.tool()
async def snapshot(
    azimuth: Annotated[float, Field(description="Camera azimuth in degrees around the robot")] = 135,
    elevation: Annotated[float, Field(description="Degrees, negative looks down")] = -25,
    distance: Annotated[float | None, Field(description="Metres; default frames the robot")] = None,
) -> Image:
    """Render the scene and return it as an image so you can see what the robot did."""
    png = await _call("snapshot", {"azimuth": azimuth, "elevation": elevation, "distance": distance})
    return Image(data=png, format="png")


@mcp.tool()
async def remove_object(name: Annotated[str, Field(description="Object name, as in get_state()['objects']")]) -> dict:
    """Take an object out of the scene (a hand holding it lets go first)."""
    return await _call("remove_object", {"name": name})


@mcp.tool()
async def move_object(
    name: Annotated[str, Field(description="Object name, as in get_state()['objects']")],
    x: float, y: float,
) -> dict:
    """Rearrange the scene: put an object at (x, y) on whatever is there (a hand holding it lets go).
    This is not a robot action; use pick_object / place_object for that."""
    return await _call("move_object", {"name": name, "x": x, "y": y})


@mcp.tool()
async def spawn_asset(
    kind: Annotated[Literal["mug", "tea_box", "lunch_box", "blue_mug"], Field(description="A 3D-scanned real object")] = "mug",
    x: float = 0.5, y: float = 0.0,
) -> dict:
    """Add a 3D-scanned real object standing on whatever is at (x, y): the floor, a table, a shelf."""
    return await _call("spawn_asset", {"kind": kind, "x": x, "y": y})


@mcp.tool()
async def import_scene(
    source: Annotated[str, Field(description="Absolute path to a phone video (.mp4/.mov), a folder of photos, a .glb from a scanning app (Polycam, Scaniverse) or an .obj mesh of a real place")],
    name: Annotated[str, Field(description="Name of the new world, lowercase (e.g. 'kitchen')")],
    height: Annotated[float, Field(description="Floor-to-ceiling height of the room in metres: gives a video its real size")] = 2.6,
) -> dict:
    """Turn a video or scan of a real place into a world (a digital twin) robots can be loaded into.

    A video is rebuilt in 3D on this Mac (Apple Object Capture): it takes several minutes. Then
    load_robot(robot, world=name) puts any robot in it, and navigation works around its walls and furniture.
    """
    return await _call("import_scene", {"source": source, "name": name, "height": height}, timeout=3600.0)


@mcp.tool()
async def stop() -> dict:
    """Stop whatever the robots are doing right now (a walk, a pick, a whole task or the production line)."""
    return await _call("stop")


@mcp.tool()
async def get_scan(robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """The robot's 2D lidar (30 cm above the floor): nearest obstacle in each direction (front, left, ...) and ranges every 5°."""
    return await _call("get_scan", {"robot": robot})


@mcp.tool()
async def camera_image(
    depth: Annotated[bool, Field(description="Depth image instead of colour (near = bright)")] = False,
    robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None,
) -> Image:
    """What the robot's own camera sees (head camera for humanoids, front camera for other mobile robots)."""
    png = await _call("camera_image", {"depth": depth, "robot": robot})
    return Image(data=png, format="png")


@mcp.tool()
async def wait(seconds: Annotated[float, Field(ge=0, le=30)] = 1.0) -> dict:
    """Let the simulation run (e.g. to watch something settle). Returns the state afterwards."""
    return await _call("wait", {"seconds": seconds})


@mcp.tool()
async def reset(robot: Annotated[str | None, Field(description="Team member to command (e.g. loader, sorter, carrier); default the selected one")] = None) -> dict:
    """Reset the robot to its initial pose (objects stay in the scene)."""
    return await _call("reset", {"robot": robot})


def main() -> None:
    # The studio is NOT started here: MCP clients launch this server on every start-up, and the
    # studio opens a browser tab. It starts on the first robot command instead (see _call).
    print(f"[anybot] MCP server ready; studio ({STUDIO_URL}) starts on the first robot command", file=sys.stderr)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
