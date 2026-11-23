"""anybot command line.

  anybot list                     bundled robots
  anybot profile <robot>...       what anybot understood about a robot
  anybot view <robot> [--pin]     open the robot in the MuJoCo viewer, driven by anybot (PD, poses)
  anybot studio [--robot panda]   web UI with the live robot (http://127.0.0.1:8765)
  anybot stop                     stop a running studio (also one started by Claude Desktop)
  anybot import-scene <video|photos|scan.glb> --name kitchen   a real place as a world (video → world)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from anybot import available_descriptions, profile


def _view(args: argparse.Namespace) -> None:
    # On macOS the interactive viewer must own the main thread, which only `mjpython` allows.
    if sys.platform == "darwin" and not os.environ.get("ANYBOT_MJPYTHON"):
        mjpython = Path(sys.executable).with_name("mjpython")
        os.environ["ANYBOT_MJPYTHON"] = "1"
        os.execv(str(mjpython), [str(mjpython), "-m", "anybot.cli", *sys.argv[1:]])

    from anybot.sim import Sim

    sim = Sim(args.robot, viewer=True, pin_base=args.pin, floating=args.floating or None)
    print(sim.profile.summary())
    print(f"skills: {', '.join(sim.skills())} | poses: {', '.join(sim.keyframes()) or '-'}")
    print("Close the viewer window (or Ctrl-C) to quit.")
    try:
        while sim._viewer is not None and sim._viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


def _stop() -> None:
    """Stop every running anybot studio: SIGTERM, then SIGKILL for anything still alive after 3 s."""
    import signal
    import subprocess

    def pids() -> list[int]:
        out = subprocess.run(["pgrep", "-f", "anybot.studio"], capture_output=True, text=True).stdout
        return [int(p) for p in out.split() if int(p) != os.getpid()]

    found = pids()
    if not found:
        print("No anybot studio is running.")
        return
    for pid in found:
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not pids():
            break
        time.sleep(0.1)
    for pid in pids():
        os.kill(pid, signal.SIGKILL)
    print(f"Stopped anybot studio (pid {', '.join(map(str, found))}).")


def main() -> None:
    parser = argparse.ArgumentParser(prog="anybot")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="Classify and describe a robot")
    p.add_argument("robots", nargs="+", help="robot_descriptions name (e.g. panda, go2) or .xml/.urdf path")
    p.add_argument("--json", action="store_true", help="print the full profile as JSON")
    p.add_argument("--floating", action="store_true", help="add a free joint to the root (URDF legged/mobile robots)")
    sub.add_parser("list", help="List bundled MuJoCo robots")
    v = sub.add_parser("view", help="Open a robot in the viewer, held by anybot's controllers")
    v.add_argument("robot")
    v.add_argument("--pin", action="store_true", help="pin a legged robot's base in place")
    v.add_argument("--floating", action="store_true", help="URDF only: add a free joint to the root")
    st = sub.add_parser("studio", help="Web UI with the live simulation")
    st.add_argument("--robot")
    st.add_argument("--port", type=int)
    st.add_argument("--no-browser", action="store_true")
    st.add_argument("--world", default="warehouse", help="warehouse, lab, apartment, bench, empty or an imported scene")
    sub.add_parser("stop", help="Stop a running studio")
    imp = sub.add_parser("import-scene", help="Make a world from a phone video, photos or a scan of a real place")
    imp.add_argument("source", help="video (.mp4/.mov), folder of photos, .glb scan or .obj mesh")
    imp.add_argument("--name", required=True, help="name of the new world (lowercase)")
    imp.add_argument("--height", type=float, default=2.6, help="floor-to-ceiling height in metres (gives a video its size)")
    imp.add_argument("--scale", type=float, help="instead of --height: multiply the rebuilt mesh by this")
    imp.add_argument("--detail", default="reduced", choices=["preview", "reduced", "medium", "full"])
    args = parser.parse_args()

    if args.cmd == "stop":
        _stop()
        return
    if args.cmd == "import-scene":
        from anybot.capture import CaptureError, import_scene

        try:
            out = import_scene(args.source, args.name, height=args.height, scale=args.scale, detail=args.detail)
        except CaptureError as e:
            sys.exit(f"anybot: {e}")
        print(f"Try it: anybot studio --world {out['world']} --robot spot")
        return
    if args.cmd == "studio":
        from anybot.studio import main as studio_main

        argv = (["--robot", args.robot] if args.robot else []) + (["--port", str(args.port)] if args.port else [])
        argv += ["--world", args.world]
        studio_main(argv + (["--no-browser"] if args.no_browser else []))
    elif args.cmd == "list":
        print("\n".join(available_descriptions()))
    elif args.cmd == "view":
        _view(args)
    else:
        for robot in args.robots:
            prof = profile(robot, floating=args.floating or None)
            print(json.dumps(prof.to_dict(), indent=2) if args.json else prof.summary())


if __name__ == "__main__":
    main()
