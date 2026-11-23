"""anybot: drop in any robot, get a natural-language-ready simulation."""

from anybot.introspect import RobotProfile, introspect
from anybot.loader import LoadedRobot, available_descriptions, load_robot


def profile(source: str, *, floating: bool | None = None) -> RobotProfile:
    """Load a robot (name or path) and return its profile."""
    robot = load_robot(source, floating=floating)
    return introspect(robot.model, robot.name, robot.injected_actuators)


__all__ = ["LoadedRobot", "RobotProfile", "available_descriptions", "introspect", "load_robot", "profile"]
