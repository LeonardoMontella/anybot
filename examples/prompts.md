# Demo prompts

Works with any MCP client (Claude Desktop, Claude Code, ...). anybot figures out the robot; the prompt
never mentions joint names or controllers.

## Arm: pick and place (panda, xarm7, so_arm100, ur5e...)
> Load the Panda. Put a red cube at x=0.5 y=0 and a blue one at x=0.4 y=0.25.
> Pick up the red cube and stack it on the blue one. Take a snapshot to check.

## Quadruped (go2, spot, anymal_c)
> Load the Go2, tell me what kind of robot it is and how it is actuated, then make it stand
> in its home pose and show me a snapshot from the side.

## Humanoid (g1)
> Load the G1 and pick up the box from the floor, then put it on table B.

## Mobile manipulation in a scanned apartment (tidybot, spot, g1)
> Load the Tidybot in the apartment, put a mug on the kitchen counter and bring it to spot_3.

## A team (amr_fleet, amr_line, mixed_line)
> Load the amr_fleet team and run the relay: pickers move every crate to table B while carriers swap.

## Zero-config import
> Load /absolute/path/to/my_robot.urdf and describe it. What can it do?
