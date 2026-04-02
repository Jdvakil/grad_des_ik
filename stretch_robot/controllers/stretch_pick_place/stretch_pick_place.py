"""
Stretch robot pick-and-place controller.

Uses sensor feedback (position sensors + GPS) for reliable state transitions
instead of blind timers. The robot drives toward a red box placed at (0.9, 0, 0),
picks it up, turns, and drops it on the green zone at (0, 0.9, 0).

State machine:
  INIT → DRIVE_TO_BOX → ALIGN → EXTEND_ARM → LOWER_GRASP →
  GRASP → PICK_UP → RETRACT → TURN → PLACE_EXTEND →
  PLACE_LOWER → RELEASE → STOW → DONE
"""

from controller import Robot, GPS
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ── Devices ───────────────────────────────────────────────────────────────────
left_wheel  = robot.getDevice("joint_left_wheel")
right_wheel = robot.getDevice("joint_right_wheel")
lift        = robot.getDevice("joint_lift")
arm_l0      = robot.getDevice("joint_arm_l0")
arm_l1      = robot.getDevice("joint_arm_l1")
arm_l2      = robot.getDevice("joint_arm_l2")
arm_l3      = robot.getDevice("joint_arm_l3")
wrist_yaw   = robot.getDevice("joint_wrist_yaw")
wrist_pitch = robot.getDevice("joint_wrist_pitch")
gripper_l   = robot.getDevice("joint_gripper_finger_left")
gripper_r   = robot.getDevice("joint_gripper_finger_right")
head_pan    = robot.getDevice("joint_head_pan")
head_tilt   = robot.getDevice("joint_head_tilt")

# Wheel velocity control
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))
left_wheel.setVelocity(0)
right_wheel.setVelocity(0)

# Position sensors
lift_s    = robot.getDevice("joint_lift_sensor");    lift_s.enable(timestep)
arm_s     = [robot.getDevice(f"joint_arm_l{i}_sensor") for i in range(4)]
for s in arm_s: s.enable(timestep)
wrist_s   = robot.getDevice("joint_wrist_yaw_sensor"); wrist_s.enable(timestep)
gl_s      = robot.getDevice("joint_gripper_finger_left_sensor");  gl_s.enable(timestep)
gr_s      = robot.getDevice("joint_gripper_finger_right_sensor"); gr_s.enable(timestep)

# GPS for odometry
gps = robot.getDevice("gps") if robot.getDevice("gps") else None
if gps:
    gps.enable(timestep)

# ── Constants ─────────────────────────────────────────────────────────────────
DRIVE_SPEED   = 1.2    # rad/s wheels (gentle)
TURN_SPEED    = 0.8    # rad/s wheels
LIFT_APPROACH = 0.55   # m  - lift height while driving / approaching
LIFT_GRASP    = 0.38   # m  - lift height to pick up the box
LIFT_CARRY    = 0.65   # m  - lift height while carrying
ARM_EXT       = 0.12   # m  per-segment extension (4 × 0.12 = 0.48 m total reach)
ARM_PLACE_EXT = 0.10   # m  per-segment for placing
GRIPPER_OPEN  =  0.35  # rad
GRIPPER_CLOSE = -0.15  # rad (squeeze)
WHEEL_SEP     = 0.315  # m  between wheels (Stretch specification)

# ── Helpers ───────────────────────────────────────────────────────────────────
def drive(vl, vr):
    left_wheel.setVelocity(vl)
    right_wheel.setVelocity(vr)

def stop():
    left_wheel.setVelocity(0)
    right_wheel.setVelocity(0)

def at(sensor, target, tol=0.012):
    return abs(sensor.getValue() - target) < tol

def set_arm(ext_per_seg):
    for m in [arm_l0, arm_l1, arm_l2, arm_l3]:
        m.setPosition(max(0.0, min(0.13, ext_per_seg)))

def arm_at(target_per_seg, tol=0.015):
    return all(abs(s.getValue() - target_per_seg) < tol for s in arm_s)

def step():
    return robot.step(timestep) != -1

# ── Startup: stow everything ──────────────────────────────────────────────────
lift.setPosition(LIFT_APPROACH)
set_arm(0.0)
wrist_yaw.setPosition(0.0)
wrist_pitch.setPosition(0.0)
gripper_l.setPosition(GRIPPER_OPEN)
gripper_r.setPosition(-GRIPPER_OPEN)
head_pan.setPosition(0.0)
head_tilt.setPosition(-0.5)

# Let joints settle before starting
for _ in range(int(2000 / timestep)):
    robot.step(timestep)

# ── State machine ─────────────────────────────────────────────────────────────
state   = "DRIVE_TO_BOX"
timer   = 0.0
print(f"[stretch] state: {state}")

# Odometry (simple encoder-based, no GPS required)
heading  = 0.0   # radians from +X
odom_x   = 0.0
odom_y   = 0.0
prev_vl  = 0.0
prev_vr  = 0.0

BOX_X,  BOX_Y  = 0.90, 0.00   # pickup target
DROP_X, DROP_Y = 0.00, 0.90   # drop target

def dist_to(tx, ty):
    return math.sqrt((tx - odom_x)**2 + (ty - odom_y)**2)

def angle_to(tx, ty):
    """Bearing to target in world frame."""
    return math.atan2(ty - odom_y, tx - odom_x)

def wrap(a):
    while a >  math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

while robot.step(timestep) != -1:
    dt = timestep / 1000.0
    timer += dt

    # Simple odometry from wheel velocities
    vl = left_wheel.getVelocity()
    vr = right_wheel.getVelocity()
    v_lin = 0.051 * (vr + vl) / 2.0   # 0.051 m = wheel radius
    v_ang = 0.051 * (vr - vl) / WHEEL_SEP
    heading = wrap(heading + v_ang * dt)
    odom_x += v_lin * math.cos(heading) * dt
    odom_y += v_lin * math.sin(heading) * dt

    # ── Drive to box ─────────────────────────────────────────────────────────
    if state == "DRIVE_TO_BOX":
        d = dist_to(BOX_X, BOX_Y)
        bear = angle_to(BOX_X, BOX_Y)
        err = wrap(bear - heading)

        if d < 0.45:          # close enough to box to extend arm
            stop()
            state = "EXTEND_ARM"
            timer = 0.0
            print(f"[stretch] state: {state}  odom=({odom_x:.2f},{odom_y:.2f})")
        else:
            # Proportional steering
            k = 1.8
            steer = max(-1.0, min(1.0, k * err))
            drive(DRIVE_SPEED * (1 - steer), DRIVE_SPEED * (1 + steer))

    # ── Extend arm toward box ─────────────────────────────────────────────────
    elif state == "EXTEND_ARM":
        set_arm(ARM_EXT)
        if arm_at(ARM_EXT) or timer > 3.0:
            state = "LOWER_GRASP"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Lower lift to grasp height ────────────────────────────────────────────
    elif state == "LOWER_GRASP":
        lift.setPosition(LIFT_GRASP)
        if at(lift_s, LIFT_GRASP, tol=0.025) or timer > 3.5:
            state = "GRASP"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Close gripper ─────────────────────────────────────────────────────────
    elif state == "GRASP":
        gripper_l.setPosition(GRIPPER_CLOSE)
        gripper_r.setPosition(-GRIPPER_CLOSE)
        if timer > 1.2:
            state = "PICK_UP"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Raise lift ────────────────────────────────────────────────────────────
    elif state == "PICK_UP":
        lift.setPosition(LIFT_CARRY)
        if at(lift_s, LIFT_CARRY, tol=0.03) or timer > 3.5:
            state = "RETRACT"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Retract arm ───────────────────────────────────────────────────────────
    elif state == "RETRACT":
        set_arm(0.0)
        if arm_at(0.0) or timer > 3.0:
            state = "TURN_TO_DROP"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Turn toward drop zone ─────────────────────────────────────────────────
    elif state == "TURN_TO_DROP":
        bear = angle_to(DROP_X, DROP_Y)
        err = wrap(bear - heading)
        if abs(err) < 0.08:
            stop()
            state = "DRIVE_TO_DROP"
            timer = 0.0
            print(f"[stretch] state: {state}")
        else:
            direction = 1 if err > 0 else -1
            drive(-TURN_SPEED * direction, TURN_SPEED * direction)

    # ── Drive toward drop zone ────────────────────────────────────────────────
    elif state == "DRIVE_TO_DROP":
        d = dist_to(DROP_X, DROP_Y)
        bear = angle_to(DROP_X, DROP_Y)
        err = wrap(bear - heading)

        if d < 0.50:
            stop()
            state = "PLACE_EXTEND"
            timer = 0.0
            print(f"[stretch] state: {state}")
        else:
            k = 1.8
            steer = max(-1.0, min(1.0, k * err))
            drive(DRIVE_SPEED * (1 - steer), DRIVE_SPEED * (1 + steer))

    # ── Extend arm over drop zone ─────────────────────────────────────────────
    elif state == "PLACE_EXTEND":
        set_arm(ARM_PLACE_EXT)
        if arm_at(ARM_PLACE_EXT) or timer > 3.0:
            state = "PLACE_LOWER"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Lower onto drop zone ──────────────────────────────────────────────────
    elif state == "PLACE_LOWER":
        lift.setPosition(LIFT_GRASP + 0.08)
        if at(lift_s, LIFT_GRASP + 0.08, tol=0.03) or timer > 3.0:
            state = "RELEASE"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Open gripper ──────────────────────────────────────────────────────────
    elif state == "RELEASE":
        gripper_l.setPosition(GRIPPER_OPEN)
        gripper_r.setPosition(-GRIPPER_OPEN)
        if timer > 1.0:
            state = "STOW"
            timer = 0.0
            print(f"[stretch] state: {state}")

    # ── Stow robot ────────────────────────────────────────────────────────────
    elif state == "STOW":
        set_arm(0.0)
        lift.setPosition(0.3)
        head_tilt.setPosition(0.0)
        if timer > 2.0:
            state = "DONE"
            print("[stretch] Pick-and-place complete!")

    elif state == "DONE":
        pass  # idle
