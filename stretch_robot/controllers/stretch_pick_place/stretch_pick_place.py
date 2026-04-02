"""
Stretch robot pick-and-place controller.

Sequence:
  1. Drive forward toward the box
  2. Raise the lift to approach height
  3. Extend the arm over the box
  4. Lower the lift to grasp height
  5. Close the gripper
  6. Raise the lift (pick)
  7. Retract the arm
  8. Turn 90 degrees
  9. Extend arm to drop position
  10. Open the gripper (place)
  11. Retract arm and celebrate
"""

from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# ── Motors ────────────────────────────────────────────────────────────────────
left_wheel  = robot.getDevice("joint_left_wheel")
right_wheel = robot.getDevice("joint_right_wheel")
lift        = robot.getDevice("joint_lift")
arm_l0      = robot.getDevice("joint_arm_l0")
arm_l1      = robot.getDevice("joint_arm_l1")
arm_l2      = robot.getDevice("joint_arm_l2")
arm_l3      = robot.getDevice("joint_arm_l3")
wrist_yaw   = robot.getDevice("joint_wrist_yaw")
gripper_l   = robot.getDevice("joint_gripper_finger_left")
gripper_r   = robot.getDevice("joint_gripper_finger_right")
head_pan    = robot.getDevice("joint_head_pan")
head_tilt   = robot.getDevice("joint_head_tilt")

# Wheels use velocity control
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))
left_wheel.setVelocity(0)
right_wheel.setVelocity(0)

# Other joints use position control
lift.setPosition(0.2)
arm_l0.setPosition(0.0)
arm_l1.setPosition(0.0)
arm_l2.setPosition(0.0)
arm_l3.setPosition(0.0)
wrist_yaw.setPosition(0.0)
gripper_l.setPosition(0.4)   # open
gripper_r.setPosition(-0.4)  # open
head_pan.setPosition(0.0)
head_tilt.setPosition(-0.3)

# ── Position sensors ──────────────────────────────────────────────────────────
lift_sensor  = robot.getDevice("joint_lift_sensor")
arm_sensors  = [robot.getDevice(f"joint_arm_l{i}_sensor") for i in range(4)]
wrist_sensor = robot.getDevice("joint_wrist_yaw_sensor")
gripper_l_s  = robot.getDevice("joint_gripper_finger_left_sensor")
gripper_r_s  = robot.getDevice("joint_gripper_finger_right_sensor")

for s in [lift_sensor, wrist_sensor, gripper_l_s, gripper_r_s] + arm_sensors:
    s.enable(timestep)

# ── Helpers ───────────────────────────────────────────────────────────────────
def drive(speed_l, speed_r):
    left_wheel.setVelocity(speed_l)
    right_wheel.setVelocity(speed_r)

def wait(seconds):
    steps = int(seconds * 1000 / timestep)
    for _ in range(steps):
        robot.step(timestep)

def near(sensor, target, tol=0.015):
    return abs(sensor.getValue() - target) < tol

def wait_for(sensor, target, tol=0.015, timeout=5.0):
    steps = int(timeout * 1000 / timestep)
    for _ in range(steps):
        robot.step(timestep)
        if near(sensor, target, tol):
            return True
    return False

def set_arm_extension(ext):
    """Distribute extension equally across the 4 arm segments (max ~0.13 each)."""
    per_joint = min(ext / 4.0, 0.13)
    for m in [arm_l0, arm_l1, arm_l2, arm_l3]:
        m.setPosition(per_joint)

def arm_extension_reached(target_ext, tol=0.02):
    per_joint = min(target_ext / 4.0, 0.13)
    return all(abs(s.getValue() - per_joint) < tol for s in arm_sensors)

# ── State machine ─────────────────────────────────────────────────────────────
APPROACH_LIFT   = 0.6   # lift height while driving (m)
GRASP_LIFT      = 0.35  # lift height to lower onto box (m)
CARRY_LIFT      = 0.7   # lift height while carrying
ARM_REACH       = 0.3   # arm extension to reach box (m total across 4 joints)
DRIVE_SPEED     = 1.5   # wheel speed (rad/s)
TURN_SPEED      = 1.0
DRIVE_FWD_TIME  = 2.5   # seconds of forward driving
TURN_TIME       = 2.3   # seconds for ~90 degree turn

state = "INIT"
timer = 0.0

print("[stretch] Starting pick-and-place sequence")

while robot.step(timestep) != -1:
    dt = timestep / 1000.0
    timer += dt

    # ── INIT: stow and wait for joints to settle ─────────────────────────────
    if state == "INIT":
        lift.setPosition(APPROACH_LIFT)
        set_arm_extension(0.0)
        gripper_l.setPosition(0.4)
        gripper_r.setPosition(-0.4)
        head_tilt.setPosition(-0.5)
        if timer > 2.0:
            state = "DRIVE_FWD"
            timer = 0.0
            print("[stretch] Driving toward box")

    # ── DRIVE_FWD: move toward the box ───────────────────────────────────────
    elif state == "DRIVE_FWD":
        drive(DRIVE_SPEED, DRIVE_SPEED)
        if timer > DRIVE_FWD_TIME:
            drive(0, 0)
            state = "EXTEND_ARM"
            timer = 0.0
            print("[stretch] Extending arm over box")

    # ── EXTEND_ARM: push arm out toward box ──────────────────────────────────
    elif state == "EXTEND_ARM":
        set_arm_extension(ARM_REACH)
        if timer > 1.5 and arm_extension_reached(ARM_REACH):
            state = "LOWER_LIFT"
            timer = 0.0
            print("[stretch] Lowering onto box")

    # ── LOWER_LIFT: descend onto box ─────────────────────────────────────────
    elif state == "LOWER_LIFT":
        lift.setPosition(GRASP_LIFT)
        if timer > 2.0 and near(lift_sensor, GRASP_LIFT, tol=0.03):
            state = "GRASP"
            timer = 0.0
            print("[stretch] Closing gripper")

    # ── GRASP: close gripper ─────────────────────────────────────────────────
    elif state == "GRASP":
        gripper_l.setPosition(-0.2)
        gripper_r.setPosition(0.2)
        if timer > 1.2:
            state = "LIFT_UP"
            timer = 0.0
            print("[stretch] Lifting box")

    # ── LIFT_UP: raise object ────────────────────────────────────────────────
    elif state == "LIFT_UP":
        lift.setPosition(CARRY_LIFT)
        if timer > 2.0 and near(lift_sensor, CARRY_LIFT, tol=0.03):
            state = "RETRACT_ARM"
            timer = 0.0
            print("[stretch] Retracting arm")

    # ── RETRACT_ARM: pull arm back in ────────────────────────────────────────
    elif state == "RETRACT_ARM":
        set_arm_extension(0.0)
        if timer > 1.5 and arm_extension_reached(0.0):
            state = "TURN"
            timer = 0.0
            print("[stretch] Turning to drop zone")

    # ── TURN: rotate ~90 degrees ─────────────────────────────────────────────
    elif state == "TURN":
        drive(-TURN_SPEED, TURN_SPEED)
        if timer > TURN_TIME:
            drive(0, 0)
            state = "EXTEND_PLACE"
            timer = 0.0
            print("[stretch] Extending to drop position")

    # ── EXTEND_PLACE: reach out to drop ──────────────────────────────────────
    elif state == "EXTEND_PLACE":
        set_arm_extension(ARM_REACH * 0.8)
        if timer > 1.5 and arm_extension_reached(ARM_REACH * 0.8):
            state = "LOWER_PLACE"
            timer = 0.0
            print("[stretch] Lowering to drop")

    # ── LOWER_PLACE: descend to drop height ──────────────────────────────────
    elif state == "LOWER_PLACE":
        lift.setPosition(GRASP_LIFT + 0.05)
        if timer > 1.5:
            state = "RELEASE"
            timer = 0.0
            print("[stretch] Releasing box")

    # ── RELEASE: open gripper ────────────────────────────────────────────────
    elif state == "RELEASE":
        gripper_l.setPosition(0.4)
        gripper_r.setPosition(-0.4)
        if timer > 1.0:
            state = "DONE"
            timer = 0.0
            print("[stretch] Pick-and-place complete!")

    # ── DONE: stow robot ─────────────────────────────────────────────────────
    elif state == "DONE":
        set_arm_extension(0.0)
        lift.setPosition(0.3)
        head_tilt.setPosition(0.0)
        # Just idle
