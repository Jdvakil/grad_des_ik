"""
Stretch Robot - Waypoint Navigation + Pick and Place
=====================================================
Undergraduate Intro to Robotics Lab

Overview
--------
This controller demonstrates a basic 2D waypoint navigation stack:

  1. A list of (x, y) waypoints is defined in advance.
  2. A heading controller (P-control on bearing error) steers the robot
     toward each waypoint while a velocity controller drives it forward.
  3. Special waypoints trigger arm actions (pick / place).

Key concepts illustrated
------------------------
  * Differential drive kinematics
  * Simple wheel-encoder odometry (dead-reckoning)
  * Proportional (P) controller for heading
  * State-machine task sequencing

Robot joints used
-----------------
  Wheels:         joint_left_wheel, joint_right_wheel  (velocity control)
  Vertical lift:  joint_lift                           (position control)
  Arm extension:  joint_arm_l0 … joint_arm_l3          (position control)
  Wrist:          joint_wrist_yaw                      (position control)
  Gripper:        joint_gripper_finger_left/right       (position control)
  Head:           joint_head_pan, joint_head_tilt       (position control)
"""

import math
from controller import Robot

# ═══════════════════════════════════════════════════════════════════════════════
# TUNEABLE PARAMETERS  (students: try changing these!)
# ═══════════════════════════════════════════════════════════════════════════════
DRIVE_SPEED     = 1.0   # rad/s – cruise wheel speed
TURN_SPEED      = 0.7   # rad/s – in-place turn speed
HEADING_KP      = 2.0   # proportional gain for heading correction
WAYPOINT_TOL    = 0.12  # m     – distance to consider a waypoint reached
HEADING_TOL     = 0.10  # rad   – angular error to consider aligned

# Robot geometry (Stretch RE1/RE2)
WHEEL_RADIUS    = 0.051  # m
WHEEL_BASELINE  = 0.315  # m  (distance between drive wheels)

# Arm / lift parameters
LIFT_TRAVEL     = 0.52   # m  – lift height while driving (arm out of the way)
LIFT_GRASP      = 0.38   # m  – lift height to lower onto box
LIFT_CARRY      = 0.60   # m  – lift height while carrying
ARM_EXT         = 0.10   # m  per segment (4 segments → 0.40 m total reach)
GRIPPER_OPEN    =  0.35  # rad
GRIPPER_CLOSE   = -0.20  # rad

# ═══════════════════════════════════════════════════════════════════════════════
# WAYPOINTS  (x, y, action)
#
# 'action' is executed when the waypoint is reached:
#   None      – just drive through
#   'pick'    – extend arm, lower, close gripper, raise
#   'place'   – lower arm, open gripper, raise
# ═══════════════════════════════════════════════════════════════════════════════
WAYPOINTS = [
    (0.80, 0.00, 'pick'),    # drive to the red box and pick it up
    (0.00, 0.80, 'place'),   # carry to the green drop zone and place it
]

# ═══════════════════════════════════════════════════════════════════════════════
# ROBOT SETUP
# ═══════════════════════════════════════════════════════════════════════════════
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

def get_motor(name):
    m = robot.getDevice(name)
    if m is None:
        raise RuntimeError(f"Motor not found: {name}")
    return m

def get_sensor(name):
    s = robot.getDevice(name)
    if s is None:
        raise RuntimeError(f"Sensor not found: {name}")
    s.enable(timestep)
    return s

# Motors
left_wheel  = get_motor("joint_left_wheel")
right_wheel = get_motor("joint_right_wheel")
lift        = get_motor("joint_lift")
arms        = [get_motor(f"joint_arm_l{i}") for i in range(4)]
wrist_yaw   = get_motor("joint_wrist_yaw")
gripper_l   = get_motor("joint_gripper_finger_left")
gripper_r   = get_motor("joint_gripper_finger_right")
head_pan    = get_motor("joint_head_pan")
head_tilt   = get_motor("joint_head_tilt")

# Wheels use velocity control
left_wheel.setPosition(float('inf'))
right_wheel.setPosition(float('inf'))
left_wheel.setVelocity(0)
right_wheel.setVelocity(0)

# Position sensors
lift_sensor = get_sensor("joint_lift_sensor")
arm_sensors = [get_sensor(f"joint_arm_l{i}_sensor") for i in range(4)]
gl_sensor   = get_sensor("joint_gripper_finger_left_sensor")
gr_sensor   = get_sensor("joint_gripper_finger_right_sensor")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def wrap_angle(a):
    """Wrap angle to [-π, π]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

def set_arm(ext_per_seg):
    """Set all 4 arm segments to the same extension (meters)."""
    ext = max(0.0, min(0.13, ext_per_seg))
    for m in arms:
        m.setPosition(ext)

def arm_reached(target, tol=0.012):
    return all(abs(s.getValue() - target) < tol for s in arm_sensors)

def lift_reached(target, tol=0.020):
    return abs(lift_sensor.getValue() - target) < tol

def wait_steps(n):
    for _ in range(n):
        robot.step(timestep)

# ═══════════════════════════════════════════════════════════════════════════════
# INITIAL POSE
# ═══════════════════════════════════════════════════════════════════════════════
lift.setPosition(LIFT_TRAVEL)
set_arm(0.0)
wrist_yaw.setPosition(0.0)
gripper_l.setPosition(GRIPPER_OPEN)
gripper_r.setPosition(-GRIPPER_OPEN)
head_pan.setPosition(0.0)
head_tilt.setPosition(-0.4)

# Let joints settle before moving
print("[stretch] Settling joints …")
wait_steps(int(2500 / timestep))

# ═══════════════════════════════════════════════════════════════════════════════
# ARM ACTION SEQUENCES
# ═══════════════════════════════════════════════════════════════════════════════
def do_pick():
    """Lower onto box, close gripper, lift."""
    print("[stretch] ACTION: pick")
    # 1. Extend arm
    set_arm(ARM_EXT)
    timeout = int(3000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if arm_reached(ARM_EXT): break

    # 2. Lower lift to grasp height
    lift.setPosition(LIFT_GRASP)
    timeout = int(4000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if lift_reached(LIFT_GRASP): break

    # 3. Close gripper
    gripper_l.setPosition(GRIPPER_CLOSE)
    gripper_r.setPosition(-GRIPPER_CLOSE)
    wait_steps(int(1200 / timestep))

    # 4. Raise lift (pick up object)
    lift.setPosition(LIFT_CARRY)
    timeout = int(4000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if lift_reached(LIFT_CARRY): break

    # 5. Retract arm before driving
    set_arm(0.0)
    timeout = int(3000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if arm_reached(0.0): break

    print("[stretch] Pick complete")


def do_place():
    """Extend arm, lower, open gripper, raise, retract."""
    print("[stretch] ACTION: place")
    # 1. Extend arm
    set_arm(ARM_EXT)
    timeout = int(3000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if arm_reached(ARM_EXT): break

    # 2. Lower lift to place height
    lift.setPosition(LIFT_GRASP + 0.07)
    timeout = int(4000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if lift_reached(LIFT_GRASP + 0.07): break

    # 3. Open gripper
    gripper_l.setPosition(GRIPPER_OPEN)
    gripper_r.setPosition(-GRIPPER_OPEN)
    wait_steps(int(1000 / timestep))

    # 4. Raise lift
    lift.setPosition(LIFT_TRAVEL)
    timeout = int(4000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if lift_reached(LIFT_TRAVEL): break

    # 5. Retract arm
    set_arm(0.0)
    timeout = int(3000 / timestep)
    for _ in range(timeout):
        robot.step(timestep)
        if arm_reached(0.0): break

    print("[stretch] Place complete")


# ═══════════════════════════════════════════════════════════════════════════════
# ODOMETRY STATE
# ═══════════════════════════════════════════════════════════════════════════════
x       = 0.0   # m
y       = 0.0   # m
theta   = 0.0   # rad   (heading, 0 = +X direction)

# Track previous commanded velocities for odometry integration
prev_vl = 0.0
prev_vr = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# WAYPOINT NAVIGATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("[stretch] Starting waypoint navigation")

for wp_index, (wx, wy, action) in enumerate(WAYPOINTS):
    print(f"[stretch] Waypoint {wp_index + 1}/{len(WAYPOINTS)}: target ({wx:.2f}, {wy:.2f})  action={action}")

    # ── Phase 1: Turn toward waypoint ──────────────────────────────────────
    while robot.step(timestep) != -1:
        # Odometry update
        dt = timestep / 1000.0
        vl = left_wheel.getVelocity()
        vr = right_wheel.getVelocity()
        v   = WHEEL_RADIUS * (vr + vl) / 2.0
        w   = WHEEL_RADIUS * (vr - vl) / WHEEL_BASELINE
        theta = wrap_angle(theta + w * dt)
        x    += v * math.cos(theta) * dt
        y    += v * math.sin(theta) * dt

        dx     = wx - x
        dy     = wy - y
        dist   = math.sqrt(dx*dx + dy*dy)
        target_heading = math.atan2(dy, dx)
        heading_err    = wrap_angle(target_heading - theta)

        if dist < WAYPOINT_TOL:
            left_wheel.setVelocity(0)
            right_wheel.setVelocity(0)
            print(f"[stretch]   Arrived at ({x:.2f},{y:.2f}), heading err={math.degrees(heading_err):.1f}°")
            break

        if abs(heading_err) > HEADING_TOL:
            # ── Turn in place to face the waypoint ──────────────────────
            turn = TURN_SPEED if heading_err > 0 else -TURN_SPEED
            left_wheel.setVelocity(-turn)
            right_wheel.setVelocity(turn)
        else:
            # ── Drive forward with proportional heading correction ───────
            # P-controller: steer = Kp * heading_error
            # v_left  = v_base - steer
            # v_right = v_base + steer
            steer = max(-DRIVE_SPEED, min(DRIVE_SPEED, HEADING_KP * heading_err))
            left_wheel.setVelocity(DRIVE_SPEED - steer)
            right_wheel.setVelocity(DRIVE_SPEED + steer)

    # ── Phase 2: Execute action at waypoint ────────────────────────────────
    left_wheel.setVelocity(0)
    right_wheel.setVelocity(0)

    if action == 'pick':
        do_pick()
    elif action == 'place':
        do_place()

# ── All waypoints done ─────────────────────────────────────────────────────
print("[stretch] All waypoints complete!")
lift.setPosition(0.3)
set_arm(0.0)
head_tilt.setPosition(0.0)

while robot.step(timestep) != -1:
    pass  # idle
