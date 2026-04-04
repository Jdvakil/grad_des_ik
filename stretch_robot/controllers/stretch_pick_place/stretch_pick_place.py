"""
Stretch Robot - Waypoint Navigation + Pick and Place
=====================================================
Undergraduate Intro to Robotics Lab

Overview
--------
This controller demonstrates a basic 2D waypoint navigation stack:

  1. A list of (x, y) waypoints is defined in advance.
  2. A unified go-to-point law maps range and bearing error to forward speed
     and turn rate (then to wheel speeds), so the base can arc in and stop
     instead of alternating pure spin vs. drive near the goal.
  3. Special waypoints trigger arm actions (pick / place).

Key concepts illustrated
------------------------
  * Differential drive kinematics
  * Ground-truth base pose from Webots GPS + InertialUnit (wheel integration
    is omitted: imported models often have joint-axis sign quirks that break θ).
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
WAYPOINT_TOL    = 0.11  # m – stop when range to waypoint below this

# Unified go-to-point: body-frame v (m/s) + ω (rad/s) → wheel ω (rad/s).
# A separate “turn then drive” mode often spins forever near the goal (bearing noise,
# or distance never shrinks while |heading_err| stays above a deadband).
K_V             = 0.50   # forward speed gain (1/s), capped by V_LIN_MAX
V_LIN_MAX       = 0.052  # m/s (~ 1 rad/s wheel * wheel radius)
K_W             = 2.0    # yaw rate vs heading error (rad/s per rad)
W_BODY_MAX      = 0.50   # |ω| cap while driving (arc toward goal)
W_SPIN_MAX      = 0.90   # |ω| cap when rotating in place (large misalignment)
ALIGN_THRESHOLD = 1.05   # rad (~60°): above this, v=0 and spin only; below, drive+steer
APPROACH_RAMP   = 0.36   # m – scale v down as dist → 0 to limit overshoot
WHEEL_OMEGA_LIM = 1.15   # rad/s – clamp each wheel command

# Robot geometry (Stretch RE1/RE2)
WHEEL_RADIUS    = 0.051  # m
WHEEL_BASELINE  = 0.315  # m  (distance between drive wheels)

# Arm / lift parameters
LIFT_TRAVEL     = 0.52   # m  – lift height while driving (arm out of the way)
LIFT_GRASP      = 0.36  # m  – lift height to lower onto box (tune vs. box top in sim)
LIFT_CARRY      = 0.60   # m  – lift height while carrying
ARM_EXT         = 0.125  # m  per segment (4 segments; proto max 0.13 m each)
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
    # Coordinates match stretch_world.wbt: box center (0.9, 0), drop disc (0, 0.9)
    (0.90, 0.00, 'pick'),
    (0.00, 0.90, 'place'),
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

# World-frame pose (see Stretch.proto: base_gps, base_imu)
base_gps = get_sensor("base_gps")
base_imu = get_sensor("base_imu")

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
# NAVIGATION STATE  (world frame: x,y from GPS; yaw from IMU)
# ═══════════════════════════════════════════════════════════════════════════════
x       = 0.0   # m
y       = 0.0   # m
theta   = 0.0   # rad, yaw about world +Z (0 = facing +X)

# ═══════════════════════════════════════════════════════════════════════════════
# WAYPOINT NAVIGATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
print("[stretch] Starting waypoint navigation")

for wp_index, (wx, wy, action) in enumerate(WAYPOINTS):
    print(f"[stretch] Waypoint {wp_index + 1}/{len(WAYPOINTS)}: target ({wx:.2f}, {wy:.2f})  action={action}")

    # ── Phase 1: Turn toward waypoint ──────────────────────────────────────
    while robot.step(timestep) != -1:
        pos = base_gps.getValues()
        x, y = pos[0], pos[1]
        rpy = base_imu.getRollPitchYaw()
        theta = wrap_angle(rpy[2])

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

        # ── v–ω command (forward = robot +X, +ω = CCW) ─────────────────────────
        if abs(heading_err) > ALIGN_THRESHOLD:
            v_lin = 0.0
            w_cmd = max(-W_SPIN_MAX, min(W_SPIN_MAX, K_W * heading_err))
        else:
            speed_scale = min(1.0, max(0.14, dist / APPROACH_RAMP))
            v_lin = min(V_LIN_MAX, K_V * dist) * speed_scale
            w_lim = W_BODY_MAX
            w_cmd = max(-w_lim, min(w_lim, K_W * heading_err))

        half_L = 0.5 * WHEEL_BASELINE
        omega_l = (v_lin - w_cmd * half_L) / WHEEL_RADIUS
        omega_r = (v_lin + w_cmd * half_L) / WHEEL_RADIUS
        omega_l = max(-WHEEL_OMEGA_LIM, min(WHEEL_OMEGA_LIM, omega_l))
        omega_r = max(-WHEEL_OMEGA_LIM, min(WHEEL_OMEGA_LIM, omega_r))
        left_wheel.setVelocity(omega_l)
        right_wheel.setVelocity(omega_r)

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
