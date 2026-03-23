"""
CSCI/ECEN 3302 - Introduction to Robotics
Lab: Gradient Descent IK Block Stacking
SOLUTION KEY - FOR TA USE ONLY
University of Colorado Boulder | TA: Jay Vakil
"""

import numpy as np
import sys

try:
    from controller import Robot, Motor, PositionSensor
except ImportError:
    print("ERROR: Must run inside Webots."); sys.exit(1)

# ============================================================
# UR5e Standard DH Parameters
# ============================================================
# UR publishes these in the STANDARD (Paul/Spong) DH convention.
# Joint | a (m)    | d (m)   | alpha (rad)
#   1   | 0.0      | 0.1625  | pi/2
#   2   | -0.425   | 0.0     | 0
#   3   | -0.3922  | 0.0     | 0
#   4   | 0.0      | 0.1333  | pi/2
#   5   | 0.0      | 0.0997  | -pi/2
#   6   | 0.0      | 0.0996  | 0

UR5E_DH = {
    'a':     [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
    'd':     [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
    'alpha': [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0],
}
JOINT_LIMITS_LOWER = np.array([-2*np.pi]*2 + [-np.pi] + [-2*np.pi]*3)
JOINT_LIMITS_UPPER = np.array([2*np.pi]*2 + [np.pi] + [2*np.pi]*3)

# ============================================================
# Task constants
# ============================================================
BLOCK_SIZE = 0.05
GRIPPER_OFFSET = 0.10
APPROACH_HEIGHT = 0.15
TABLE_HEIGHT = 0.74

# Robot base pose in world frame (from .wbt file)
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.74])
ROBOT_BASE_YAW = np.pi / 2  # 90 deg about Z

# Stack target in world frame
STACK_TARGET_WORLD = np.array([0.35, 0.2])

# IK parameters
GD_LEARNING_RATE = 0.5
GD_MAX_ITERATIONS = 2000
GD_TOLERANCE = 1e-3
GD_ORIENTATION_WEIGHT = 0.3
DLS_DAMPING = 0.05


# ============================================================
# Coordinate frame transforms
# ============================================================
def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

_R_base = _rot_z(ROBOT_BASE_YAW)
_R_base_inv = _R_base.T

def world_to_base(pos_world):
    return _R_base_inv @ (np.array(pos_world) - ROBOT_BASE_POS)

def base_to_world(pos_base):
    return _R_base @ np.array(pos_base) + ROBOT_BASE_POS


# ============================================================
# Robot Interface
# ============================================================
class UR5eInterface:
    MOTOR_NAMES = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    SENSOR_NAMES = [s + "_sensor" for s in MOTOR_NAMES]
    GRIPPER_MOTORS = ["finger_left", "finger_right"]

    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.motors, self.sensors = [], []
        for i in range(6):
            m = robot.getDevice(self.MOTOR_NAMES[i])
            s = robot.getDevice(self.SENSOR_NAMES[i])
            s.enable(self.timestep); m.setVelocity(1.0)
            self.motors.append(m); self.sensors.append(s)
        self.gripper_motors = []
        for name in self.GRIPPER_MOTORS:
            m = robot.getDevice(name)
            if m:
                m.setVelocity(0.5)
                self.gripper_motors.append(m)
        for _ in range(10):
            robot.step(self.timestep)

    def get_joint_positions(self):
        return np.array([s.getValue() for s in self.sensors])

    def set_joint_positions(self, q):
        q = np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
        for i in range(6):
            self.motors[i].setPosition(float(q[i]))

    def open_gripper(self):
        if len(self.gripper_motors) >= 2:
            self.gripper_motors[0].setPosition(0.0)
            self.gripper_motors[1].setPosition(0.0)

    def close_gripper(self):
        if len(self.gripper_motors) >= 2:
            self.gripper_motors[0].setPosition(-0.04)
            self.gripper_motors[1].setPosition(0.04)

    def step(self, duration_ms=None):
        if duration_ms is None:
            return self.robot.step(self.timestep)
        for _ in range(int(duration_ms / self.timestep)):
            if self.robot.step(self.timestep) == -1:
                return -1
        return 0

    def get_block_positions_world(self):
        positions = {}
        try:
            for name in ['block_A', 'block_B', 'block_C']:
                node = self.robot.getFromDef(name.upper())
                if node is None:
                    node = self.robot.getFromDef(name)
                if node is not None:
                    pos = node.getField('translation').getSFVec3f()
                    positions[name] = np.array(pos)
        except Exception:
            pass
        if len(positions) < 3:
            positions = {
                'block_A': np.array([0.35, -0.2, 0.79]),
                'block_B': np.array([0.5, -0.25, 0.79]),
                'block_C': np.array([0.6, -0.12, 0.79]),
            }
        return positions


# ============================================================
# Helpers
# ============================================================
def orientation_error(R_desired, R_current):
    R_err = R_desired @ R_current.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    if abs(angle) < 1e-8:
        return np.zeros(3)
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1]
    ]) / (2 * np.sin(angle))
    return angle * axis


# ============================================================
# SOLUTION: Forward Kinematics (Standard DH)
# ============================================================
def dh_transform(a, d, alpha, theta):
    """Standard (Paul/Spong) DH transformation matrix."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa,  a*ct],
        [st,  ct*ca, -ct*sa,  a*st],
        [0,   sa,     ca,     d   ],
        [0,   0,      0,      1   ]])


def forward_kinematics(q):
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(UR5E_DH['a'][i], UR5E_DH['d'][i],
                              UR5E_DH['alpha'][i], q[i])
    T_tool = np.eye(4)
    T_tool[2, 3] = GRIPPER_OFFSET
    return T @ T_tool


# ============================================================
# SOLUTION: Jacobian
# ============================================================
def compute_jacobian(q, delta=1e-5):
    J = np.zeros((6, 6))
    for i in range(6):
        qp, qm = q.copy(), q.copy()
        qp[i] += delta
        qm[i] -= delta
        Tp = forward_kinematics(qp)
        Tm = forward_kinematics(qm)
        J[0:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * delta)
        J[3:6, i] = orientation_error(Tp[:3, :3], Tm[:3, :3]) / (2 * delta)
    return J


# ============================================================
# SOLUTION: Gradient Descent IK (Jacobian Transpose)
# ============================================================
def gradient_descent_ik(target_pose, q_init, max_iter=GD_MAX_ITERATIONS,
                         lr=GD_LEARNING_RATE, tol=GD_TOLERANCE,
                         orientation_weight=GD_ORIENTATION_WEIGHT):
    q = q_init.copy()
    p_target = target_pose[:3, 3]
    R_target = target_pose[:3, :3]
    error_log = []
    for _ in range(max_iter):
        T = forward_kinematics(q)
        pos_err = p_target - T[:3, 3]
        orient_err = orientation_error(R_target, T[:3, :3])
        cost = np.linalg.norm(pos_err)**2 + orientation_weight * np.linalg.norm(orient_err)**2
        error_log.append(cost)
        if np.linalg.norm(pos_err) < tol and np.linalg.norm(orient_err) < tol * 5:
            return q, True, error_log
        J = compute_jacobian(q)
        e = np.concatenate([pos_err, orientation_weight * orient_err])
        q = np.clip(q + lr * J.T @ e, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    return q, False, error_log


# ============================================================
# SOLUTION: Damped Least Squares IK
# ============================================================
def damped_least_squares_ik(target_pose, q_init, max_iter=GD_MAX_ITERATIONS,
                             damping=DLS_DAMPING, tol=GD_TOLERANCE,
                             orientation_weight=GD_ORIENTATION_WEIGHT):
    q = q_init.copy()
    p_target = target_pose[:3, 3]
    R_target = target_pose[:3, :3]
    lam = damping
    prev_cost = float('inf')
    error_log = []
    for _ in range(max_iter):
        T = forward_kinematics(q)
        pos_err = p_target - T[:3, 3]
        orient_err = orientation_error(R_target, T[:3, :3])
        cost = np.linalg.norm(pos_err)**2 + orientation_weight * np.linalg.norm(orient_err)**2
        error_log.append(cost)
        if np.linalg.norm(pos_err) < tol and np.linalg.norm(orient_err) < tol * 5:
            return q, True, error_log
        if cost < prev_cost:
            lam = max(lam * 0.7, 1e-4)
        else:
            lam = min(lam * 2.0, 1.0)
        prev_cost = cost
        J = compute_jacobian(q)
        e = np.concatenate([pos_err, orientation_weight * orient_err])
        dq = np.linalg.solve(J.T @ J + lam**2 * np.eye(6), J.T @ e)
        q = np.clip(q + dq, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)
    return q, False, error_log


# ============================================================
# SOLUTION: Trajectory Interpolation
# ============================================================
def rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        r = q0 + t * (q1 - q0)
        return r / np.linalg.norm(r)
    th = np.arccos(np.clip(dot, -1, 1))
    sth = np.sin(th)
    r = (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / sth
    return r / np.linalg.norm(r)


def interpolate_trajectory(pose_start, pose_end, num_waypoints=20):
    ps = pose_start[:3, 3]
    pe = pose_end[:3, 3]
    qs = rot_to_quat(pose_start[:3, :3])
    qe = rot_to_quat(pose_end[:3, :3])
    waypoints = []
    for i in range(num_waypoints):
        t = i / max(num_waypoints - 1, 1)
        s = 6*t**5 - 15*t**4 + 10*t**3
        T = np.eye(4)
        T[:3, 3] = (1 - s) * ps + s * pe
        T[:3, :3] = quat_to_rot(slerp(qs, qe, s))
        waypoints.append(T)
    return waypoints


# ============================================================
# SOLUTION: Stacking Task
# ============================================================
def make_base_pose(pos_world, R_base):
    T = np.eye(4)
    T[:3, :3] = R_base
    T[:3, 3] = world_to_base(pos_world)
    return T


def move_to(iface, target_base, q, nwp=20, delay=100):
    T_cur = forward_kinematics(q)
    waypoints = interpolate_trajectory(T_cur, target_base, nwp)
    for wp in waypoints:
        q_new, conv, _ = damped_least_squares_ik(wp, q, max_iter=300, tol=2e-3)
        if not conv:
            err = np.linalg.norm(wp[:3, 3] - forward_kinematics(q_new)[:3, 3])
            if err > 0.01:
                print(f"  WARNING: IK pos err: {err:.4f}")
        q = q_new
        iface.set_joint_positions(q)
        iface.step(delay)
    return q


def execute_stacking_task(iface):
    blocks_world = iface.get_block_positions_world()
    print(f"Block positions (world): {blocks_world}")

    # Grasp orientation: tool pointing straight down in base frame
    R_grasp = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    q = iface.get_joint_positions()
    iface.open_gripper()
    iface.step(500)

    T_cur = forward_kinematics(q)
    print(f"Current EE base: {T_cur[:3, 3].round(4)}")
    print(f"Current EE world: {base_to_world(T_cur[:3, 3]).round(4)}")

    for level, name in enumerate(['block_A', 'block_B', 'block_C']):
        print(f"\n{'='*50}")
        print(f"PICKING: {name} (stack level {level})")
        print(f"{'='*50}")

        bp = blocks_world[name]
        print(f"  Block world: {bp.round(4)}, base: {world_to_base(bp).round(4)}")

        grasp_z = bp[2]  # block center Z (world)
        place_z = TABLE_HEIGHT + level * BLOCK_SIZE + BLOCK_SIZE / 2

        # 1. Pre-grasp
        q = move_to(iface, make_base_pose([bp[0], bp[1], grasp_z + APPROACH_HEIGHT], R_grasp), q, 25, 80)
        print(f"  -> Pre-grasp done")

        # 2. Descend
        q = move_to(iface, make_base_pose([bp[0], bp[1], grasp_z], R_grasp), q, 10, 100)
        print(f"  -> Descended")

        # 3. Grasp
        iface.close_gripper()
        iface.step(800)
        print(f"  -> Grasped")

        # 4. Lift
        q = move_to(iface, make_base_pose([bp[0], bp[1], grasp_z + APPROACH_HEIGHT + 0.05], R_grasp), q, 10, 80)
        print(f"  -> Lifted")

        # 5. Transit high
        mid = [(bp[0] + STACK_TARGET_WORLD[0]) / 2,
               (bp[1] + STACK_TARGET_WORLD[1]) / 2,
               TABLE_HEIGHT + 0.35]
        q = move_to(iface, make_base_pose(mid, R_grasp), q, 15, 80)
        print(f"  -> Transit")

        # 6. Pre-place
        q = move_to(iface, make_base_pose([STACK_TARGET_WORLD[0], STACK_TARGET_WORLD[1],
                                            place_z + APPROACH_HEIGHT], R_grasp), q, 15, 80)
        print(f"  -> Pre-place")

        # 7. Place
        q = move_to(iface, make_base_pose([STACK_TARGET_WORLD[0], STACK_TARGET_WORLD[1],
                                            place_z], R_grasp), q, 10, 100)
        print(f"  -> Placed")

        # 8. Release
        iface.open_gripper()
        iface.step(800)
        print(f"  -> Released")

        # 9. Retreat
        q = move_to(iface, make_base_pose([STACK_TARGET_WORLD[0], STACK_TARGET_WORLD[1],
                                            place_z + APPROACH_HEIGHT], R_grasp), q, 8, 80)
        print(f"  {name} DONE!")

    print(f"\n{'='*50}")
    print("STACKING COMPLETE!")
    print(f"{'='*50}")


# ============================================================
# Diagnostics
# ============================================================
def run_diagnostics(iface):
    print("=" * 60)
    print("  DIAGNOSTIC TESTS")
    print("=" * 60)

    q_home = np.zeros(6)
    q_start = iface.get_joint_positions()

    T_home = forward_kinematics(q_home)
    print(f"  FK at q=0 (base):  {T_home[:3, 3].round(4)}")
    print(f"  FK at q=0 (world): {base_to_world(T_home[:3, 3]).round(4)}")

    T_start = forward_kinematics(q_start)
    print(f"  FK at q_start (base):  {T_start[:3, 3].round(4)}")
    print(f"  FK at q_start (world): {base_to_world(T_start[:3, 3]).round(4)}")

    J = compute_jacobian(q_start)
    print(f"  Jacobian rank: {np.linalg.matrix_rank(J, tol=1e-4)}, "
          f"cond: {np.linalg.cond(J):.2f}")

    # Reachability test: Block A
    bp_base = world_to_base([0.35, -0.2, 0.79])
    R_grasp = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    target = np.eye(4)
    target[:3, :3] = R_grasp
    target[:3, 3] = bp_base

    q_sol, conv, errs = damped_least_squares_ik(target, q_start, max_iter=500)
    T_sol = forward_kinematics(q_sol)
    err = np.linalg.norm(T_sol[:3, 3] - bp_base)
    print(f"\n  Block A reach test:")
    print(f"    converged: {conv}, pos err: {err:.6f} m, iters: {len(errs)}")
    print(f"    EE base:  {T_sol[:3, 3].round(4)}")
    print(f"    EE world: {base_to_world(T_sol[:3, 3]).round(4)}")

    return conv or err < 0.01


def main():
    robot = Robot()
    iface = UR5eInterface(robot)
    print("\n" + "=" * 60)
    print("  SOLUTION KEY - GD IK Block Stacking")
    print("=" * 60)

    q_start = np.array([0.0, -np.pi/4, np.pi/4, -np.pi/2, -np.pi/2, 0.0])
    iface.set_joint_positions(q_start)
    iface.step(2000)

    if run_diagnostics(iface):
        print("\nDiagnostics passed! Starting stacking...\n")
        iface.step(1000)
        execute_stacking_task(iface)
    else:
        print("\nDiagnostics FAILED - IK cannot reach targets.")

    while robot.step(int(robot.getBasicTimeStep())) != -1:
        pass


if __name__ == "__main__":
    main()
