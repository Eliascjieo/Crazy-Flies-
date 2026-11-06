"""3-Drone (Scalable to N-Drone) 3D Q-Table + Lyapunov APF Supervisory Node.

Executes a complete hardware/simulation flight experiment:
  Phase 1: Wait for all drone services + odometry & load/train 3D Q-table (432 states)
  Phase 2: Synchronized Takeoff to base_altitude_m (default 1.0 m)
  Phase 3: Fly to 3D Antipodal Sphere/Circle Starting Positions (Radius R = 1.4 m)
  Phase 4: Closed-loop 3D Q-Table + Lyapunov APF crossing through (0, 0, base_z)
  Phase 5: Synchronized Landing & CSV/NPZ Data Logging for the research paper
"""
from __future__ import annotations

import csv
import math
import os
import pickle
import random
import threading
from time import monotonic, sleep

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool

from crazybridge_interfaces.srv import GoTo, Land, Takeoff
from crazybridge.interface import seconds_to_duration

# =====================================================================
# 1. 3D Q-TABLE & LYAPUNOV APF CONSTANTS (EXACT PAPER + OPTION 1 IN 3D)
# =====================================================================
GAIN_SMOOTH_ALPHA = 0.25
DOWNWASH_Z_SCALE = 1.35

BASE_GAINS = {
    'K_I': 0.05,
    'GAMMA_I': 2.5,
    'K_J': 0.20,
    'K_S': 3.0,
    'D_S': 1.25,
    'SWIRL_XY': 0.22,
    'LIFT_Z': 0.08,
    'C_I': 180.0,
}

# Scaled for indoor OptiTrack cage safety (arena radius ~1.4m - 1.8m)
GAIN_ACTIONS = {
    0: {'name': 'Nominal 3D Balanced',  'K_I': 0.11, 'GAMMA_I': 4.2, 'K_J': 0.08, 'K_S': 1.6, 'D_S': 1.15, 'SWIRL_XY': 0.45, 'LIFT_Z': 0.15},
    1: {'name': 'High-Priority Surge',  'K_I': 0.16, 'GAMMA_I': 6.0, 'K_J': 0.01, 'K_S': 1.2, 'D_S': 0.65, 'SWIRL_XY': 0.15, 'LIFT_Z': 0.05},
    2: {'name': 'Micro-Slip 3D Bypass', 'K_I': 0.14, 'GAMMA_I': 5.0, 'K_J': 0.06, 'K_S': 1.4, 'D_S': 1.05, 'SWIRL_XY': 0.70, 'LIFT_Z': 0.25},
    3: {'name': 'Horizontal Yield Arc', 'K_I': 0.11, 'GAMMA_I': 4.0, 'K_J': 0.12, 'K_S': 1.7, 'D_S': 1.45, 'SWIRL_XY': 0.85, 'LIFT_Z': 0.20},
    4: {'name': '3D Altitude + Swirl',  'K_I': 0.09, 'GAMMA_I': 3.5, 'K_J': 0.17, 'K_S': 1.9, 'D_S': 1.75, 'SWIRL_XY': 0.80, 'LIFT_Z': 0.55},
}


# =====================================================================
# 2. 432-STATE 3D DISCRETIZATION FUNCTIONS
# =====================================================================
def discretize_quadrant_xy(vector_3d: np.ndarray) -> str:
    vec_xy = vector_3d[:2]
    if np.linalg.norm(vec_xy) < 1e-5:
        return 'front'
    angle = np.degrees(np.arctan2(vec_xy[1], vec_xy[0]))
    if -45 <= angle < 45:
        return 'right'
    elif 45 <= angle < 135:
        return 'front'
    elif -135 <= angle < -45:
        return 'back'
    return 'left'


def discretize_vertical_bin(relative_position_3d: np.ndarray) -> str:
    dz = relative_position_3d[2]
    if dz > 0.25:
        return 'above'
    elif dz < -0.25:
        return 'below'
    return 'level'


def discretize_ttc_3d(relative_position: np.ndarray, relative_velocity: np.ndarray) -> str:
    distance = float(np.linalg.norm(relative_position))
    if distance < 0.85:
        return 'critical'
    if distance < 1.60:
        return 'warning'

    closing_speed = -float(np.dot(relative_position, relative_velocity)) / (distance + 1e-6)
    if closing_speed <= 0.05:
        return 'safe'

    ttc = distance / closing_speed
    if ttc < 1.8:
        return 'critical'
    elif ttc < 3.8:
        return 'warning'
    return 'safe'


def get_effective_priority(drone: 'LyapunovDroneState3D', all_drones: list['LyapunovDroneState3D']) -> float:
    active_priorities = [other.priority for other in all_drones if not other.reached_goal]
    max_active_priority = max(active_priorities) if active_priorities else 1.0

    if drone.priority >= max_active_priority - 1e-5:
        base_prio = 1.0
    else:
        base_prio = drone.priority

    tie_breaker = 0.001 * (len(all_drones) - drone.id)
    return base_prio + tie_breaker


def discretize_priority_difference(
    self_drone: 'LyapunovDroneState3D',
    other_drone: 'LyapunovDroneState3D | None',
    all_drones: list['LyapunovDroneState3D']
) -> str:
    if other_drone is None:
        return 'higher'
    eff_self = get_effective_priority(self_drone, all_drones)
    eff_other = get_effective_priority(other_drone, all_drones)
    delta = eff_self - eff_other
    if delta > 0.02:
        return 'higher'
    elif delta < -0.02:
        return 'lower'
    return 'higher' if delta >= 0 else 'equal'


def get_closest_drone_3d(
    self_drone: 'LyapunovDroneState3D',
    all_drones: list['LyapunovDroneState3D'],
    ignore_finished_and_passed: bool = True
) -> 'LyapunovDroneState3D | None':
    closest = None
    min_dist = float('inf')
    goal_vec = self_drone.goal - self_drone.position
    goal_dist = float(np.linalg.norm(goal_vec))
    goal_dir = goal_vec / goal_dist if goal_dist > 1e-5 else np.array([1.0, 0.0, 0.0])

    for other in all_drones:
        if other.id == self_drone.id:
            continue
        if ignore_finished_and_passed and other.reached_goal:
            continue

        rel_pos = other.position - self_drone.position
        dist = float(np.linalg.norm(rel_pos))

        if ignore_finished_and_passed and dist > 0.65:
            rel_vel = other.velocity - self_drone.velocity
            closing_speed = -float(np.dot(rel_pos, rel_vel)) / (dist + 1e-6)
            forward_proj = float(np.dot(rel_pos / (dist + 1e-6), goal_dir))
            if forward_proj < -0.15 and closing_speed <= 0.05:
                continue

        if dist < min_dist:
            min_dist = dist
            closest = other
    return closest


def get_observation_3d(
    self_drone: 'LyapunovDroneState3D',
    other_drone: 'LyapunovDroneState3D | None',
    all_drones: list['LyapunovDroneState3D']
) -> tuple[str, str, str, str, str]:
    if other_drone is None:
        return ('front', 'back', 'level', 'safe', 'higher')
    rel_pos = other_drone.position - self_drone.position
    rel_vel = other_drone.velocity - self_drone.velocity
    return (
        discretize_quadrant_xy(self_drone.goal - self_drone.position),
        discretize_quadrant_xy(rel_pos),
        discretize_vertical_bin(rel_pos),
        discretize_ttc_3d(rel_pos, rel_vel),
        discretize_priority_difference(self_drone, other_drone, all_drones),
    )


def select_gain_action_3d(q_table: dict, state: tuple) -> int:
    vert_bin, ttc, prio_diff = state[2], state[3], state[4]
    if ttc == 'safe' or prio_diff == 'higher':
        return 1
    if state in q_table:
        return int(np.argmax(q_table[state]))
    if prio_diff == 'equal':
        return 2
    return 4 if (ttc == 'critical' or vert_bin == 'level') else 3


# =====================================================================
# 3. 3D LYAPUNOV APF DRONE STATE & CONTROLLER
# =====================================================================
class LyapunovDroneState3D:
    def __init__(self, drone_id: int, name: str, start_3d: np.ndarray,
                 goal_3d: np.ndarray, priority: float,
                 v_max: float = 0.5, a_max: float = 1.5,
                 goal_radius: float = 0.22) -> None:
        self.id = drone_id
        self.name = name
        self.start = np.array(start_3d, dtype=float)
        self.position = np.array(start_3d, dtype=float)
        self.goal = np.array(goal_3d, dtype=float)
        self.p_di = np.array(goal_3d, dtype=float)
        self.priority = float(priority)

        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.cmd_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_odom_time: float | None = None
        self.has_odom: bool = False

        self.v_max = v_max
        self.a_max = a_max
        self.goal_radius = goal_radius

        self.k_i = BASE_GAINS['K_I']
        self.gamma_i = BASE_GAINS['GAMMA_I']
        self.k_j = BASE_GAINS['K_J']
        self.k_s = BASE_GAINS['K_S']
        self.d_s = BASE_GAINS['D_S']
        self.swirl_xy = BASE_GAINS['SWIRL_XY']
        self.lift_z = BASE_GAINS['LIFT_Z']
        self.c_i = BASE_GAINS['C_I']

        self.last_action: int = 1
        self.total_distance: float = 0.0
        self.completion_time: float | None = None
        self.reached_goal: bool = False

    def distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal - self.position))

    def update_from_odom(self, pos_xyz: np.ndarray, stamp_s: float) -> None:
        if self.has_odom and self.last_odom_time is not None:
            dt = stamp_s - self.last_odom_time
            if 1e-4 < dt < 0.5:
                raw_vel = (pos_xyz - self.position) / dt
                # Low-pass filter (alpha = 0.35) to reject mocap differentiation noise
                self.velocity = 0.65 * self.velocity + 0.35 * raw_vel
            step_dist = float(np.linalg.norm(pos_xyz - self.position))
            if step_dist < 0.5:
                self.total_distance += step_dist
        self.position = pos_xyz.copy()
        self.last_odom_time = stamp_s
        self.has_odom = True

    def apply_gain_action(self, action_idx: int | None, smooth: bool = True) -> None:
        if action_idx is None:
            self.last_action = -1
            self.k_i = BASE_GAINS['K_I']
            self.gamma_i = BASE_GAINS['GAMMA_I']
            self.k_j = BASE_GAINS['K_J']
            self.k_s = BASE_GAINS['K_S']
            self.d_s = BASE_GAINS['D_S']
            self.swirl_xy = BASE_GAINS['SWIRL_XY']
            self.lift_z = BASE_GAINS['LIFT_Z']
            return

        self.last_action = int(action_idx)
        target = GAIN_ACTIONS[self.last_action]
        alpha = GAIN_SMOOTH_ALPHA if smooth else 1.0
        self.k_i = (1.0 - alpha) * self.k_i + alpha * target['K_I']
        self.gamma_i = (1.0 - alpha) * self.gamma_i + alpha * target['GAMMA_I']
        self.k_j = (1.0 - alpha) * self.k_j + alpha * target['K_J']
        self.k_s = (1.0 - alpha) * self.k_s + alpha * target['K_S']
        self.d_s = (1.0 - alpha) * self.d_s + alpha * target['D_S']
        self.swirl_xy = (1.0 - alpha) * self.swirl_xy + alpha * target['SWIRL_XY']
        self.lift_z = (1.0 - alpha) * self.lift_z + alpha * target['LIFT_Z']

    @staticmethod
    def _effective_distance_3d(diff_j: np.ndarray, use_downwash_ellipsoid: bool = True) -> float:
        if not use_downwash_ellipsoid:
            return float(np.linalg.norm(diff_j))
        dx, dy, dz = diff_j[0], diff_j[1], diff_j[2]
        return float(np.sqrt(dx * dx + dy * dy + (dz / DOWNWASH_Z_SCALE) ** 2))

    def compute_phi(self, all_drones: list['LyapunovDroneState3D'],
                    use_q_improvements: bool = True) -> float:
        dist_term = self.k_i * (self.position - self.p_di)
        norm_dist = float(np.linalg.norm(dist_term))
        phi_att = self.gamma_i * (norm_dist + np.log1p(np.exp(-2.0 * norm_dist)) - np.log(2.0))

        phi_rep = 0.0
        for other in all_drones:
            if other.id == self.id:
                continue
            if use_q_improvements and other.reached_goal:
                continue
            diff_j = self.position - other.position
            dist_j = self._effective_distance_3d(diff_j, use_downwash_ellipsoid=use_q_improvements)
            if 0.001 < dist_j < self.d_s:
                exponent = np.clip(self.k_s * (self.d_s - dist_j), -15.0, 15.0)
                phi_rep += self.k_j * (np.exp(exponent) / dist_j)
        return float(phi_att + phi_rep)

    def compute_gradients(
        self, all_drones: list['LyapunovDroneState3D'], use_q_improvements: bool = True
    ) -> tuple[np.ndarray, float]:
        dist_vector = self.position - self.p_di
        scaled_vec = self.k_i * dist_vector
        norm_dist = float(np.linalg.norm(scaled_vec))

        if norm_dist < 1e-5:
            grad_pi = np.array([0.0, 0.0, 0.0])
        else:
            grad_pi = self.gamma_i * np.tanh(norm_dist) * (self.k_i * scaled_vec) / norm_dist

        sum_grad_pj_vj = 0.0
        goal_vec = self.goal - self.position
        goal_norm = float(np.linalg.norm(goal_vec))
        goal_dir = goal_vec / goal_norm if goal_norm > 1e-5 else np.array([1.0, 0.0, 0.0])

        for other in all_drones:
            if other.id == self.id:
                continue
            if use_q_improvements and other.reached_goal:
                continue

            diff_j = self.position - other.position
            real_dist = float(np.linalg.norm(diff_j))
            dist_j = self._effective_distance_3d(diff_j, use_downwash_ellipsoid=use_q_improvements)

            if 0.001 < dist_j < self.d_s:
                if use_q_improvements and real_dist > 0.65:
                    rel_pos = other.position - self.position
                    rel_vel = other.velocity - self.velocity
                    closing = -float(np.dot(rel_pos, rel_vel)) / real_dist
                    if float(np.dot(rel_pos / real_dist, goal_dir)) < -0.15 and closing <= 0.05:
                        continue

                exponent = np.clip(self.k_s * (self.d_s - dist_j), -15.0, 15.0)
                rep_mag = self.k_j * ((self.k_s * dist_j + 1.0) * np.exp(exponent)) / (dist_j ** 3)

                if use_q_improvements:
                    envelope_taper = np.clip((self.d_s - dist_j) / (0.35 * self.d_s), 0.0, 1.0)
                    rep_mag *= envelope_taper

                radial_dir = diff_j / (real_dist + 1e-8)
                xy_norm = float(np.linalg.norm(diff_j[:2]))
                if xy_norm > 1e-5:
                    tangential_xy = np.array([-diff_j[1] / xy_norm, diff_j[0] / xy_norm, 0.0])
                else:
                    tangential_xy = np.array([0.0, 1.0, 0.0])

                z_sign = 1.0 if (diff_j[2] >= -0.05 or self.id % 2 == 0) else -1.0
                vertical_z = np.array([0.0, 0.0, z_sign])

                if use_q_improvements:
                    radial_weight = max(0.25, 1.0 - 0.60 * self.swirl_xy - 0.30 * self.lift_z)
                    combined_dir = (
                        radial_weight * radial_dir
                        + self.swirl_xy * tangential_xy
                        + self.lift_z * vertical_z
                    )
                else:
                    combined_dir = radial_dir + 0.20 * tangential_xy + 0.05 * vertical_z

                combined_dir /= (np.linalg.norm(combined_dir) + 1e-8)
                grad_pj = rep_mag * dist_j * combined_dir
                grad_pi -= grad_pj
                sum_grad_pj_vj += float(np.dot(grad_pj, other.velocity))

        return grad_pi, sum_grad_pj_vj

    def compute_desired_velocity(
        self, all_drones: list['LyapunovDroneState3D'],
        dt: float, elapsed_s: float,
        use_q_improvements: bool = True,
        min_z: float = 0.45, max_z: float = 1.75, max_xy: float = 2.4
    ) -> np.ndarray:
        """Computes Eq. 10 3D velocity with acceleration slew-rate and arena clamps."""
        if self.reached_goal or self.distance_to_goal() < self.goal_radius:
            if not self.reached_goal:
                self.reached_goal = True
                self.completion_time = elapsed_s
            self.cmd_velocity = np.array([0.0, 0.0, 0.0])
            return self.cmd_velocity

        phi_i = self.compute_phi(all_drones, use_q_improvements=use_q_improvements)
        grad_pi, sum_grad_pj_vj = self.compute_gradients(
            all_drones, use_q_improvements=use_q_improvements
        )
        norm_grad_pi_sq = float(np.dot(grad_pi, grad_pi))

        if norm_grad_pi_sq < 1e-7:
            v_desired = np.array([0.0, 0.0, 0.0])
        else:
            w_i = self.c_i * phi_i
            numerator = max(0.15 * w_i, w_i + sum_grad_pj_vj)
            v_desired = -(numerator / norm_grad_pi_sq) * grad_pi

        # 1. Velocity Magnitude Clipping (V_MAX)
        speed = float(np.linalg.norm(v_desired))
        if speed > self.v_max:
            v_desired = (v_desired / speed) * self.v_max

        # 2. Slew-Rate / Acceleration Clamping (||dv|| <= a_max * dt)
        dv = v_desired - self.cmd_velocity
        dv_norm = float(np.linalg.norm(dv))
        max_dv = self.a_max * dt
        if dv_norm > max_dv:
            v_desired = self.cmd_velocity + (dv / dv_norm) * max_dv

        # 3. Cage Boundary & Floor/Ceiling Protection
        if self.position[2] < min_z and v_desired[2] < 0.0:
            v_desired[2] = 0.15
        elif self.position[2] > max_z and v_desired[2] > 0.0:
            v_desired[2] = -0.15

        for axis in (0, 1):
            if self.position[axis] > max_xy and v_desired[axis] > 0.0:
                v_desired[axis] = 0.0
            elif self.position[axis] < -max_xy and v_desired[axis] < 0.0:
                v_desired[axis] = 0.0

        self.cmd_velocity = v_desired
        return self.cmd_velocity


# =====================================================================
# 4. FAST OFFLINE 3D Q-TABLE AUTO-TRAINER (FALLBACK IF .PKL MISSING)
# =====================================================================
def ensure_3d_q_table(pkl_path: str, logger) -> dict:
    if os.path.exists(pkl_path):
        logger.info(f'Loading pre-trained 3D Q-Table from {pkl_path}')
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    logger.info(
        f'{pkl_path} not found. Training 432-state 3D Q-Table offline now (~4 seconds)...'
    )
    q_table: dict = {}
    alpha, gamma, epsilon = 0.18, 0.92, 1.0
    dt_sim = 0.05

    for _ep in range(900):
        base_theta = random.uniform(0, 2 * math.pi)
        prios = [1.0, 0.5, 0.0]
        sim_drones = []
        for i in range(6):
            theta = base_theta + i * (2 * math.pi / 6)
            z_s = random.uniform(-0.4, 0.4)
            r_xy = 1.5
            start = np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), 1.0 + z_s])
            goal = np.array([-start[0], -start[1], 1.0 - z_s])
            sim_drones.append(
                LyapunovDroneState3D(i, f'sim_{i}', start, goal, prios[i % 3], v_max=0.6)
            )

        states = [
            get_observation_3d(d, get_closest_drone_3d(d, sim_drones, True), sim_drones)
            for d in sim_drones
        ]
        for _step in range(120):
            actions = []
            for s in states:
                if s not in q_table:
                    q_table[s] = np.zeros(len(GAIN_ACTIONS))
                if random.uniform(0, 1) < epsilon:
                    actions.append(random.choice(list(GAIN_ACTIONS.keys())))
                else:
                    actions.append(int(np.argmax(q_table[s])))

            old_dists = [d.distance_to_goal() for d in sim_drones]
            was_done = [d.reached_goal for d in sim_drones]

            for i, d in enumerate(sim_drones):
                d.apply_gain_action(actions[i], smooth=True)
            for d in sim_drones:
                v_cmd = d.compute_desired_velocity(
                    sim_drones, dt_sim, _step * dt_sim, use_q_improvements=True
                )
                d.position += v_cmd * dt_sim
                d.velocity = v_cmd

            next_states = [
                get_observation_3d(d, get_closest_drone_3d(d, sim_drones, True), sim_drones)
                for d in sim_drones
            ]
            for i, d in enumerate(sim_drones):
                if was_done[i]:
                    continue
                s, a, ns = states[i], actions[i], next_states[i]
                if ns not in q_table:
                    q_table[ns] = np.zeros(len(GAIN_ACTIONS))
                prog = old_dists[i] - d.distance_to_goal()
                eff_p = min(1.0, get_effective_priority(d, sim_drones))
                r = prog * 85.0 * (1.0 + 0.6 * eff_p) - 1.2
                if prog < 0:
                    r += prog * 60.0
                vert_bin, ttc_s, prio_d = s[2], s[3], s[4]
                if ttc_s == 'safe' or prio_d == 'higher':
                    r += 5.0 if a == 1 else (-4.0 if a >= 3 else 0.0)
                elif prio_d == 'equal' and a == 2:
                    r += 5.0
                elif prio_d == 'lower':
                    if vert_bin == 'level' and ttc_s == 'critical' and a == 4:
                        r += 6.0
                    elif ttc_s in ('warning', 'critical') and a in (3, 4):
                        r += 5.0
                if d.reached_goal:
                    r += 1000.0
                q_table[s][a] += alpha * (r + gamma * float(np.max(q_table[ns])) - q_table[s][a])

            states = next_states
            if all(d.reached_goal for d in sim_drones):
                break
        epsilon = max(0.05, epsilon * 0.994)

    os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(q_table, f)
    logger.info(f'Saved 3D Q-Table ({len(q_table)} states) to {pkl_path}')
    return q_table


# =====================================================================
# 5. ROS 2 MULTI-DRONE SUPERVISORY CONTROLLER NODE
# =====================================================================
class SwarmQApfNode(Node):
    def __init__(self) -> None:
        super().__init__('swarm_q_apf_node')

        self.declare_parameter('drone_names', ['cf1', 'cf2', 'cf3'])
        self.declare_parameter('priorities', [1.0, 0.5, 0.0])
        self.declare_parameter('arena_radius_m', 1.35)
        self.declare_parameter('base_altitude_m', 1.0)
        self.declare_parameter('z_offsets_m', [0.0, 0.15, -0.15])
        self.declare_parameter('v_max_mps', 0.45)
        self.declare_parameter('a_max_mps2', 1.2)
        self.declare_parameter('goal_radius_m', 0.22)
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('max_flight_time_s', 25.0)
        self.declare_parameter('use_q_table', True)
        # 'goto' (safe High-Level Commander lookahead) or 'cmd_vel' (20Hz world velocity)
        self.declare_parameter('control_mode', 'goto')
        self.declare_parameter('q_table_path', '/tmp/q_table_apf_gains_3d.pkl')
        self.declare_parameter('output_dir', '/tmp/swarm_q_apf_results')
        self.declare_parameter('startup_timeout_s', 45.0)

        self.drone_names: list[str] = list(
            self.get_parameter('drone_names').get_parameter_value().string_array_value
        )
        priorities: list[float] = list(
            self.get_parameter('priorities').get_parameter_value().double_array_value
        )
        z_offsets: list[float] = list(
            self.get_parameter('z_offsets_m').get_parameter_value().double_array_value
        )
        self.radius = self.get_parameter('arena_radius_m').get_parameter_value().double_value
        self.base_z = self.get_parameter('base_altitude_m').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max_mps').get_parameter_value().double_value
        self.a_max = self.get_parameter('a_max_mps2').get_parameter_value().double_value
        self.goal_radius = self.get_parameter('goal_radius_m').get_parameter_value().double_value
        self.rate_hz = self.get_parameter('control_rate_hz').get_parameter_value().double_value
        self.max_time_s = self.get_parameter('max_flight_time_s').get_parameter_value().double_value
        self.use_q_table = self.get_parameter('use_q_table').get_parameter_value().bool_value
        self.control_mode = self.get_parameter('control_mode').get_parameter_value().string_value
        self.q_table_path = self.get_parameter('q_table_path').get_parameter_value().string_value
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.startup_timeout_s = self.get_parameter('startup_timeout_s').get_parameter_value().double_value

        num_drones = len(self.drone_names)
        self.drones: list[LyapunovDroneState3D] = []
        self.takeoff_clients = {}
        self.land_clients = {}
        self.goto_clients = {}
        self.kill_clients = {}
        self.cmd_vel_pubs = {}
        self._lock = threading.Lock()
        self.airborne = False

        for i, name in enumerate(self.drone_names):
            theta = i * (2.0 * math.pi / num_drones)
            z_off = z_offsets[i % len(z_offsets)] if z_offsets else 0.0
            start_3d = np.array([
                self.radius * math.cos(theta),
                self.radius * math.sin(theta),
                self.base_z + z_off,
            ])
            goal_3d = np.array([
                -start_3d[0],
                -start_3d[1],
                self.base_z - z_off,
            ])
            prio = priorities[i % len(priorities)] if priorities else 0.5
            drone = LyapunovDroneState3D(
                i, name, start_3d, goal_3d, prio,
                v_max=self.v_max, a_max=self.a_max, goal_radius=self.goal_radius
            )
            self.drones.append(drone)

            self.takeoff_clients[name] = self.create_client(Takeoff, f'/{name}/takeoff')
            self.land_clients[name] = self.create_client(Land, f'/{name}/land')
            self.goto_clients[name] = self.create_client(GoTo, f'/{name}/go_to')
            self.kill_clients[name] = self.create_client(SetBool, f'/{name}/kill')
            self.cmd_vel_pubs[name] = self.create_publisher(Vector3, f'/{name}/cmd_vel', 10)

            self.create_subscription(
                Odometry, f'/{name}/odometry',
                lambda msg, idx=i: self._odom_cb(idx, msg), 10
            )

        self.history_rows: list[list] = []

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom_cb(self, idx: int, msg: Odometry) -> None:
        p = msg.pose.pose.position
        pos_xyz = np.array([float(p.x), float(p.y), float(p.z)])
        with self._lock:
            self.drones[idx].update_from_odom(pos_xyz, self._now_s())

    def wait_for_swarm_ready(self) -> None:
        self.get_logger().info(
            f'Waiting up to {self.startup_timeout_s:.0f}s for {self.drone_names} '
            'odometry and /takeoff services (OptiTrack safety lock)...'
        )
        deadline = monotonic() + self.startup_timeout_s
        while monotonic() < deadline:
            all_services = all(
                self.takeoff_clients[n].service_is_ready() and
                self.land_clients[n].service_is_ready() and
                self.goto_clients[n].service_is_ready()
                for n in self.drone_names
            )
            with self._lock:
                all_odom = all(d.has_odom for d in self.drones)
            if all_services and all_odom:
                self.get_logger().info('All drones connected, tracked by OptiTrack, and ready!')
                return
            sleep(0.2)

        missing_srv = [
            n for n in self.drone_names if not self.takeoff_clients[n].service_is_ready()
        ]
        with self._lock:
            missing_odom = [d.name for d in self.drones if not d.has_odom]
        raise RuntimeError(
            f'Swarm startup timed out! Missing takeoff service: {missing_srv}, '
            f'Missing odometry: {missing_odom}'
        )

    def _call_all(self, clients_dict: dict, req_fn, label: str, timeout_s: float = 10.0) -> None:
        futures = {}
        for name in self.drone_names:
            req = req_fn(name)
            futures[name] = clients_dict[name].call_async(req)

        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            if all(f.done() for f in futures.values()):
                break
            sleep(0.02)

        for name, f in futures.items():
            if not f.done():
                raise RuntimeError(f'{label} timed out on {name}')
            resp = f.result()
            if not getattr(resp, 'success', True):
                raise RuntimeError(f'{label} failed on {name}: {getattr(resp, "message", "")}')
        self.get_logger().info(f'{label}: OK across all {len(self.drone_names)} drones')

    def run_mission(self) -> None:
        q_table = ensure_3d_q_table(self.q_table_path, self.get_logger())
        self.wait_for_swarm_ready()

        # Phase 1: Synchronized Takeoff
        self.airborne = True
        self.get_logger().info(f'Phase 1: Taking off all drones to z={self.base_z:.2f}m...')
        self._call_all(
            self.takeoff_clients,
            lambda _n: Takeoff.Request(
                height=float(self.base_z),
                duration=seconds_to_duration(3.0),
                group_mask=0
            ),
            'takeoff'
        )
        sleep(4.5)

        # Phase 2: Move to Antipodal Start Positions on the Circle/Sphere
        self.get_logger().info('Phase 2: Flying to 3D antipodal start positions...')
        name_to_drone = {d.name: d for d in self.drones}
        self._call_all(
            self.goto_clients,
            lambda n: GoTo.Request(
                relative=False,
                goal=Point(
                    x=float(name_to_drone[n].start[0]),
                    y=float(name_to_drone[n].start[1]),
                    z=float(name_to_drone[n].start[2]),
                ),
                yaw=0.0,
                duration=seconds_to_duration(4.0),
                group_mask=0
            ),
            'go_to_start'
        )
        sleep(5.0)

        # Reset distance counters right before the Q-Table + APF crossing starts
        with self._lock:
            for d in self.drones:
                d.total_distance = 0.0
                d.reached_goal = False
                if self.use_q_table:
                    d.apply_gain_action(1, smooth=False)
                else:
                    d.apply_gain_action(None, smooth=False)

        # Phase 3: Closed-Loop 3D Q-Table + Lyapunov APF Crossing
        self.get_logger().info(
            f'Phase 3: Starting 3D {"Q-Table + APF" if self.use_q_table else "Baseline Static APF"} '
            f'crossing (mode={self.control_mode}, rate={self.rate_hz}Hz)...'
        )
        dt = 1.0 / self.rate_hz
        t_start = monotonic()
        step_idx = 0

        while monotonic() - t_start < self.max_time_s:
            loop_t0 = monotonic()
            elapsed = loop_t0 - t_start

            with self._lock:
                if all(d.reached_goal for d in self.drones):
                    self.get_logger().info(f'All drones reached their 3D goals in {elapsed:.2f}s!')
                    break

                # 1. Q-Table Gain Selection
                for d in self.drones:
                    if d.reached_goal:
                        continue
                    if self.use_q_table:
                        closest = get_closest_drone_3d(d, self.drones, ignore_finished_and_passed=True)
                        state = get_observation_3d(d, closest, self.drones)
                        action = select_gain_action_3d(q_table, state)
                        d.apply_gain_action(action, smooth=True)
                    else:
                        d.apply_gain_action(None, smooth=False)

                # 2. Continuous 3D Lyapunov APF Velocity Update
                cmds = {}
                for d in self.drones:
                    v_cmd = d.compute_desired_velocity(
                        self.drones, dt, elapsed, use_q_improvements=self.use_q_table
                    )
                    cmds[d.name] = (d.position.copy(), v_cmd.copy(), d.reached_goal)

                # Log telemetry row
                for d in self.drones:
                    self.history_rows.append([
                        elapsed, d.id, d.name, d.priority,
                        d.position[0], d.position[1], d.position[2],
                        d.velocity[0], d.velocity[1], d.velocity[2],
                        d.cmd_velocity[0], d.cmd_velocity[1], d.cmd_velocity[2],
                        d.last_action, d.d_s, int(d.reached_goal)
                    ])

            # 3. Dispatch Setpoints to Crazyflies
            for name, (pos, v_cmd, done) in cmds.items():
                if self.control_mode == 'cmd_vel':
                    self.cmd_vel_pubs[name].publish(
                        Vector3(x=float(v_cmd[0]), y=float(v_cmd[1]), z=float(v_cmd[2]))
                    )
                else:
                    # 'goto' mode: issue smooth High-Level Commander lookahead every 2 ticks (10 Hz)
                    if step_idx % 2 == 0 and not done:
                        horizon = 0.45
                        target = pos + v_cmd * horizon
                        req = GoTo.Request(
                            relative=False,
                            goal=Point(x=float(target[0]), y=float(target[1]), z=float(target[2])),
                            yaw=0.0,
                            duration=seconds_to_duration(0.45),
                            group_mask=0
                        )
                        self.goto_clients[name].call_async(req)

            step_idx += 1
            sleep_time = dt - (monotonic() - loop_t0)
            if sleep_time > 0:
                sleep(sleep_time)

        # Phase 4: Synchronized Landing
        self.get_logger().info('Phase 4: Landing all drones...')
        self._call_all(
            self.land_clients,
            lambda _n: Land.Request(
                height=0.0,
                duration=seconds_to_duration(3.0),
                group_mask=0
            ),
            'land'
        )
        sleep(3.5)
        self.airborne = False
        self.save_results()

    def emergency_stop_all(self) -> None:
        self.get_logger().error('Executing Emergency Land + Kill across all drones!')
        for name in self.drone_names:
            try:
                self.land_clients[name].call_async(
                    Land.Request(height=0.0, duration=seconds_to_duration(2.0), group_mask=0)
                )
            except Exception:
                pass
        sleep(2.2)
        for name in self.drone_names:
            try:
                self.kill_clients[name].call_async(SetBool.Request(data=True))
            except Exception:
                pass

    def save_results(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        tag = 'q_table_3d' if self.use_q_table else 'baseline_apf_3d'
        csv_path = os.path.join(self.output_dir, f'flight_log_{tag}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time_s', 'drone_id', 'name', 'priority',
                'x', 'y', 'z', 'vx', 'vy', 'vz',
                'cmd_vx', 'cmd_vy', 'cmd_vz', 'action', 'd_s', 'reached_goal'
            ])
            writer.writerows(self.history_rows)

        self.get_logger().info('=' * 72)
        self.get_logger().info(f'3D FLIGHT SUMMARY ({tag.upper()}):')
        for d in self.drones:
            t_fin = d.completion_time if d.completion_time is not None else self.max_time_s
            self.get_logger().info(
                f'  {d.name} (Priority {d.priority:.1f}) | '
                f'3D Distance = {d.total_distance:.2f} m | '
                f'Time = {t_fin:.2f} s | Reached Goal = {d.reached_goal}'
            )
        self.get_logger().info(f'Saved full telemetry CSV to: {csv_path}')
        self.get_logger().info('=' * 72)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SwarmQApfNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    exit_code = 0
    try:
        node.run_mission()
    except KeyboardInterrupt:
        node.get_logger().warning('KeyboardInterrupt detected!')
        if node.airborne:
            node.emergency_stop_all()
    except Exception as exc:
        node.get_logger().error(f'Swarm mission failed: {exc}')
        if node.airborne:
            node.emergency_stop_all()
        exit_code = 1
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()