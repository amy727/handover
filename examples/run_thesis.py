# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import os
import handover
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client
import numpy as np
import hydra
import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import polymetis_pb2
from polymetis import RobotInterface, GripperInterface
import polysim
from polymetis.utils.data_dir import get_full_path_to_urdf

from scipy.spatial.transform import Rotation as Rot

from handover.config import get_config_from_args
from handover.benchmark_runner import BenchmarkRunner

from demo_benchmark_wrapper import start_conf, time_wait
import threading


panda_urdf_file = os.path.join(
    os.path.dirname(handover.__file__), "data", "assets", "franka_panda", "panda_gripper.urdf"
)
grasp_dir = os.path.join(os.path.dirname(handover.__file__), "data", "assets", "grasps")


class WeightedPoseDistance:
    def __init__(self, position_wt=1.0, rotation_wt=0.001):
        self._position_wt = position_wt
        self._rotation_wt = rotation_wt

    def __call__(self, pq1, pq2):
        pos = self._position_wt * np.sum((pq1[..., 0:3] - pq2[..., 0:3]) ** 2, axis=-1)
        rot = self._rotation_wt * quat_loss(pq1[..., 3:7], pq2[..., 3:7])
        return pos + rot


def quat_loss(q1, q2):
    return 1 - np.power(np.sum(q1 * q2, axis=-1), 2)


class ApproachRegionCondition:
    def __init__(self, slope=10.0, pos_tol=1.5e-2, max_pos_tol=5e-2, theta_tol=np.radians(10.0)):
        self._slope = slope
        self._pos_tol = pos_tol
        self._max_pos_tol = max_pos_tol
        self._theta_tol = theta_tol

        self._start_pose = None
        self._final_pose = None

    def set_region(self, pose0, pose1):
        self._start_pose = pose0
        self._final_pose = pose1

    def __call__(self, pose):
        if self._start_pose is None or self._final_pose is None:
            return False

        actor_pos = pose[0:3]
        actor_rot = pose[3:7]

        frame_pos = self._start_pose[0:3]
        frame_rot = self._start_pose[3:7]
        grasp_pos = self._final_pose[0:3]
        grasp_rot = self._final_pose[3:7]

        theta_frame = quat_to_angle(actor_rot, frame_rot)
        theta_grasp = quat_to_angle(actor_rot, grasp_rot)
        theta = min(abs(theta_frame), abs(theta_grasp))

        a = actor_pos - frame_pos
        b = grasp_pos - frame_pos
        l2 = np.linalg.norm(grasp_pos - frame_pos, ord=2)
        proj = np.dot(a, b) / l2
        dist_line = max(min(proj, 1.0), 0.0) / l2
        proj_pt = frame_pos + dist_line * (grasp_pos - frame_pos)
        dist = np.linalg.norm(proj_pt - actor_pos)
        pos_tol = min(self._pos_tol + (self._pos_tol * self._slope * l2), self._max_pos_tol)
        ok = dist < pos_tol and theta < self._theta_tol

        return ok


def quat_to_angle(q1, q2):
    res = 2 * (np.sum(q1 * q2)) ** 2 - 1
    res = np.clip(res, -1.0, +1.0)
    return np.arccos(res)


class AtPoseCondition:
    def __init__(self, position_tol, rotation_tol):
        self._position_tol = position_tol
        self._rotation_tol = rotation_tol

        self._goal_pose = None

    def set_goal(self, goal_pose):
        self._goal_pose = goal_pose

    def __call__(self, pose):
        actor_pos = pose[0:3]
        actor_rot = pose[3:7]

        grasp_pos = self._goal_pose[0:3]
        grasp_rot = self._goal_pose[3:7]

        dist = np.linalg.norm(grasp_pos - actor_pos)
        theta_grasp = quat_to_angle(actor_rot, grasp_rot)

        return dist < self._position_tol and theta_grasp < self._rotation_tol


class BulletPanda:
    def __init__(self, urdf_file, base_pos, base_orn):
        self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._id = self._p.loadURDF(urdf_file)
        self._p.resetBasePositionAndOrientation(self._id, base_pos, base_orn)

        self._ee_idx = 7

    def ik(self, q0, pos, rot=None, tol=1e-3, theta_tol=0.1, max_iter=1000):
        self._set_joint_position(q0)
        for _ in range(max_iter):
            kwargs = {"restPoses": q0}
            if rot is not None:
                kwargs["targetOrientation"] = rot
            conf = self._p.calculateInverseKinematics(self._id, self._ee_idx, pos, **kwargs)
            pos_fk, rot_fk = self._fk(conf)
            dist = np.linalg.norm(pos_fk - pos)
            if dist < tol and (rot is None or quat_to_angle(rot_fk, rot) < theta_tol):
                return conf[: self._ee_idx]
            q0 = conf
        return None

    def _set_joint_position(self, position):
        for i in range(self._ee_idx):
            self._p.resetJointState(self._id, i, position[i])

    def _fk(self, q):
        self._set_joint_position(q)
        state = self._p.getLinkState(self._id, self._ee_idx)
        pos, rot = state[:2]
        return np.array(pos), np.array(rot)


def compose_qq(q1, q2):
    qww = q1[..., 6] * q2[..., 6]
    qxx = q1[..., 3] * q2[..., 3]
    qyy = q1[..., 4] * q2[..., 4]
    qzz = q1[..., 5] * q2[..., 5]

    q1w2x = q1[..., 6] * q2[..., 3]
    q2w1x = q2[..., 6] * q1[..., 3]
    q1y2z = q1[..., 4] * q2[..., 5]
    q2y1z = q2[..., 4] * q1[..., 5]

    q1w2y = q1[..., 6] * q2[..., 4]
    q2w1y = q2[..., 6] * q1[..., 4]
    q1z2x = q1[..., 5] * q2[..., 3]
    q2z1x = q2[..., 5] * q1[..., 3]

    q1w2z = q1[..., 6] * q2[..., 5]
    q2w1z = q2[..., 6] * q1[..., 5]
    q1x2y = q1[..., 3] * q2[..., 4]
    q2x1y = q2[..., 3] * q1[..., 4]

    q3 = np.zeros(np.broadcast_shapes(q1.shape, q2.shape))
    q3[..., 0:3] = compose_qp(q1, q2[..., 0:3])
    q3[..., 3] = q1w2x + q2w1x + q1y2z - q2y1z
    q3[..., 4] = q1w2y + q2w1y + q1z2x - q2z1x
    q3[..., 5] = q1w2z + q2w1z + q1x2y - q2x1y
    q3[..., 6] = qww - qxx - qyy - qzz

    return q3


def compose_qp(q, pt):
    px = pt[..., 0]
    py = pt[..., 1]
    pz = pt[..., 2]

    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    qx = q[..., 3]
    qy = q[..., 4]
    qz = q[..., 5]
    qw = q[..., 6]

    qxx = qx**2
    qyy = qy**2
    qzz = qz**2
    qwx = qw * qx
    qwy = qw * qy
    qwz = qw * qz
    qxy = qx * qy
    qxz = qx * qz
    qyz = qy * qz

    pt2 = np.zeros((*np.broadcast_shapes(q.shape[:-1], pt.shape[:-1]), 3))
    pt2[..., 0] = x + px + 2 * ((-1 * (qyy + qzz) * px) + ((qxy - qwz) * py) + ((qwy + qxz) * pz))
    pt2[..., 1] = y + py + 2 * (((qwz + qxy) * px) + (-1 * (qxx + qzz) * py) + ((qyz - qwx) * pz))
    pt2[..., 2] = z + pz + 2 * (((qxz - qwy) * px) + ((qwx + qyz) * py) + (-1 * (qxx + qyy) * pz))

    return pt2


def simple_extend(q1, q2, step_size=0.1):
    dq = q2 - q1
    dist = np.linalg.norm(dq)
    if dist < step_size:
        return q2
    else:
        q3 = q1.copy()
        q3 += (dq / dist) * step_size
        return q3


class ThesisPolicy:
    def __init__(self, cfg, bullet_manipulator=None, time_wait=time_wait, time_action_repeat=0.1, time_close_gripper=0.5):
        self._cfg = cfg
        self._steps_wait = int(time_wait / self._cfg.SIM.TIME_STEP)
        self._steps_action_repeat = int(time_action_repeat / self._cfg.SIM.TIME_STEP)
        self._steps_close_gripper = int(time_close_gripper / self._cfg.SIM.TIME_STEP)

        self._to_ee_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.7071068, -0.7071068])
        self._standoff_offset = np.array([0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 1.0])

        self._base_wt = 1.0
        self._home_wt = 1.0
        self._q_pref = np.array(
            [0.6914180, -0.4314530, 1.5511079, 0.6429956, 0.6701149, 0.2626269, -0.2617819]
        )
        self._max_opts = 25

        self._metric = WeightedPoseDistance(position_wt=1.0, rotation_wt=0.5)

        self._in_approach_region = ApproachRegionCondition(
            slope=50.0, pos_tol=1.5e-2, max_pos_tol=4e-2, theta_tol=np.radians(15.0)
        )
        self._at_grasp_pose = AtPoseCondition(position_tol=0.005, rotation_tol=np.radians(15.0))

        # self._bullet_panda = bullet_manipulator
        self._bullet_panda = BulletPanda(
            panda_urdf_file, self._cfg.ENV.PANDA_BASE_POSITION, self._cfg.ENV.PANDA_BASE_ORIENTATION
        )

        self._waypoints = None
        self._num_waypoints = 0
        self._current_waypoint_index = 0

    @property
    def name(self):
        return "amy-thesis"

    def reset(self):
        self._done = False
        self._done_frame = None
        self._q_standoff = None
        self._q_grasp = None
        self._current_grasps = None
        self._action_repeat = None
        self._back = None
        self._waypoints = None
        self._num_waypoints = 0
        self._current_waypoint_index = 0

    def set_waypoints(self, waypoints):
        """Set the waypoints for the robot to follow."""
        self._waypoints = waypoints
        self._num_waypoints = len(waypoints)
        self._current_waypoint_index = 0
    
    def forward(self, obs):
        if self._current_grasps is None:
            self._current_grasps = self._load_grasps(obs)

        if obs["frame"] < self._steps_wait:
            action = start_conf.copy()
        else:
            if not self._done:
                if (obs["frame"] - self._steps_wait) % self._steps_action_repeat == 0:
                    current_cfg = self._get_current_cfg(obs)
                    object_pose = self._get_object_pose(obs)
                    ee_pose = self._get_ee_pose(obs)
                    waypoint = self._waypoints[self._current_waypoint_index]
                    action = self._get_action(current_cfg, waypoint)
                    self._action_repeat = action.copy()
                    self._current_waypoint_index += 1
                    if self._current_waypoint_index == self._num_waypoints:
                        self._done = True
                    print("Current waypoint:", waypoint)
                    print("Current action:", action)
                else:
                    action = self._action_repeat.copy()

            if self._done:
                if self._done_frame is None:
                    self._done_frame = obs["frame"]
                if obs["frame"] < self._done_frame + self._steps_close_gripper:
                    current_cfg = self._get_current_cfg(obs)
                    action = current_cfg.copy()
                    action[7:9] = 0.0
                else:
                    if self._back is None:
                        current_cfg = self._get_current_cfg(obs)
                        self._back = self._get_back(current_cfg)
                    action = self._back.copy()
        # print("======== FORWARD ==========")
        # print(action)
        return action, {}

    def _load_grasps(self, obs):
        class_name = obs["ycb_classes"][list(obs["ycb_bodies"])[0]]

        grasp_file = os.path.join(grasp_dir, "{}.npy".format(class_name))
        data = np.load(grasp_file, allow_pickle=True, encoding="bytes")
        grasps = data.item()[b"transforms"]

        grasps_pq = np.zeros((len(grasps), 7))
        grasps_pq[:, 0:3] = grasps[:, :3, 3]
        grasps_pq[:, 3:7] = Rot.from_matrix(grasps[:, :3, :3]).as_quat()

        return grasps_pq

    def _get_current_cfg(self, obs):
        return obs["panda_body"].dof_state[0, :, 0].numpy()

    def _get_object_pose(self, obs):
        return obs["ycb_bodies"][list(obs["ycb_bodies"])[0]].link_state[0, 6, 0:7].numpy()

    def _get_ee_pose(self, obs):
        return obs["panda_body"].link_state[0, obs["panda_link_ind_hand"], 0:7].numpy()

    def _get_action(self, current_cfg, waypoint):

        action = current_cfg.copy()
        action[0:7] = self._compute_ik(waypoint, current_cfg)
        action[7:9] = 0.04

        return action

    def _compute_ik(self, pose, cfg):
        pos = pose[0:3]
        rot = pose[3:7]
        print("Compute ik:", pos, rot)
        return self._bullet_panda.ik(cfg, pos, rot=rot)

    def _get_back(self, current_cfg):
        pos = self._cfg.BENCHMARK.GOAL_CENTER
        conf = self._bullet_panda.ik(current_cfg, pos)
        back = current_cfg.copy()
        back[0:7] = conf
        back[7:9] = 0.0
        return back



class BulletManipulator:
    def __init__(
        self,
        hz: int,
        cfg: DictConfig,
        gui: bool,
        base_pos: list,
        base_orn: list,
        gravity: float = 9.81,
    ):
        self.cfg = cfg
        self.dt = 1.0 / hz

        # Initialize PyBullet simulation
        gui = False
        if gui:
            self.sim = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.sim = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

        urdf_path = "/home/chenam14/fairo/polymetis/polymetis/data/franka_panda/panda_arm_hand.urdf"

        self.robot_id = self.sim.loadURDF(
            urdf_path,
            basePosition=base_pos,
            baseOrientation=base_orn,
            useFixedBase=True,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
        )
        # print("ROBOT ID:", self.robot_id)

        self.controlled_joints = [0, 1, 2, 3, 4, 5, 6]
        self.gripper_controlled_joints = [9, 10]
        self.gripper_moving_threshold = 0.002
        self.rest_pose = [-0.13935425877571106, -0.020481698215007782, -0.05201413854956627, -2.0691256523132324, 0.05058913677930832, 2.0028650760650635, -0.9167874455451965]

        for i in range(7):
            self.sim.resetJointState(
                bodyUniqueId=self.robot_id,
                jointIndex=self.controlled_joints[i],
                targetValue=self.rest_pose[i],
                targetVelocity=0,
            )

        # Initialize states
        self.arm_state = polymetis_pb2.RobotState()
        self.arm_state.prev_joint_torques_computed[:] = np.zeros(7)
        self.arm_state.prev_joint_torques_computed_safened[:] = np.zeros(7)
        self.arm_state.motor_torques_measured[:] = np.zeros(7)
        self.arm_state.motor_torques_external[:] = np.zeros(7)

        self.arm_state.prev_command_successful = True
        self.arm_state.error_code = 0

        self.gripper_state = polymetis_pb2.GripperState()

        self.t = 0

    def get_arm_state(self) -> polymetis_pb2.RobotState:
        # Timestamp
        self.arm_state.timestamp.GetCurrentTime()

        # Joint pos & vel
        joint_cur_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints
        )
        self.arm_state.joint_positions[:] = [joint_cur_states[i][0] for i in range(7)]
        self.arm_state.joint_velocities[:] = [joint_cur_states[i][1] for i in range(7)]

        return self.arm_state

    def get_gripper_state(self) -> polymetis_pb2.GripperState:
        # Timestamp
        self.gripper_state.timestamp.GetCurrentTime()

        # Gripper states
        joint_cur_states = self.sim.getJointStates(
            self.robot_id, self.gripper_controlled_joints
        )
        self.gripper_state.width = float(
            joint_cur_states[0][0] + joint_cur_states[1][0]
        )
        self.gripper_state.is_grasped = False  # TODO
        self.gripper_state.is_moving = np.all(
            [
                abs(joint_cur_states[i][1]) < self.gripper_moving_threshold
                for i in range(2)
            ]
        )

        return self.gripper_state

    def apply_arm_control(self, cmd: polymetis_pb2.TorqueCommand):
        # Extract torques
        commanded_torques = np.array(list(cmd.joint_torques))

        # Compute grav comp
        joint_pos = list(self.arm_state.joint_positions)
        finger_pos = [self.gripper_state.width / 2.0] * 2
        grav_comp_torques = self.sim.calculateInverseDynamics(
            self.robot_id,
            joint_pos + finger_pos,
            [0] * 9,
            [0] * 9,
        )[:7]

        # Set sim torques
        applied_torques = commanded_torques + grav_comp_torques
        self.sim.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.controlled_joints,
            controlMode=pybullet.TORQUE_CONTROL,
            forces=applied_torques,
        )

        # Populate torques in state
        self.arm_state.prev_joint_torques_computed[:] = commanded_torques
        self.arm_state.prev_joint_torques_computed_safened[:] = commanded_torques
        self.arm_state.motor_torques_measured[:] = applied_torques
        self.arm_state.motor_torques_external[:] = np.zeros_like(applied_torques)

        self.arm_state.prev_command_successful = True

    def apply_gripper_control(self, cmd: polymetis_pb2.GripperCommand):
        self.sim.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=self.gripper_controlled_joints,
            controlMode=pybullet.POSITION_CONTROL,
            targetPositions=[cmd.width / 2.0] * 2,
        )

    def step(self):
        self.sim.stepSimulation()
        self.t += self.dt

def main():
    # Handover config
    handover_cfg = get_config_from_args()
    handover_cfg.BENCHMARK.SETUP = "s0"
    handover_cfg.SIM.RENDER = True
    handover_cfg.ENV.RENDER_OFFSCREEN = True
    # handover_cfg.BENCHMARK.SAVE_RESULT = True
    # handover_cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER = True

    policy = ThesisPolicy(handover_cfg)

    benchmark_runner = BenchmarkRunner(handover_cfg)
    benchmark_runner.run(policy)
    

if __name__ == "__main__":
    main()
