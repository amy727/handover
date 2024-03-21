# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import gym
import os
import functools
import time
import cv2
import numpy as np
import torch

from datetime import datetime

from handover.benchmark_wrapper import EpisodeStatus, HandoverBenchmarkWrapper

import openai
from voxposer.arguments import get_config
from voxposer.interfaces import setup_LMP
from voxposer.visualizers import ValueMapVisualizer
from voxposer.utils import set_lmp_objects


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


class BenchmarkRunner:
    def __init__(self, cfg, sim_client=None, robot=None, gripper=None):
        self._cfg = cfg

        self._env = HandoverBenchmarkWrapper(gym.make(self._cfg.ENV.ID, cfg=self._cfg))

        self.sim_client = sim_client
        self.robot = robot
        self.gripper = gripper

        self.config = get_config('handover')
        self.lmps, self.lmp_env = setup_LMP(self._env, self.config, debug=False)
        self.voxposer_ui = self.lmps['plan_ui']

        # Hardcode workspace bounds for now
        self.workspace_bounds_min = np.array([-20,-20,-20])
        self.workspace_bounds_max = np.array([20,20,20])

        # Setup visualizer and add it to the simulation env
        self.visualizer = ValueMapVisualizer(self.config['visualizer'])
        if self.visualizer is not None:
            self.visualizer.update_bounds(self.workspace_bounds_min, self.workspace_bounds_max)
        self._env.visualizer = self.visualizer
        self.voxposer_ui = self.lmps['plan_ui']

    def run(self, policy, res_dir=None, index=None):
        if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
            if not self._cfg.ENV.RENDER_OFFSCREEN:
                raise ValueError(
                    "SAVE_OFFSCREEN_RENDER can only be True when RENDER_OFFSCREEN is set to True"
                )
            if not self._cfg.BENCHMARK.SAVE_RESULT and res_dir is None:
                raise ValueError(
                    "SAVE_OFFSCREEN_RENDER can only be True when SAVE_RESULT is set to True or "
                    "`res_dir` is not None"
                )
            if 1.0 / self._cfg.BENCHMARK.OFFSCREEN_RENDER_FRAME_RATE < self._cfg.SIM.TIME_STEP:
                raise ValueError("Offscreen render time step must not be smaller than TIME_STEP")

            self._render_steps = (
                1.0 / self._cfg.BENCHMARK.OFFSCREEN_RENDER_FRAME_RATE / self._cfg.SIM.TIME_STEP
            )

        if self._cfg.BENCHMARK.SAVE_RESULT:
            if res_dir is not None:
                raise ValueError("SAVE_RESULT can only be True when `res_dir` is None")

            dt = datetime.now()
            dt = dt.strftime("%Y-%m-%d_%H-%M-%S")
            # res_dir = os.path.join(
            #     self._cfg.BENCHMARK.RESULT_DIR,
            #     "{}_{}_{}_{}".format(
            #         dt, policy.name, self._cfg.BENCHMARK.SETUP, self._cfg.BENCHMARK.SPLIT
            #     ),
            # )
            res_dir = os.path.join(
                self._cfg.BENCHMARK.RESULT_DIR,
                "{}_{}_{}_{}".format(
                    dt, "thesis", self._cfg.BENCHMARK.SETUP, self._cfg.BENCHMARK.SPLIT
                ),
            )
            os.makedirs(res_dir, exist_ok=True)

            cfg_file = os.path.join(res_dir, "config.yaml")
            with open(cfg_file, "w") as f:
                self._cfg.dump(stream=f, default_flow_style=None)

        if index is None:
            indices = range(self._env.num_scenes)
        else:
            indices = [index]

        for idx in indices:
            print(
                "{:04d}/{:04d}: scene {}".format(
                    idx + 1, self._env.num_scenes, self._env.scene_ids[idx]
                )
            )

            kwargs = {}
            # if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
            if True:
                kwargs["render_dir"] = os.path.join(res_dir, "{:03d}".format(idx))
                os.makedirs(kwargs["render_dir"], exist_ok=True)

            result, elapsed_time = self._run_scene(idx, policy, **kwargs)

            print("time:   {:6.2f}".format(elapsed_time))
            print("frame:  {:5d}".format(result["elapsed_frame"]))
            if result["result"] == EpisodeStatus.SUCCESS:
                print("result:  success")
            else:
                failure_1 = (
                    result["result"] & EpisodeStatus.FAILURE_HUMAN_CONTACT
                    == EpisodeStatus.FAILURE_HUMAN_CONTACT
                )
                failure_2 = (
                    result["result"] & EpisodeStatus.FAILURE_OBJECT_DROP
                    == EpisodeStatus.FAILURE_OBJECT_DROP
                )
                failure_3 = (
                    result["result"] & EpisodeStatus.FAILURE_TIMEOUT
                    == EpisodeStatus.FAILURE_TIMEOUT
                )
                print("result:  failure {:d} {:d} {:d}".format(failure_1, failure_2, failure_3))

            if self._cfg.BENCHMARK.SAVE_RESULT:
                res_file = os.path.join(res_dir, "{:03d}.npz".format(idx))
                np.savez_compressed(res_file, **result)

    @timer
    def _run_scene(self, idx, policy, render_dir=None):
        obs = self._env.reset(idx=idx)
        policy.reset()

        result = {}
        result["action"] = []
        result["elapsed_time"] = []

        set_lmp_objects(self.lmps, self._env.get_object_names())  # set the object names to be used by voxposer

        # if self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER:
        if True:
            self._render_offscreen_and_save(render_dir)

        # VOXPOSER INSTRUCTION
        instruction = "Avoid the table and reach for the hand."
        self.voxposer_ui(instruction)
        traj_world = self._env.execute_info[0]['traj_world']
        print("========== TRAJECTORY ===========")
        print(traj_world)

        # for i, waypoints in enumerate(traj_world):
        #     # Command robot to ee xyz position
        #     ee_pos_desired = torch.from_numpy(waypoints[0])
        #     ee_ori_desired = torch.from_numpy(waypoints[1])

        #     print(f"\nMoving ee pos to: {ee_pos_desired} ...\n")
        #     state_log = self.robot.move_to_ee_pose(
        #         position=ee_pos_desired, 
        #         orientation=ee_ori_desired, 
        #         time_to_go=2.0
        #     )

        #     # result["action"].append(action)
        #     # result["elapsed_time"].append(elapsed_time)

        #     # obs, _, _, info = self._env.step(action)
        #     self._env.frame += 1

        #     self._render_offscreen_and_save(render_dir)

        while True:
            (action, info), elapsed_time = self._run_policy(policy, obs)

            if "obs_time" in info:
                elapsed_time -= info["obs_time"]

            result["action"].append(action)
            result["elapsed_time"].append(elapsed_time)

            obs, _, _, info = self._env.step(action)

            if (
                self._cfg.BENCHMARK.SAVE_OFFSCREEN_RENDER
                and (self._env.frame % self._render_steps)
                <= (self._env.frame - 1) % self._render_steps
            ):
                self._render_offscreen_and_save(render_dir)

            if info["status"] != 0:
                break

        result["action"] = np.array(result["action"])
        result["elapsed_time"] = np.array(result["elapsed_time"])
        result["elapsed_frame"] = self._env.frame
        result["result"] = info["status"]       

        return result

    def _save_pointcloud_as_ply(self, filename, pointcloud, colors=None):
        """
        Save a pointcloud to a PLY file.

        Args:
            filename (str): The output filename.
            pointcloud (np.ndarray): Numpy array containing point coordinates of shape (N, 3).
            colors (np.ndarray, optional): Numpy array containing RGB colors of shape (N, 3).
        """
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pointcloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(pointcloud)):
                line = f"{pointcloud[i, 0]} {pointcloud[i, 1]} {pointcloud[i, 2]}"
                if colors is not None:
                    line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
                f.write(line + "\n")

    def _render_offscreen_and_save(self, render_dir):
        render_dir = "/home/chenam14/ws/handover-sim/results/thesis"
        # print("RENDER_DIR", render_dir)
        data = self._env.render_offscreen()
        
        img_render_file = os.path.join(render_dir, "{:06d}_img.jpg".format(self._env.frame))
        # print("img_render_file:", img_render_file)
        cv2.imwrite(img_render_file, data["color"][:, :, [2, 1, 0, 3]])

        depth_render_file = os.path.join(render_dir, "{:06d}_depth.png".format(self._env.frame))
        normalized_depth = cv2.normalize(data["depth"], None, 0, 255, cv2.NORM_MINMAX)
        depth_image = np.uint8(normalized_depth)
        cv2.imwrite(depth_render_file, depth_image)

        # seg_render_file = os.path.join(render_dir, "{:06d}_seg.jpg".format(self._env.frame))
        # cv2.imwrite(seg_render_file, data["segmentation"])
        
        pc_render_file = os.path.join(render_dir, "{:06d}_pc.ply".format(self._env.frame))
        self._save_pointcloud_as_ply(pc_render_file, data["pc"][0], data["pc"][1])

        pc_render_file2 = os.path.join(render_dir, "{:06d}_pc_obj.ply".format(self._env.frame))
        self._save_pointcloud_as_ply(pc_render_file2, data["pc_obj"][0])

    @timer
    def _run_policy(self, policy, obs):
        return policy.forward(obs)
