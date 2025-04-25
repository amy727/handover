# Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the BSD 3-Clause License [see LICENSE for details].

import easysim
import abc
import numpy as np

from handover.table import Table
from handover.panda import Panda, PandaHandCamera, PandaTcp
from handover.dex_ycb import DexYCB
from handover.ycb import YCB
from handover.mano import MANO
from handover.transform3d import *

import open3d as o3d

from scipy.spatial.transform import Rotation as R

class HandoverEnv(easysim.SimulatorEnv):
    def init(self):
        self._table = Table(self.cfg, self.scene)

        self._panda = self._get_panda_cls()(self.cfg, self.scene)

        self._dex_ycb = DexYCB(self.cfg)
        self._ycb = YCB(self.cfg, self.scene, self.dex_ycb)
        # print("YCB SELF:", self._ycb.name)
        self._mano = MANO(self.cfg, self.scene, self.dex_ycb)

        self._release_step_thresh = self.cfg.ENV.RELEASE_TIME_THRESH / self.cfg.SIM.TIME_STEP

        if self.cfg.ENV.DRAW_RELEASE_CONTACT:
            self._draw_release_contact_init()

        if self.cfg.ENV.RENDER_OFFSCREEN:
            self._render_offscreen_init()

    def set_workspace_bounds(self, workspace_bounds_min, workspace_bounds_max):
        self.workspace_bounds_min = workspace_bounds_min
        self.workspace_bounds_max = workspace_bounds_max

    def set_up_objects(self):
        """
        Setup a mapping between the names of all objects and their ids so that objects
        can be easily queried later on.
        Also saves the names of the hand and object that is handed over.  
        """
        self.name2ids = {}
        self.ycb_name = "ycb"
        self.hand_name = "subject"
        for name, obj in self.scene._name_to_body.items():
            self.name2ids[name] = self.scene._bodies.index(obj) + 1
            print(name, obj, self.scene.ycb_obj)
            if obj == self.scene.ycb_obj:
                self.ycb_name = name
            elif "subject" in name:
                self.hand_name = name
        print(f"name2ids {self.name2ids}")

    def get_object_names(self):
        obj_names = list(self.scene._name_to_body.keys())
        print("OBJ", self.mano.body)
        print(f"obj names: {obj_names}")
        return obj_names

    @abc.abstractmethod
    def _get_panda_cls(self):
        """ """

    @property
    def table(self):
        return self._table

    @property
    def panda(self):
        return self._panda

    @property
    def dex_ycb(self):
        return self._dex_ycb

    @property
    def ycb(self):
        return self._ycb

    @property
    def mano(self):
        return self._mano

    def _render_offscreen_init(self):
        camera = easysim.Camera()
        camera.name = "offscreen_renderer"
        camera.width = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_WIDTH
        camera.height = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_HEIGHT
        camera.vertical_fov = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_VERTICAL_FOV
        camera.near = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_NEAR
        camera.far = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_FAR
        camera.position = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_POSITION
        camera.target = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_TARGET
        camera.up_vector = (0.0, 0.0, 1.0)
        self.scene.add_camera(camera)
        self._camera = camera

        camera = easysim.Camera()
        camera.name = "cam2"
        camera.width = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_WIDTH
        camera.height = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_HEIGHT
        camera.vertical_fov = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_VERTICAL_FOV
        camera.near = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_NEAR
        camera.far = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_FAR
        camera.position = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_POSITION2
        camera.target = self.cfg.ENV.OFFSCREEN_RENDERER_CAMERA_TARGET2
        camera.up_vector = (0.0, 0.0, 1.0)
        self.scene.add_camera(camera)

    def pre_reset(self, env_ids, scene_id):
        self.ycb.reset(scene_id)
        self.mano.reset(scene_id)

        if self.cfg.ENV.DRAW_RELEASE_CONTACT:
            self._draw_release_contact_reset()

    def post_reset(self, env_ids, scene_id):
        self._release_reset()
        self._frame = 0
        return self._get_observation()

    def pre_step(self, action):
        self.panda.step(action)
        self.ycb.step()
        self.mano.step(self.simulator)

        if self.cfg.ENV.DRAW_RELEASE_CONTACT:
            self._draw_release_contact_step()

    def post_step(self, action):
        if not self.ycb.released and self._release_check():
            self.ycb.release()

        self._frame += 1

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        return observation, reward, done, info

    @property
    def frame(self):
        return self._frame

    @abc.abstractmethod
    def _get_observation(self):
        """ """

    @abc.abstractmethod
    def _get_reward(self):
        """ """

    @abc.abstractmethod
    def _get_done(self):
        """ """

    @abc.abstractmethod
    def _get_info(self):
        """ """

    def _release_reset(self):
        self._release_step_counter_passive = 0
        self._release_step_counter_active = 0

    def _release_check(self):
        contact = self.contact[0]

        contact = contact[contact["force"] > self.cfg.ENV.RELEASE_FORCE_THRESH]

        if len(contact) == 0:
            contact_panda_release_region = [False] * len(self.panda.LINK_IND_FINGERS)
            contact_panda_body = False
        else:
            contact_1 = contact[
                (contact["body_id_a"] == self.ycb.bodies[self.ycb.ids[0]].contact_id[0])
                & (contact["body_id_b"] == self.panda.body.contact_id[0])
            ]
            contact_2 = contact[
                (contact["body_id_a"] == self.panda.body.contact_id[0])
                & (contact["body_id_b"] == self.ycb.bodies[self.ycb.ids[0]].contact_id[0])
            ]
            contact_2[["body_id_a", "body_id_b"]] = contact_2[["body_id_b", "body_id_a"]]
            contact_2[["link_id_a", "link_id_b"]] = contact_2[["link_id_b", "link_id_a"]]
            contact_2[["position_a_world", "position_b_world"]] = contact_2[
                ["position_b_world", "position_a_world"]
            ]
            contact_2[["position_a_link", "position_b_link"]] = contact_2[
                ["position_b_link", "position_a_link"]
            ]
            contact_2["normal"]["x"] *= -1
            contact_2["normal"]["y"] *= -1
            contact_2["normal"]["z"] *= -1
            contact = np.concatenate((contact_1, contact_2))

            contact_panda_body = len(contact) > 0
            contact_panda_release_region = []

            for link_index in self.panda.LINK_IND_FINGERS:
                contact_link = contact[contact["link_id_b"] == link_index]

                if len(contact_link) == 0:
                    contact_panda_release_region.append(False)
                else:
                    if np.any(np.isnan(contact_link["position_b_link"]["x"])):
                        pos = self.panda.body.link_state[0, link_index, 0:3]
                        orn = self.panda.body.link_state[0, link_index, 3:7]
                        t3d = get_t3d_from_qt(orn, pos)
                        t3d = t3d.inverse()
                        position = (
                            np.ascontiguousarray(contact_link["position_b_world"])
                            .view(np.float32)
                            .reshape(-1, 3)
                        )
                        position = t3d.transform_points(position)
                    else:
                        position = (
                            np.ascontiguousarray(contact_link["position_b_link"])
                            .view(np.float32)
                            .reshape(-1, 3)
                        )

                    is_in_release_region = (
                        (position[:, 0] > self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X[0])
                        & (position[:, 0] < self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X[1])
                        & (position[:, 1] > self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y[0])
                        & (position[:, 1] < self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y[1])
                        & (position[:, 2] > self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z[0])
                        & (position[:, 2] < self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z[1])
                    )
                    contact_panda_release_region.append(np.any(is_in_release_region))

        if not any(contact_panda_release_region) and contact_panda_body:
            self._release_step_counter_passive += 1
        else:
            if self._release_step_counter_passive != 0:
                self._release_step_counter_passive = 0

        if all(contact_panda_release_region):
            self._release_step_counter_active += 1
        else:
            if self._release_step_counter_active != 0:
                self._release_step_counter_active = 0

        return (
            self._release_step_counter_passive >= self._release_step_thresh
            or self._release_step_counter_active >= self._release_step_thresh
        )

    def _draw_release_contact_init(self):
        self._release_contact_center = np.array(
            [
                [
                    sum(self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X) / 2,
                    sum(self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y) / 2,
                    sum(self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z) / 2,
                ]
            ]
        )
        vertices = []
        for x in self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X:
            for y in self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y:
                for z in self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z:
                    vertices.append([x, y, z])
        self._release_contact_vertices = np.array(vertices)

        release_contact_half_extents = [
            (
                self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X[1]
                - self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_X[0]
            )
            / 2,
            (
                self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y[1]
                - self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Y[0]
            )
            / 2,
            (
                self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z[1]
                - self.cfg.ENV.RELEASE_CONTACT_REGION_RANGE_Z[0]
            )
            / 2,
        ]

        self._release_contact_bodies = {}

        for link_index in self.panda.LINK_IND_FINGERS:
            body = easysim.Body()
            body.name = "region_{:02d}".format(link_index)
            body.geometry_type = easysim.GeometryType.BOX
            body.box_half_extent = release_contact_half_extents
            body.initial_base_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            body.link_color = [self.cfg.ENV.RELEASE_CONTACT_REGION_COLOR]
            body.link_collision_filter = [0]
            self.scene.add_body(body)
            self._release_contact_bodies[body.name] = body

            for i in range(len(self._release_contact_vertices)):
                body = easysim.Body()
                body.name = "vertex_{:02d}_{:d}".format(link_index, i)
                body.geometry_type = easysim.GeometryType.SPHERE
                body.sphere_radius = self.cfg.ENV.RELEASE_CONTACT_VERTEX_RADIUS
                body.initial_base_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                body.link_color = [self.cfg.ENV.RELEASE_CONTACT_VERTEX_COLOR]
                body.link_collision_filter = [0]
                self.scene.add_body(body)
                self._release_contact_bodies[body.name] = body

    def _draw_release_contact_reset(self):
        for link_index in self.panda.LINK_IND_FINGERS:
            self._release_contact_bodies[
                "region_{:02d}".format(link_index)
            ].initial_base_position = (
                self.cfg.ENV.PANDA_BASE_POSITION + self.cfg.ENV.PANDA_BASE_ORIENTATION
            )
            self._release_contact_bodies[
                "region_{:02d}".format(link_index)
            ].env_ids_reset_base_state = [0]

            for i in range(len(self._release_contact_vertices)):
                self._release_contact_bodies[
                    "vertex_{:02d}_{:d}".format(link_index, i)
                ].initial_base_position = (
                    self.cfg.ENV.PANDA_BASE_POSITION + self.cfg.ENV.PANDA_BASE_ORIENTATION
                )
                self._release_contact_bodies[
                    "vertex_{:02d}_{:d}".format(link_index, i)
                ].env_ids_reset_base_state = [0]

    def _draw_release_contact_step(self):
        for link_index in self.panda.LINK_IND_FINGERS:
            pos = self.panda.body.link_state[0, link_index, 0:3]
            orn = self.panda.body.link_state[0, link_index, 3:7]
            t3d = get_t3d_from_qt(orn, pos)
            center = t3d.transform_points(self._release_contact_center)[0]
            vertices = t3d.transform_points(self._release_contact_vertices)

            self._release_contact_bodies[
                "region_{:02d}".format(link_index)
            ].initial_base_position = (center.tolist() + orn.tolist())
            self._release_contact_bodies[
                "region_{:02d}".format(link_index)
            ].env_ids_reset_base_state = [0]

            for i in range(len(self._release_contact_vertices)):
                self._release_contact_bodies[
                    "vertex_{:02d}_{:d}".format(link_index, i)
                ].initial_base_position = vertices[i].tolist() + [0.0, 0.0, 0.0, 1.0]
                self._release_contact_bodies[
                    "vertex_{:02d}_{:d}".format(link_index, i)
                ].env_ids_reset_base_state = [0]

    def reset_to_default_pose(self):
        #TODO
        return self.get_ee_pose()

    def get_ee_pose(self):
        ############################# OLD
        ee_pose = self.panda._body.link_state[0, 11, :].numpy()
        
        ############################# NEW

        # Try to get middle of fingers
        # LINK_IND_FINGERS = (9, 10)
        # Get the link states of the fingers
        # finger_states = self.panda._body.link_state[0, self.panda.LINK_IND_FINGERS, :].numpy()
        # # Print both the finger states and the ee pose

        # # Average the positions (0:3)
        # avg_position = np.mean(finger_states[:, 0:3], axis=0)

        # # Use the quaternion of one finger (e.g., the left finger, index 0)
        # orientation_quat = finger_states[0, 3:7]

        # # Combine position and quaternion
        # ee_pose = np.concatenate([avg_position, orientation_quat])
        
        return ee_pose

    def get_ee_pos(self):
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        return self.get_ee_pose()[3:7]
    

    def set_camera_matrices(self, view_matrix, projection_matrix):
        """
        Set view and projection matrix for all the cameras
        If there are multiple cameras, it would be a dictionary with the camera name as the key
        and the view/projection matrix as the value.
        """
        self.view_matrix = view_matrix
        self.proj_matrix = projection_matrix

    def get_point_cloud(self, camera, depth_frame, color_frame=None, seg_frame=None):
        view_matrix = np.array(self.view_matrix[camera.name]).reshape([4, 4], order="F")
        proj_matrix = np.array(self.proj_matrix[camera.name]).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        if depth_frame is not None:
            depth_frame = self.meters_to_normalized_depth(depth_frame, camera.near, camera.far)
            depth_frame = np.array(depth_frame, copy=True)
            height, width = depth_frame.shape

            # Create pixel grid
            y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
            y *= -1.0
            x = x.reshape(-1)
            y = y.reshape(-1)
            z = depth_frame.reshape(-1)
            h = np.ones_like(z)

            # Filter out invalid z (set to -1.0 earlier)
            valid = (z >= 0.0) & (z < 0.99)
            x, y, z, h = x[valid], y[valid], z[valid], h[valid]

            # Map normalized z from [0,1] to [-1,1] clip space
            z = 2.0 * z - 1.0

            pixels = np.stack([x, y, z, h], axis=1)

            # Unproject
            points = np.matmul(tran_pix_world, pixels.T).T
            points /= points[:, 3:4]
            points = points[:, :3]

            # Optional: color + seg
            colors = None
            if color_frame is not None:
                color_frame = color_frame[:, :, :3].reshape(-1, 3)
                colors = color_frame[valid]

            seg = None
            if seg_frame is not None:
                seg_frame = seg_frame.reshape(-1)
                seg = seg_frame[valid]

            return points, colors, seg

    # def get_point_cloud(self, camera, depth_frame, color_frame=None, seg_frame=None):
    #     """Generates a point cloud from depth, color, and segmentation frames."""
    #     view_matrix = np.array(self.view_matrix[camera.name]).reshape([4, 4], order="F")
    #     proj_matrix = np.array(self.proj_matrix[camera.name]).reshape([4, 4], order="F")
    #     tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    #     if depth_frame is not None:
    #         depth_frame = np.array(depth_frame, copy=True)
    #         height, width = depth_frame.shape
    #         # print("Depth shape:", width, height)

    #         # create a grid with pixel coordinates and depth values
    #         y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    #         y *= -1.0
    #         x, y, z = x.reshape(-1), y.reshape(-1), depth_frame.reshape(-1)
    #         h = np.ones_like(z)

    #         pixels = np.stack([x, y, z, h], axis=1)
    #         # filter out "infinite" depths
    #         # pixels = pixels[z < 0.99]
    #         pixels[:, 2] = 2 * pixels[:, 2] - 1

    #         # turn pixels to world coordinates
    #         points = np.matmul(tran_pix_world, pixels.T).T
    #         points /= points[:, 3:4]
    #         points = points[:, :3]

    #         # Add colors to the point cloud
    #         colors = None
    #         if color_frame is not None:
    #             colors = color_frame[:, :, :3].reshape(-1, 3)
    #             # print(f"Color shape: {colors.shape}")

    #         seg = None
    #         if seg_frame is not None:
    #             seg = seg_frame.reshape(-1)
    #             # print(f"Mask shape: {seg.shape}")

    #         # print(f"Points shape: {points.shape}")
    #         return points, colors, seg
    
    def meters_to_normalized_depth(self, depth_meters, near=0.1, far=1.1):
        depth_meters = np.copy(depth_meters)
        invalid_mask = (depth_meters <= 1e-3) | np.isnan(depth_meters)
        depth_meters[invalid_mask] = np.nan

        z_n = (2.0 * near * far / depth_meters - (far + near)) / (far - near)
        z_norm = 0.5 * (-z_n + 1.0)

        # Return NaNs clearly so they can be removed
        return z_norm

        
    def get_3d_obs_by_name(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.
        This is called by voxposer.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        # self.name2ids = {}
        # for name, obj in self.scene._name_to_body.items():
        #     self.name2ids[name] = self.scene._bodies.index(obj) + 1
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        
        # gather points and masks from all cameras
        points, masks, normals = [], [], []
        for cam in self.scene._cameras:
            depth = cam.depth[0].numpy()
            #depth = self.meters_to_normalized_depth(depth, cam.near, cam.far)

            seg = cam.segmentation[0].numpy()

            # Generate point cloud
            pc, _, s = self.get_point_cloud(cam, depth, seg_frame=seg)
            points.append(pc)
            masks.append(s)
            
            # estimate normals using o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[-1])
            pcd.estimate_normals()
            cam_normals = np.asarray(pcd.normals)
            # use lookat/target vector to adjust normal vectors
            flip_indices = np.dot(cam_normals, cam.target) > 0
            cam_normals[flip_indices] *= -1
            normals.append(cam_normals)
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        normals = np.concatenate(normals, axis=0)
        
        # print("Get object points")
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")
        obj_normals = normals[np.isin(masks, obj_ids)]
        # print("Object downsample")
        
        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points)
        pcd.normals = o3d.utility.Vector3dVector(obj_normals)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.0005)
        obj_points = np.asarray(pcd_downsampled.points)
        obj_normals = np.asarray(pcd_downsampled.normals)
        # print("Done")
        return obj_points, obj_normals

    def get_obj_points(self, query_name):
        """
        Retrieves 3D point cloud observations and normals of an object by its name.
        This is called by voxposer.

        Args:
            query_name (str): The name of the object to query.

        Returns:
            tuple: A tuple containing object points and object normals.
        """
        assert query_name in self.name2ids, f"Unknown object name: {query_name}"
        obj_ids = self.name2ids[query_name]
        
        # gather points and masks from all cameras
        points, masks = [], []
        for cam in self.scene._cameras:
            depth = cam.depth[0].numpy()
            #depth = self.meters_to_normalized_depth(depth, cam.near, cam.far)

            seg = cam.segmentation[0].numpy()

            # Generate point cloud
            pc, _, s = self.get_point_cloud(cam, depth, seg_frame=seg)
            points.append(pc)
            masks.append(s)
            
        points = np.concatenate(points, axis=0)
        masks = np.concatenate(masks, axis=0)
        
        # print("Get object points")
        # get object points
        obj_points = points[np.isin(masks, obj_ids)]
        if len(obj_points) == 0:
            raise ValueError(f"Object {query_name} not found in the scene")

        return obj_points

    def calculate_distances(self, waypoint, object_points):
        """
        Calculate the Euclidean distances from a single waypoint to all the points in an object.
        Can be used to help with calculations of minimum distances of the waypoint to the hand
        to evaluate safety.

        Parameters:
        - waypoint (array-like): The 3D coordinates of the waypoint, format: [x, y, z].
        - object_points (array-like): An Nx3 array containing the 3D coordinates of N object points.

        Returns:
        - distances (numpy.ndarray): An array of distances from the waypoint to each object point.
        """
        # Convert the inputs to NumPy arrays to ensure compatibility with NumPy operations
        waypoint = np.array(waypoint)
        object_points = np.array(object_points)

        # Calculate the differences in each dimension
        differences = object_points - waypoint

        # Calculate the squared differences
        squared_differences = differences ** 2

        # Sum the squared differences across the columns (i.e., sum over x, y, z differences)
        sum_squared_differences = np.sum(squared_differences, axis=1)

        # Take the square root of the sum of squared differences to get the Euclidean distances
        distances = np.sqrt(sum_squared_differences)

        return np.min(distances)

    def get_scene_3d_obs(self, ignore_robot=False, ignore_grasped_obj=False):
        """
        Retrieves the entire scene's 3D point cloud observations and colors.

        Args:
            ignore_robot (bool): Whether to ignore points corresponding to the robot.
            ignore_grasped_obj (bool): Whether to ignore points corresponding to grasped objects.

        Returns:
            tuple: A tuple containing scene points and colors.
        """
        points, colors, masks = [], [], []
        for cam in self.scene._cameras:
            color = cam.color[0].numpy()
            depth = cam.depth[0].numpy()
            #depth = self.meters_to_normalized_depth(depth, cam.near, cam.far)
            seg = cam.segmentation[0].numpy()

            pc, c, s = self.get_point_cloud(cam, depth, color, seg)
            points.append(pc)
            colors.append(c)
            masks.append(s)
        points = np.concatenate(points, axis=0)
        colors = np.concatenate(colors, axis=0)
        masks = np.concatenate(masks, axis=0)

        # # only keep points within workspace
        chosen_idx_x = (points[:, 0] > self.workspace_bounds_min[0]) & (points[:, 0] < self.workspace_bounds_max[0])
        chosen_idx_y = (points[:, 1] > self.workspace_bounds_min[1]) & (points[:, 1] < self.workspace_bounds_max[1])
        chosen_idx_z = (points[:, 2] > self.workspace_bounds_min[2]) & (points[:, 2] < self.workspace_bounds_max[2])
        points = points[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        colors = colors[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]
        masks = masks[(chosen_idx_x & chosen_idx_y & chosen_idx_z)]

        # if ignore_robot:
        #     robot_mask = np.isin(masks, self.robot_mask_ids)
        #     points = points[~robot_mask]
        #     colors = colors[~robot_mask]
        #     masks = masks[~robot_mask]
        # if self.grasped_obj_ids and ignore_grasped_obj:
        #     grasped_mask = np.isin(masks, self.grasped_obj_ids)
        #     points = points[~grasped_mask]
        #     colors = colors[~grasped_mask]
        #     masks = masks[~grasped_mask]

        # voxel downsample using o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size=0.0005)
        points = np.asarray(pcd_downsampled.points).astype(np.float16)
        colors = np.asarray(pcd_downsampled.colors).astype(np.uint8)

        return points, colors

    def render_offscreen(self):
        if not self.cfg.ENV.RENDER_OFFSCREEN:
            raise ValueError(
                "`render_offscreen()` can only be called when RENDER_OFFSCREEN is set to True"
            )
        data = {}
        data["color"] = self._camera.color[0].numpy()
        data["depth"] = self._camera.depth[0].numpy()
        data["segmentation"] = self._camera.segmentation[0].numpy()

        return data

    def callback_get_reward_post_status(self, reward, status):
        """ """


class HandoverStateEnv(HandoverEnv):
    def _get_panda_cls(self):
        return PandaTcp

    def _get_observation(self):
        observation = {}
        observation["frame"] = self.frame
        observation["panda_link_ind_hand"] = self.panda.LINK_IND_HAND
        observation["panda_body"] = self.panda.body
        observation["ycb_classes"] = self.ycb.CLASSES
        observation["ycb_bodies"] = self.ycb.bodies
        observation["mano_body"] = self.mano.body
        return observation

    def _get_reward(self):
        return None

    def _get_done(self):
        return False

    def _get_info(self):
        return {}


class HandoverHandCameraPointStateEnv(HandoverEnv):
    def _get_panda_cls(self):
        return PandaHandCamera

    def post_reset(self, env_ids, scene_id):
        self._point_states = None

        return super().post_reset(env_ids, scene_id)

    def post_step(self, action):
        self._point_states = None

        return super().post_step(action)

    def _get_observation(self):
        observation = {}
        observation["frame"] = self.frame
        observation["panda_link_ind_hand"] = self.panda.LINK_IND_HAND
        observation["panda_body"] = self.panda.body
        observation["callback_get_point_states"] = self._get_point_states
        return observation

    def _get_reward(self):
        return None

    def _get_done(self):
        return False

    def _get_info(self):
        return {}

    def _get_point_states(self):
        if self._point_states is None:
            segmentation_ids = []
            segmentation_ids.append(self.ycb.bodies[self.ycb.ids[0]].contact_id[0])
            if (
                self.cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV.COMPUTE_MANO_POINT_STATE
                and self.mano.body is not None
            ):
                segmentation_ids.append(self.mano.body.contact_id[0])

            self._point_states = self.panda.get_point_states(segmentation_ids)

            if (
                self.cfg.ENV.HANDOVER_HAND_CAMERA_POINT_STATE_ENV.COMPUTE_MANO_POINT_STATE
                and self.mano.body is None
            ):
                self._point_states.append(np.zeros((0, 3), dtype=np.float32))

        return self._point_states