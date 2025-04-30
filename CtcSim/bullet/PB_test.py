import os
osp = os.path
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
import json
from scipy.spatial.transform import Rotation as R
import torch
import time
import trimesh as tm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import cv2
from manopth.manolayer import ManoLayer
from tqdm import tqdm
import traceback
from multiprocessing import Process, Queue
from itertools import product
import os


# from viz_tool import sim_result_loader
# from ...viz import sim_result_loader
"""
What we need:
1. a abstract class for fundamental pybullet operations
2. a class for robot loading and controlling
3. a class for objects loading and etc.
4. a class for data feeding.
"""

class PybulletBase:
    def __init__(
        self,
        connection_mode: str = "GUI",
    ):
        self.connect = p.DIRECT if connection_mode == "DIRECT" else p.GUI
        # self.pc = bc.BulletClient(connection_mode=self.connect)
        # self.pc.setAdditionalSearchPath(pd.getDataPath())
        # # self.pc.loadURDF("plane.urdf", [0, 0, -0.5], useFixedBase=True)
        # self.pc.setTimeStep(1 / 1000)
        # self.pc.setGravity(0, 0, -9.8)
        # self.pc.setPhysicsEngineParameter(numSolverIterations=200)  # Increase solver accuracy
        # self.pc.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        # self.pc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # # self.pc.setGravity(0, 0, 0) #set gravity to 0 as the optimization is done with the net force = 0 not the net force = gravity
        # self._bodies_idx = {}
        # self.pc.setRealTimeSimulation(False)
        # self.n_steps = 0
        # self.lowest_point = None
        # self.control_joint = None
    
    def _connect_(self) -> None:
        """Connect to the simulation"""
        self.pc = bc.BulletClient(connection_mode=self.connect)
        self.pc.setAdditionalSearchPath(pd.getDataPath())
        # self.pc.loadURDF("plane.urdf", [0, 0, -0.5], useFixedBase=True)
        self.pc.setTimeStep(1 / 1000)
        self.pc.setGravity(0, 0, -9.8)
        self.pc.setPhysicsEngineParameter(numSolverIterations=200)  # Increase solver accuracy
        
        
        camera_distance = 0.6 # Distance from camera to target
        camera_yaw = 0       # Horizontal angle (degrees)
        camera_pitch = -30     # Vertical angle (degrees)
        camera_target = [0, 0, 0]
        
        
        self.pc.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, cameraPitch=camera_pitch, cameraTargetPosition=camera_target)
        self.pc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        self.width, self.height = 1920, 1080

        self.view_matrix = self.pc.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_target,
                    distance=camera_distance,
                    yaw=camera_yaw,
                    pitch=camera_pitch,
                    roll=0,
                    upAxisIndex=2  # Z-axis is up (2 for Z, 1 for Y)
                    )

        fov = 60  # Degrees
        aspect = self.width / self.height
        near, far = 0.1, 100.0
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=aspect,
            nearVal=near,
            farVal=far
)

        
        # self.pc.setGravity(0, 0, 0) #set gravity to 0 as the optimization is done with the net force = 0 not the net force = gravity
        self._bodies_idx = {}
        self.pc.setRealTimeSimulation(False)
        self.n_steps = 0
        self.lowest_point = None
        self.control_joint = None
        

    @property
    def dt(self):
        return self.timeStep * self.n_steps
    
    def step(self) -> None:
        self.pc.stepSimulation()
    
    def close(self) -> None:
        """Close the simulation"""
        if self.pc.isConnected():
            self.pc.disconnect()
    
    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.pc.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)
    
    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as (x, y, z, w).
        """
        orientation = self.pc.getBasePositionAndOrientation(self._bodies_idx[body])[1]
        return np.array(orientation)
    
    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        quaternion = self.get_base_orientation(body)
        if type == "euler":
            rotation = self.pc.getEulerFromQuaternion(quaternion)
            return np.array(rotation)
        elif type == "quaternion":
            return np.array(quaternion)
        else:
            raise ValueError("""type must be "euler" or "quaternion".""")
   
    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.pc.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.pc.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)
    
    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.pc.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.pc.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.pc.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.pc.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)       

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.pc.getJointState(self._bodies_idx[body], joint)[0]
    
    def get_joint_angles(self, body: str, joints: list) -> np.ndarray:
        """Get the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            np.ndarray: The angles.
        """
        return np.array([self.get_joint_angle(body, j) for j in joints])
    
    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The velocity.
        """
        return self.pc.getJointState(self._bodies_idx[body], joint)[1]
    
    def get_joint_velocities(self, body: str, joints: list) -> np.ndarray:
        """Get the velocities of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            np.ndarray: The velocities.
        """
        return np.array([self.get_joint_velocity(body, j) for j in joints])

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if len(orientation) == 3:
            orientation = self.pc.getQuaternionFromEuler(orientation)
        self.pc.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_joint_angles(self, body: str, joints: list, angles: list) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)
    
    def set_joint_angles_dof(self, body: str, angles: list) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        if self.control_joint is None:
            self.get_controllable_joints(body)
        assert len(angles) == len(self.control_joint), "The number of angles must match the number of controllable joints."
        
        for joint, angle in zip(self.control_joint, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)
    
    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.pc.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)
    
    def inverse_kinematics(self, body: str, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        joint_state = self.pc.calculateInverseKinematics(
            bodyIndex=self._bodies_idx[body],
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
        )
        return np.array(joint_state)

    def get_num_joints(self, body: str) -> int:
        """Get the number of joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            int: The number of joints.
        """
        return self.pc.getNumJoints(self._bodies_idx[body])

    def get_controllable_joints(self, body: str) -> list:
        """Get the indices of the controllable joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            list: List of joint indices.
        """
        joints_num = self.get_num_joints(body)
        self.control_joint = []
        for i in range(joints_num):
            joint_info = self.pc.getJointInfo(self._bodies_idx[body], i)
            if joint_info[2] == 0:
                self.control_joint.append(i)
        
    def get_joint_angles_dof(self, body: str) -> np.ndarray:
        """Get the degree of freedom of the joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The degree of freedom.
        """
        if not self.control_joint:
            self.get_controllable_joints(body)

        joint_angle_dof = self.get_joint_angles(body, self.control_joint)
        return joint_angle_dof

    def get_joint_velocities_dof(self, body: str) -> np.ndarray:
        """Get the velocities of the joints of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocities.
        """
        if not self.control_joint:
            self.get_controllable_joints(body)

        joint_velocity_dof = self.get_joint_velocities(body, self.control_joint)
        return joint_velocity_dof
    

    def get_dynamics_matrices(self, body: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the mass matrix, h matrix (which is sum of coriolis and gravity), and gravity vector.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The mass matrix, h matrix, and gravity vector.
        """
        
        joint_positions = self.get_joint_angles_dof(body)
        joint_velocities = self.get_joint_velocities_dof(body)
        zero_vec = [0.0] * len(joint_positions)
        
        M = self.pc.calculateMassMatrix(self._bodies_idx[body], joint_positions)
        G = self.pc.calculateInverseDynamics(self._bodies_idx[body], joint_positions, joint_velocities, zero_vec)
        
        return M, G
    
    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.pc.loadURDF(**kwargs)


        
    def load_obj_as_mesh(self, body_name: str, obj_path: str, position: np.ndarray, orientation: np.ndarray, obj_mass: float=0.0, frictional_coe: float = 0.0) -> None:
        """Load obj file and create mesh/collision shape from it.
            ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createVisualShape.py
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            obj_path (str): Path to the obj file.
        """
        """
        Convert the mesh to vhacd
        """
        
        obj_fn = obj_path.split(os.sep)[-1]
        obj_fn_no_ext = obj_fn.split('.')[0]
        vhacd_path = os.path.join(os.path.dirname(obj_path), obj_fn_no_ext + '_vhacd.obj')
        vhacd_log_path = os.path.join(os.path.dirname(obj_path), obj_fn_no_ext + '_vhacd.log')
        
        if not os.path.exists(vhacd_path):
            p.vhacd(fileNameIn=obj_path, fileNameOut=vhacd_path, fileNameLogging=vhacd_log_path, resolution=50000)
        
        obj_visual_shape_id = self.pc.createVisualShape(
                                                    shapeType=p.GEOM_MESH, 
                                                    fileName=obj_path, 
                                                    rgbaColor=[1, 1, 1, 1],
                                                    meshScale=[1, 1, 1])

        obj_collision_shape_id = self.pc.createCollisionShape(
                                                shapeType = p.GEOM_MESH,
                                                fileName = vhacd_path,
                                                # flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                                # p.GEOM_CONCAVE_INTERNAL_EDGE,
                                                # flags=p.GEOM_FORCE_CONVEX_MESH,
                                                meshScale=[1, 1, 1],
                                                )
        
        self._bodies_idx[body_name] = self.pc.createMultiBody(
                                baseMass=obj_mass,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=obj_collision_shape_id,
                                baseVisualShapeIndex=obj_visual_shape_id,
                                basePosition=position,
                                baseOrientation=orientation,
                                useMaximalCoordinates=True)
        
        self.pc.changeDynamics(self._bodies_idx[body_name], -1, 
                                lateralFriction=frictional_coe, 
                                restitution=0.0,
                                spinningFriction=0.01,
                                rollingFriction=0.01,
                                mass = obj_mass,
                                contactStiffness=5000,
                                contactDamping=200,
                                collisionMargin=0.001,
                                ccdSweptSphereRadius=0.001,
                                contactProcessingThreshold=0.001,
                                )
        
        mesh = tm.load(obj_path, force='mesh', process=False)
        mesh.apply_transform(np.vstack((np.hstack((R.from_quat(orientation).as_matrix(), position.reshape(3,1))), np.array([0, 0, 0, 1]))))
        verts = np.array(mesh.vertices)
        lowest_point_z = np.min(verts[:,2])
        self.lowest_point = lowest_point_z
        
        
        # visual_data = self.pc.getVisualShapeData(self._bodies_idx[body_name])
        
    def load_plane(self) -> None:
        if self.lowest_point is None:
            raise ValueError("Please load an object first.")
        self._bodies_idx["plane"] = self.pc.loadURDF("plane.urdf", [0, 0, self.lowest_point-0.5], useFixedBase=True)
        self.pc.changeDynamics(self._bodies_idx["plane"], -1, lateralFriction=1.0, restitution=0.0)
        
    def set_lateral_friction(self, body: str, link: list, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        for lk in link:
            self.pc.changeDynamics(
                bodyUniqueId=self._bodies_idx[body],
                linkIndex=lk,
                lateralFriction=lateral_friction,
            ) 
        
    def set_mass(self, body: str, link: int, mass: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.pc.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            mass=mass,
        ) 
        
    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.pc.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )

    def set_rolling_friction(self, body: str, link: int, rolling_friction: float) -> None:
        """Set the rolling friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            rolling_friction (float): Rolling friction.
        """
        self.pc.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            rollingFriction=rolling_friction,
        )
        
    def set_friction_all(self, body: str, lateral_friction: float) -> None:
        """Set the lateral friction of all links.

        Args:
            body (str): Body unique name.
            lateral_friction (float): Lateral friction.
        """
        for i in range(-1, self.get_num_joints(body)):
            self.pc.changeDynamics(
                bodyUniqueId=self._bodies_idx[body],
                linkIndex=i,
                lateralFriction=lateral_friction,
                spinningFriction=0.01,
                rollingFriction=0.01,
            )
            
    def remove_body(self, body: str) -> None:
        """Remove the body.

        Args:
            body (str): Body unique name.
        """
        self.pc.removeBody(self._bodies_idx[body])

    def position_control(self, body: str, traget_q: list) -> None:
        """Position control the robot.

        Args:
            body (str): Body unique name.
            traget_q (list): Target joint angles.
        """
        if not self.control_joint:
            self.get_controllable_joints(body)
        assert len(traget_q) == len(self.control_joint), "The number of angles must match the number of controllable joints."
        
        for i in range(len(self.control_joint)):
            self.pc.setJointMotorControl2(
                bodyIndex=self._bodies_idx[body],
                jointIndex=self.control_joint[i],
                controlMode=p.POSITION_CONTROL,
                targetPosition=traget_q[i],
            )
    
    def disable_motor(self, body: str) -> None:
        if not self.control_joint:
            self.get_controllable_joints(body)
        for i in self.control_joint:
            self.pc.setJointMotorControl2(
                bodyIndex=self._bodies_idx[body],
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                force=0,
            )

    def setup_dynamics(self, body: str) -> None:
        self.pc.changeDynamics(self._bodies_idx[body], -1, 
                                   mass=0.1,
                                   restitution=0.0,
                                   )
        for i in range(0, self.get_num_joints(body)):
            self.pc.changeDynamics(
                bodyUniqueId=self._bodies_idx[body],
                linkIndex=i,
                restitution=0.0,
                contactStiffness=5000,
                contactDamping=200,
                collisionMargin=0.001,
                ccdSweptSphereRadius=0.001,
                contactProcessingThreshold=0.001,
                )
                
    def start_record(self, file_name: str) -> None:
        self.pc.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file_name)
    
    def stop_record(self) -> None:
        self.pc.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
        
    def reset_simulation(self) -> None:
        self.pc.resetSimulation()
        for constraint_id in range(self.pc.getNumConstraints()):
            self.pc.removeConstraint(constraint_id)

        # for body_id in range(self.pc.getNumBodies()):
        #     self.pc.applyExternalForce(body_id, -1, [0, 0, 0], [0, 0, 0], self.pc.WORLD_FRAME)
            
    def remove_all(self) -> None:
        objects = [self.pc.getBodyUniqueId(i) for i in range(self.pc.getNumBodies())]
        
        for obj in objects:
            self.pc.removeBody(obj)
    
    def get_camera_image(self) -> np.ndarray:

        
        width_m, height_m, rgb, depth, seg = self.pc.getCameraImage(self.width, 
                                                                    self.height,
                                                                    self.view_matrix,
                                                                    self.proj_matrix,
                                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(rgb, dtype=np.uint8)
        rgb_array = rgb.reshape((self.height, self.width, 4))[:, :, :3]  # Remove alpha
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)  
        return bgr_array
    
    def reset_jointstate(self, body: str, joint: np.ndarray ) -> None:
        for i in range(self.get_num_joints(body)):
            self.pc.resetJointState(self._bodies_idx[body], i, joint[i])

class robot(ABC):
    def __init__(
        self,
        sim: PybulletBase,

    )-> None:
        self.sim = sim
        self.body_name = None
        self.urdf_path = None
        # self.base_position = None
        # self.base_orientation = None
        self.endeffect = None
        self.init_joint = None
        self.calibration_rot = None
        self.calibration_trans = None
        self.pre_grasp = None
        #self.set_robot_info()
        #self.load_robot()
         
    @abstractmethod
    def set_robot_info(self) -> None:
        pass
    
    def load_robot(self, base_position, base_orientation, joints_val: np.ndarray, lateral_mu: float, Fixbase: int) -> None:
        if self.urdf_path is None:
            self.set_robot_info()
        
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=self.urdf_path,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=Fixbase,
        )
        self.sim.set_joint_angles_dof(self.body_name, angles=joints_val)
        # self.sim.set_lateral_friction(self.body_name, link=self.endeffect, lateral_friction=lateral_mu)
        # self.sim.set_spinning_friction(self.body_name, link=self.endeffect, spinning_friction=lateral_mu)
        # self.sim.set_rolling_friction(self.body_name, link=self.endeffect, rolling_friction=lateral_mu)
        self.sim.set_friction_all(self.body_name, lateral_friction=lateral_mu)
        self.sim.setup_dynamics(self.body_name)
        #self.sim.set_joint_angles_dof(self.body_name, angles=self.init_joint)
    
    def reset(self, position: np.ndarray, orientation: np.ndarray, joints_val: np.ndarray, lateral_mu: float) -> None:
        self.sim.set_base_pose(self.body_name, position, orientation)

        self.sim.set_joint_angles_dof(self.body_name, angles=joints_val)
        self.sim.set_lateral_friction(self.body_name, link=self.endeffect, lateral_friction=lateral_mu)
    
    def pre_transformation(self, rot: np.ndarray, trans: np.ndarray):

        R_f = np.matmul(rot, self.calibration_rot)
        T_f = -np.matmul(rot, np.matmul(self.calibration_rot, self.calibration_trans.reshape(3,1))) + trans.reshape(3,1)

        global_rot = R.from_matrix(R_f).as_quat()
        global_trans = T_f.reshape(3)
        return global_rot, global_trans
    
    def step(self) -> None:
        self.sim.step()
    
    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)
    
    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)
    
    def get_joint_angles_dof(self) -> np.ndarray:
        """Returns the angles of all joints.

        Returns:
            np.ndarray: Joint angles
        """
        return self.sim.get_joint_angles_dof(self.body_name)

    def get_joint_velocities_dof(self) -> np.ndarray:
        """Returns the velocities of all joints.

        Returns:
            np.ndarray: Joint velocities
        """
        return self.sim.get_joint_velocities_dof(self.body_name)

    def set_joint_angles(self, angles: list) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)
        
    def disable_motor(self) -> None:
        "disable motor"
        self.sim.disable_motor(self.body_name)
    
    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics
    
    def position_control(self, target_q: list) -> None:
        """Position control the robot.

        Args:
            target_q (list): Target joint angles.
        """
        self.sim.position_control(self.body_name, target_q)

    def has_reached(self, target_position: np.ndarray, threshold: float) -> bool:
        joint_position = self.get_joint_angles_dof()
        return np.linalg.norm(joint_position - target_position) < threshold
    
    def get_base_orientation(self) -> np.ndarray:
        return self.sim.get_base_orientation(self.body_name)
    
    def get_base_position(self) -> np.ndarray:
        return self.sim.get_base_position(self.body_name)

    
class objects:
    def __init__(
        self,
        sim: PybulletBase,
    )-> None:
        self.sim = sim
        self.obj_path = None
        self.position = None
        self.orientation = None
        self.obj_mass = None
        self.body_name = "object"
        self.mesh_cache = None
        self.vhacd_path = None
        self.global_constraint = None
    def convert_vhacd(self) -> None:
        """
        Convert the mesh to vhacd
        """
        obj_fn = self.obj_path.split(os.sep)[-1]
        obj_fn_no_ext = obj_fn.split('.')[0]
        vhacd_path = os.path.join(os.path.dirname(self.obj_path), obj_fn_no_ext + '_vhacd.obj')
        vhacd_log_path = os.path.join(os.path.dirname(self.obj_path), obj_fn_no_ext + '_vhacd.log')
        
        if not os.path.exists(vhacd_path):
            p.vhacd(fileNameIn=self.obj_path, fileNameOut=vhacd_path, fileNameLogging=vhacd_log_path, resolution=50000)
        
        self.vhacd_path = vhacd_path
    
    
    def load_vhacd_cache(self) -> None:
        """
        load the mesh as trimesh object
        """    
        self.mesh_cache = tm.load(self.vhacd_path, force='mesh', process=False)
    
    def set_object_info(self, file_path: str, position: np.ndarray, orientation: np.ndarray, obj_mass: float) -> None:
        self.obj_path = file_path
        self.position = position
        self.orientation = orientation
        self.obj_mass = obj_mass
    
    def load_object(self, mu) -> None:
        self.sim.load_obj_as_mesh(body_name=self.body_name, 
                                    obj_path=self.obj_path, 
                                    position=self.position, 
                                    orientation=self.orientation,
                                    obj_mass=self.obj_mass,
                                    frictional_coe=mu)
    
    def remove_object(self) -> None:
        self.sim.remove_body("object")  
        
    def set_mass(self, mass: float) -> None:
        self.sim.set_mass("object", link=-1, mass=mass)
        
    # def stable_position(self) -> None:
    def mesh_inertia(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.mesh_cache:
            self.load_vhacd_cache()
        self.mesh_cache.density = self.obj_mass / self.mesh_cache.volume
        return self.mesh_cache.moment_inertia, self.mesh_cache.center_mass

    def centroid(self) -> np.ndarray:
        if not self.mesh_cache:
            self.load_vhacd_cache()
        return self.mesh_cache.centroid
    
    def center_of_mass(self) -> np.ndarray:
        if not self.mesh_cache:
            self.load_vhacd_cache()
        return self.mesh_cache.center_mass
    
    def create_world_contraint(self) -> None:
        fix_base = self.sim.pc.createMultiBody(
            baseMass=0,  # Mass = 0 to keep it static
            baseCollisionShapeIndex=-1,  # No collision shape (invisible)
            baseVisualShapeIndex=-1,  # No visual shape
            basePosition=self.position,  # Same position as the object
            baseOrientation=self.orientation  # Same orientation as the object
            )
        
        self.global_constraint = self.sim.pc.createConstraint(
            parentBodyUniqueId=fix_base,
            parentLinkIndex=-1,
            childBodyUniqueId=self.sim._bodies_idx[self.body_name],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
    
    def create_mount_contraint(self) -> None:
        m_pose, m_quat = self.sim.get_base_position(self.body_name), self.sim.get_base_orientation(self.body_name)
        self.sim.loadURDF(  body_name="obj_mount",
                            fileName=osp.join(osp.dirname(__file__), 'xyz.urdf'),
                            basePosition=m_pose,
                            baseOrientation=m_quat,
                            useFixedBase=True)
        for joint in range(self.sim.get_num_joints("obj_mount")):
            info = self.sim.pc.getJointInfo(self.sim._bodies_idx["obj_mount"], joint)
            if info[12].decode('utf-8') == "end_effector_link":
                eelink_id = info[0]
                
        self.global_constraint = self.sim.pc.createConstraint(
            parentBodyUniqueId=self.sim._bodies_idx["obj_mount"],
            parentLinkIndex=eelink_id,
            childBodyUniqueId=self.sim._bodies_idx[self.body_name],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        
        for joint in range(self.sim.get_num_joints("obj_mount")):
            self.sim.pc.setJointMotorControl2(
                bodyIndex=self.sim._bodies_idx["obj_mount"], 
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL, 
                targetPosition=0,
                positionGain=0.2,
                velocityGain=0.1,
                force=10.0)
        
    def remove_world_contraint(self) -> None:
        self.sim.pc.removeConstraint(self.global_constraint)
    
    def generate_urdf(self) -> None:
        # Generate URDF file
        name = self.body_name
        mass = self.obj_mass
        origin = [0,0,0]
            
class dexycb_obj(Dataset):
    def __init__(self, dexycb_dir = '/home/liyuan/DexYCB/', task = "test", data_dir = osp.join(osp.dirname(osp.abspath(__file__)),"..","..", "CtcSDF", "data"), data_sample = 5):
        
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset("s0", task)
        config_file = osp.join(data_dir, "dexycb_{}_s0.json".format(task))
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        if data_sample is not None:
            
            sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
            with open(sample_dir, 'r') as f:
                sample = json.load(f)
                all_indices = [np.array(i) for i in sample.values()]
                self.sample = np.concatenate(all_indices)    
            self.len = len(self.sample)
        else:
            self.len = len(self.config)
            self.sample = None
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx_):
        if self.sample is not None:
            idx = self.sample[idx_]
        else:
            idx = idx_
        s0_id = self.config['images'][idx]['id']
        sample = self.getdata[s0_id]
        ycb_id = sample['ycb_grasp_ind']
        label = np.load(sample['label_file'])
        pose_y = label['pose_y'][ycb_id]
        obj_file = self.getdata.obj_file[sample['ycb_ids'][ycb_id]]
        pose = np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32)))
        
        hand_trans = np.array(self.config['annotations'][idx]['hand_trans'], dtype = np.float32)
        pose[:3, 3] = pose[:3, 3] - hand_trans
        
        translation = pose[:3, 3]
        rotation = R.from_matrix(pose[:3, :3])
        quat = rotation.as_quat()
        #OBJ loaded from different places
        return obj_file, pose, translation, quat

class result_fetcher(Dataset):
    def __init__(self, robot_name, mu, data_sample):
        self.dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results", "robot_name", "mu"+str(mu))
        self.robot_name = robot_name
        self.mu = mu
        self.data_sample = data_sample
    def __len__(self):
        return len(os.listdir(self.dir))
    
    def __getitem__(self, idx):
        image_id = idx // 3
        sub_id = idx % 3
        
        file_name = str(image_id) + "_" + self.robot_name + "_" + str(sub_id) + "mu_" + str(self.mu) + ".json"
        quat, t, val = self.opt_result_parser(osp.join(self.dir, file_name))
        return quat, t, val
                    
    def opt_result_parser(self, file_path: str) -> Dict:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        q = np.array(data['qr']).squeeze()
        #global_rmat = rodrigues(torch.tensor(q[3:6], dtype=torch.float32)).view(3, 3).numpy()
        global_quat = rodrigues(q[3:6])
        global_t = q[:,:3]
        joint_val = q[:,6:]
        
        return global_quat, global_t, joint_val 
        
class ycb_opt_fetcher(Dataset):
    def __init__(self, 
                 dexycb_dir = '/home/liyuan/DexYCB/', 
                 task = "test", 
                 data_dir = osp.join(osp.dirname(osp.abspath(__file__)),"..","..", "CtcSDF", "data"), 
                 data_sample = 5,
                 robot_name = "Shadow",
                 mu = 0.1,
                 SDF_source = False,
                 repeat = 1,
                 load_assert = False,
                 exp_name = "genhand",
                 test_sdf = False,
                ):
        self.sdf_source = SDF_source
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset("s0", task)
        self.robot_name = robot_name
        self.mu = mu
        self.data_sample = data_sample
        if exp_name == "genhand" and test_sdf == False:
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results", self.robot_name, "mu"+str(mu))
        elif exp_name == "nv" and test_sdf == False:
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results_nv", self.robot_name, "mu"+str(mu))
        elif task == "train":
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..")
        elif task == "test" and exp_name is None:
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..")
        elif test_sdf == True and exp_name == 'genhand':
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results_sdf", self.robot_name, "mu"+str(mu))
        elif test_sdf == True and exp_name == 'nv':
            self.result_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "results_nv_sdf", self.robot_name, "mu"+str(mu))
        
        self.hand_pred_pose_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "CtcSDF_v2", "hmano_osdf", "hand_pose_results")
        self.obj_pred_pose_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "CtcSDF_v2", "hmano_osdf", "obj_pose_results")
        self.obj_pred_mesh_dir = osp.join(osp.abspath(osp.dirname(__file__)), "..", "..", "CtcSDF_v2", "hmano_osdf", "mesh")
        self.mesh_dir = osp.join(data_dir, 'mesh_data', 'mesh_obj')
        self.repeat = repeat
        self.load_assert = load_assert
        config_file = osp.join(data_dir, "dexycb_{}_s0.json".format(task))
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        if data_sample is not None:
            sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
            with open(sample_dir, 'r') as f:
                sample = json.load(f)
                all_indices = [np.array(i) for i in sample.values()]
                self.sample = np.concatenate(all_indices)
        else:
            self.sample = None
            
        if self.load_assert:
            self.mano_layer = ManoLayer(
                        ncomps = 45,
                        side = 'right',
                        mano_root = os.path.join(os.path.dirname(__file__),'..', '..', 'manopth','mano','models'),
                        use_pca=True,
                        flat_hand_mean=False
                        )
            self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)),'..', '..', 'CtcSDF','closed_fmano.npy')))            
        
    def __len__(self):
        return len(os.listdir(self.result_dir))

    def __getitem__(self, idx_):


        sample_id = idx_ // self.repeat
        res_id = idx_ % self.repeat
        if self.sample is not None:
            idx = self.sample[sample_id]
        else:
            idx = sample_id

        s0_id = self.config['images'][idx]['id']
        file_name = self.config['images'][idx]['file_name']
        sample = self.getdata[s0_id]
        ycb_id = sample['ycb_grasp_ind']
        label = np.load(sample['label_file'])
        pose_y = label['pose_y'][ycb_id]

        fx = sample['intrinsics']['fx']
        fy = sample['intrinsics']['fy']
        cx = sample['intrinsics']['ppx']
        cy = sample['intrinsics']['ppy']
        w = self.getdata.w
        h = self.getdata.h  

        if self.sdf_source:
            hand_trans = np.array(self.config['annotations'][idx]['hand_trans'], dtype = np.float32)
            obj_name = self.config["images"][idx]['file_name']+'.obj'
            obj_file = osp.normpath(osp.join(self.mesh_dir, obj_name))
            # print("dir check", obj_file)
            translation = - hand_trans
            quat = np.array([0, 0, 0, 1])    
            mat = np.eye(3)
            pose = np.vstack((np.hstack([mat, translation.reshape(3, 1)]), np.array([0, 0, 0, 1])))
        else:
            # print("using ycb source")
            # print("obj categroy", self.getdata.obj_file)
            obj_file = self.getdata.obj_file[sample['ycb_ids'][ycb_id]]
            # print("dir check", obj_file)
            pose = np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32)))
            hand_trans = np.array(self.config['annotations'][idx]['hand_trans'], dtype = np.float32)
            translation = pose[:3, 3]- hand_trans
            pose[:3, 3] = pose[:3, 3]- hand_trans

            rotation = R.from_matrix(pose[:3, :3])
            quat = rotation.as_quat()
        
        if self.load_assert:
            color = cv2.imread(self.getdata[s0_id]['color_file'])
            pose_m = torch.from_numpy(label['pose_m'])
            bates = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
            verts,_,_,_,_,_ = self.mano_layer( pose_m[:, 0:48],bates, pose_m[:, 48:51])
            #th_verts, th_jtr, th_full_pose,th_results_global,center_joint, root_trans
            # print("hand trans", pose_m[:, 48:51])
            verts = verts.view(778,3).numpy()
            pose_y_ = np.vstack((label['pose_y'][ycb_id],np.array([[0, 0, 0, 1]], dtype=np.float32)))
        
        obj_centre = np.array(self.config['annotations'][idx]['obj_center_3d'])
        obj_centre = obj_centre - hand_trans
        result_file = str(sample_id) + "_" + self.robot_name + "_" + str(res_id) + "_mu_" + str(self.mu) + ".json"
        print(osp.join(self.result_dir, result_file))
        # print("result file", result_file)
        try:
            r_quat, r_t, q_val, mat, q = self.opt_result_parser(osp.join(self.result_dir, result_file))
        except Exception as e:
            traceback.print_exc()
            print("Error in loading", result_file)
            r_quat = 0
            r_t = 0
            q_val = 0
            mat = 0
            q = 0
        if self.load_assert:
            return dict(file_name = file_name, color = color ,obj_file =obj_file, pose_y = pose_y_, hand_verts = verts, hand_faces = self.faces, obj_quat = quat, obj_trans = translation, obj_centre = obj_centre, rob_quat = r_quat, rob_trans = r_t, joint_val = q_val, rob_mat = mat, full_q = q, se3 = pose, fx = fx, fy = fy, cx = cx, cy = cy, w = w, h = h)
        else:
            return dict(obj_file =obj_file, obj_quat = quat, obj_trans = translation, obj_centre = obj_centre, rob_quat = r_quat, rob_trans = r_t, joint_val = q_val, rob_mat = mat, full_q = q, se3 = pose, fx = fx, fy = fy, cx = cx, cy = cy, w = w, h = h)
        
    def opt_result_parser(self, file_path: str) -> Dict:
        print(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
        # if data['distance_loss'] > 0.01:

        #     q = np.array(data['qr']).squeeze()
        # else:
        q = np.array(data['q']).squeeze()
        #global_rmat = rodrigues(torch.tensor(q[3:6], dtype=torch.float32)).view(3, 3).numpy()
        global_quat, mat = rodrigues(q[3:6])
        global_t = q[:3]
        if self.robot_name == "Robotiq":
            joint_val = q[6:] * np.array([-1, -1, 1, -1, 1, -1])
            # [-1, -1, 1, -1, 1, -1]
            # joint_val = q[6:] * np.array([1, 1, -1, -1, 1, -1])

        else:
            joint_val = q[6:]
        # se3 = np.vstack([np.hstack([mat, global_t.reshape(3, 1)]), np.array([0, 0, 0, 1])])
        
        return global_quat, global_t, joint_val, mat, q#, se3

class sim_result_loader(Dataset):
    def __init__(self,robot,mu,repeat=1):
        super(sim_result_loader, self).__init__()
        self.robot = robot
        self.mu = mu
        self.repeat = repeat
        self.data_dir = self.result_path = osp.join(osp.dirname(osp.abspath(__file__)),'sim_result')
        self.file_list = [f for f in os.listdir(self.result_path) if robot in f and (str(mu) in f)]
        print(len(self.file_list))
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_name = self.robot + '_' + str(self.mu) + '_' + str(idx) + '.json'
        with open(osp.join(self.result_path, file_name), 'r') as file:
            data = json.load(file)
        if data['flag'] == True:
            return 1
        elif data['flag'] == False:
            return 0
        else:
            print(data['flag'])
            return -1
        
class shadow(robot):
    def set_robot_info(self) -> None:
         self.body_name = "Shadow"
         self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'sr_description','urdf','shadow_hand_pybullet.urdf')
        #  self.base_position = position
        #  self.base_orientation = orentation
         #self.endeffect =[7, 12, 17, 23, 29]
         self.endeffect = [6, 11, 16, 22, 28]
         self.pre_grasp = np.array([0.0, 0.2, -0.2])
         self.init_joint = [0., 0., -0.349, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -0.349, 0., 0., 0., -1.047, 1.222, 0., 0., 0.]
         self.calibration_rot = rodrigues(np.array([0,-(np.pi/2),0]))[1] #mat
         self.calibration_trans = np.array([0.0000, -0.0100, 0.2130])
         
class allegro(robot):
    def set_robot_info(self) -> None:
        self.body_name = "Allegro"
        self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'allegro_hand','allegro_hand_description_right.urdf')
        self.endeffect = []
        self.pre_grasp = np.array([-0.2, 0.0, -0.2])

        self.init_joint = [0.0]*16
        self.calibration_rot = euler_to_mat(np.array([-torch.pi/2,torch.pi,torch.pi/2]), degrees=False) #mat
        self.calibration_trans = np.array([0.0500, 0.0000, 0.1000])

class barrett(robot):
    def set_robot_info(self) -> None:
        self.body_name = "Barrett"
        self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'barrett_adagrasp','model.urdf')
        self.endeffect = []
        self.pre_grasp = np.array([0.0, 0.0, -0.2]) 
        self.init_joint = [0.0]*8
        self.calibration_rot = euler_to_mat(np.array([0,torch.pi/2,torch.pi]), degrees=False) #mat
        self.calibration_trans = np.array([0.0000, 0.0000, 0.0000])

class robotiq(robot):
    def set_robot_info(self) -> None:
        self.body_name = "Robotiq"
        self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'robotiq_arg85','urdf','robotiq_arg85_description.urdf')
        # self.urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..',  'model', 'urdf', 'robotiq_85v2', 'robotiq_2f_85_v3.urdf')
       
        self.endeffect = []
        self.pre_grasp = np.array([0.0, 0.0, -0.2])
        self.init_joint = [0.0]*6
        self.calibration_rot = euler_to_mat(np.array([torch.pi/2,torch.pi,3*torch.pi/2]), degrees=False) #mat
        self.calibration_trans = np.array([0.0000, 0.0000, 0.0000])


class MountingBase():
    def __init__(self, 
                 sim: PybulletBase,
                 gripper: robot,
                 ) -> None:
        self.sim = sim
        self.body_name = "base"
        self.urdf_path = osp.join(osp.dirname(__file__), 'xyz.urdf')
        self.gripper = gripper
        
    def mounting_gripper(self):
        m_pose, m_quat = self.sim.get_base_position(self.gripper.body_name), self.sim.get_base_orientation(self.gripper.body_name)
        self.sim.loadURDF(body_name=self.body_name, 
                                  fileName=self.urdf_path, 
                                  basePosition=m_pose, 
                                  baseOrientation=m_quat, 
                                  useFixedBase=True)
        
        for joint in range(self.sim.get_num_joints(self.body_name)):
            info = self.sim.pc.getJointInfo(self.sim._bodies_idx[self.body_name], joint)
            if info[12].decode('utf-8') == "end_effector_link":
                eelink_id = info[0]
        
        c_id = self.sim.pc.createConstraint(self.sim._bodies_idx[self.body_name], eelink_id, self.sim._bodies_idx[self.gripper.body_name], -1, 
                                            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
                                            parentFrameOrientation=[0, 0, 0, 1], childFrameOrientation=[0, 0, 0, 1]
        )
        
        # self.sim.pc.setJointMotorControlArray(self.sim._bodies_idx[self.body_name], 
        #                                       list(range(self.sim.get_num_joints(self.body_name))), 
        #                                       controlMode=p.POSITION_CONTROL, 
        #                                       targetPositions=[0.0]*self.sim.get_num_joints(self.body_name),
        #                                       forces=[1000.0]*self.sim.get_num_joints(self.body_name))
        return self.sim._bodies_idx[self.body_name], c_id

    def reset_mount(self, position: np.ndarray, orientation: np.ndarray):
        self.sim.set_base_pose(self.body_name, position, orientation)
    
    def get_num_joints(self):
        return self.sim.get_num_joints(self.body_name)
    
    def get_base_position(self):
        return self.sim.get_base_position(self.body_name)
    
    def get_base_orientation(self):
        return self.sim.get_base_orientation(self.body_name)
    
    def get_joint_angles(self):
        return self.sim.get_joint_angles(self.body_name, list(range(self.get_num_joints())))

def rodrigues(axisang):
    """rewrite from the torch version batch_rodrigues from mano repo
       The quaternions are written in scaler-last manner to be compatible with bullet3
    """
    axisang_norm = np.linalg.norm(axisang + 1e-8, ord=2)
    axisang_normalized = axisang/axisang_norm
    angle = axisang_norm * 0.5
    
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    quat = np.hstack([v_sin*axisang_normalized, v_cos]) #xyzw
    quat_r = R.from_quat(quat)
    mat = quat_r.as_matrix()
    return quat, mat

def quat_mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = x1 * w2 + w1 * x2 + y1 * z2 - z1 * y2
    y = y1 * w2 + w1 * y2 + z1 * x2 - x1 * z2
    z = z1 * w2 + w1 * z2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])

def quaternion_conjugate(q):
    """
    Computes the conjugate of a quaternion (scalar-last format).
    Args:
        q: Quaternion [x, y, z, w]
    Returns:
        Conjugate quaternion [-x, -y, -z, w]
    """
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def rotate_vector(vector, quaternion):
    """
    Rotates a 3D vector using a quaternion (scalar-last format).
    Args:
        vector: 3D vector [vx, vy, vz]
        quaternion: Rotation quaternion [x, y, z, w]
    Returns:
        Rotated 3D vector [vx', vy', vz']
    """
    q = normalize_quaternion(quaternion)
    # Convert vector to quaternion (q_v)
    q_v = np.array([*vector, 0])  # Scalar part is 0

    # Compute rotated quaternion: q_rotated = q * q_v * q_conjugate
    q_conjugate = quaternion_conjugate(quaternion)
    q_rotated = quat_mult(quat_mult(quaternion, q_v), q_conjugate)

    # Extract the rotated vector (ignore the scalar part)
    return q_rotated[:3]

def cubic_motion_planning(initial_positions: np.ndarray, target_positions: np.ndarray, t0: int, tf: int, num_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plan a cubic motion trajectory for multiple joints.
    
    initial_positions: List of initial joint positions
    target_positions: List of target joint positions
    initial_velocities: List of initial joint velocities
    target_velocities: List of target joint velocities
    t0: Initial time
    tf: Final time
    num_points: Number of points in the trajectory
    
    Returns:
    positions: Joint positions trajectory
    velocities: Joint velocities trajectory
    accelerations: Joint accelerations trajectory
    time_steps: Time steps of the trajectory
    """
    initial_velocities = np.zeros_like(initial_positions) 
    target_velocities = np.zeros_like(target_positions)
    
    def cubic_spline_coefficients(q0, qf, v0, vf, t0, tf):
        A = np.array([[1, t0, t0**2, t0**3],
                    [0, 1, 2*t0, 3*t0**2],
                    [1, tf, tf**2, tf**3],
                    [0, 1, 2*tf, 3*tf**2]])
        b = np.array([q0, v0, qf, vf])
        coefficients = np.linalg.solve(A, b)
        return coefficients

    def cubic_spline_trajectory(coefficients, t):
        a0, a1, a2, a3 = coefficients
        q = a0 + a1*t + a2*t**2 + a3*t**3
        q_dot = a1 + 2*a2*t + 3*a3*t**2
        q_ddot = 2*a2 + 6*a3*t
        return q, q_dot, q_ddot

    num_joints = len(initial_positions)
    time_steps = np.linspace(t0, tf, num_points)

    positions = np.zeros((num_points, num_joints))
    velocities = np.zeros((num_points, num_joints))
    accelerations = np.zeros((num_points, num_joints))

    for i in range(num_joints):
        coeffs = cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                        initial_velocities[i], target_velocities[i],
                                        t0, tf)
        for j, t in enumerate(time_steps):
            q, q_dot, q_ddot = cubic_spline_trajectory(coeffs, t)
            positions[j, i] = q
            velocities[j, i] = q_dot
            accelerations[j, i] = q_ddot

    return positions, velocities, accelerations, time_steps

def rotate_vector_inverse(vector, quaternion):
    """
    Rotate a vector using the inverse of a quaternion
    """
    #Normalize the quaternion
    q = normalize_quaternion(quaternion)
    
    q_v = np.array([*vector, 0])
    
    q_conjugate = quaternion_conjugate(q)
    
    q_rotated = quat_mult(quat_mult(q_conjugate, q_v), q)
    
    return q_rotated[:3]
    
class Trajectory:
    def __init__(self, initial_positions: np.ndarray, initial_joint:np.ndarray, grasp_position: np.ndarray, grasp_joint: np.ndarray, lift_up: np.ndarray, time_step = 1000) -> None:
        self.initial_positions = initial_positions
        self.initial_joint = initial_joint
        self.grasp_position = grasp_position
        self.grasp_joint = grasp_joint
        self.liftup_position = lift_up
        self.tm = time_step
        self.pre_grasp = self.grasp_position*0.5
        self.time_interval = [0, 2, 2, 3]
        # print("pre_grasp", self.pre_grasp)
        # print("liftup", self.liftup_position)
        # self.coefficients = None        
    
    @property
    def get_duration(self):
        return self.time_interval[-1]
    
    @property
    def get_reach_duration(self):
        return self.time_interval[-2]
    
    def cubic_motion_planning(self, initial_positions: np.ndarray, target_positions: np.ndarray, t0: int, tf: int, num_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Plan a cubic motion trajectory for multiple joints.
        
        initial_positions: List of initial joint positions
        target_positions: List of target joint positions
        initial_velocities: List of initial joint velocities
        target_velocities: List of target joint velocities
        t0: Initial time
        tf: Final time
        num_points: Number of points in the trajectory
        
        Returns:
        positions: Joint positions trajectory
        velocities: Joint velocities trajectory
        accelerations: Joint accelerations trajectory
        time_steps: Time steps of the trajectory
        """
        initial_velocities = np.zeros_like(initial_positions) 
        target_velocities = np.zeros_like(target_positions)
        
        def cubic_spline_coefficients(q0, qf, v0, vf, t0, tf):
            A = np.array([[1, t0, t0**2, t0**3],
                        [0, 1, 2*t0, 3*t0**2],
                        [1, tf, tf**2, tf**3],
                        [0, 1, 2*tf, 3*tf**2]])
            b = np.array([q0, v0, qf, vf])
            coefficients = np.linalg.solve(A, b)
            return coefficients

        def cubic_spline_trajectory(coefficients, t):
            a0, a1, a2, a3 = coefficients
            q = a0 + a1*t + a2*t**2 + a3*t**3
            q_dot = a1 + 2*a2*t + 3*a3*t**2
            q_ddot = 2*a2 + 6*a3*t
            return q, q_dot, q_ddot

        num_joints = len(initial_positions)
        time_steps = np.linspace(t0, tf, num_points)

        positions = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        accelerations = np.zeros((num_points, num_joints))

        for i in range(num_joints):
            coeffs = cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            t0, tf)
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot

        return positions, velocities, accelerations, time_steps
    
    def solve_cube_polynomial(self, coefficients, x, t_interval):
        """
        Solve the cubic polynomial for t given x.
        a3*t^3 + a2*t^2 + a1*t + (a0 - x) = 0
        """
        
        a0, a1, a2, a3 = coefficients
        new_coe = [a3, a2, a1, a0 - x]
        roots = np.roots(new_coe)
        real_roots = [t.real for t in roots if np.isreal(t) and t_interval[0] <= t.real <= t_interval[1]]
        if real_roots:
            return min(real_roots)
        else:
            return None
        
    def cubic_spline_coefficients(self, q0, qf, v0, vf, t0, tf):
            A = np.array([[1, t0, t0**2, t0**3],
                        [0, 1, 2*t0, 3*t0**2],
                        [1, tf, tf**2, tf**3],
                        [0, 1, 2*tf, 3*tf**2]])
            b = np.array([q0, v0, qf, vf])
            coefficients = np.linalg.solve(A, b)
            return coefficients            
    
    def cubic_spline_trajectory(self, coefficients, t):
            a0, a1, a2, a3 = coefficients
            q = a0 + a1*t + a2*t**2 + a3*t**3
            q_dot = a1 + 2*a2*t + 3*a3*t**2
            q_ddot = 2*a2 + 6*a3*t
            return q, q_dot, q_ddot
    
    def basic_motion_planning(self, initial_positions: np.ndarray, target_positions: np.ndarray, t0: int, tf: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial_velocities = np.zeros_like(initial_positions)
        target_velocities = np.zeros_like(target_positions)
        num_joints = len(initial_positions) 
        time_steps = np.linspace(t0, tf, num_points)
        
        positions = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        accelerations = np.zeros((num_points, num_joints))
        
        for i in range(num_joints):
            coeffs = self.cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            t0, tf)
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = self.cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot

        return positions, velocities, accelerations, time_steps        
    
    def basic_motion_planning_asynb(self, initial_positions: np.ndarray, target_positions: np.ndarray, via_position: np.ndarray, t0: int, tf: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        initial_velocities = np.zeros_like(initial_positions)
        target_velocities = np.zeros_like(target_positions)
        
        num_joints = len(initial_positions) 
        time_steps = np.linspace(t0, tf, num_points)
        # print("chect base setup", initial_positions, target_positions, via_position, t0, tf)
        positions = np.zeros((num_points, num_joints))
        velocities = np.zeros((num_points, num_joints))
        accelerations = np.zeros((num_points, num_joints))
        
        via_time = np.zeros_like(target_positions)
        
        for i in range(num_joints):
            coeffs = self.cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            t0, tf)
            via_time[i] = self.solve_cube_polynomial(coeffs, via_position[i], [t0, tf])
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = self.cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot
        timestamp = (np.nanmean(via_time) * self.tm).astype(int)
        return positions, velocities, accelerations, time_steps, timestamp, np.nanmean(via_time)
        
    def basic_motion_planning_asynj(self, initial_positions: np.ndarray, target_positions: np.ndarray, tf: int, tt: int, timestamp: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial_velocities = np.zeros_like(initial_positions)
        target_velocities = np.zeros_like(target_positions)
        
        num_joints = len(initial_positions)
        time_steps = np.linspace(tf, tt, num_points-timestamp)
        
        positions = np.zeros((num_points-timestamp, num_joints))
        velocities = np.zeros((num_points-timestamp, num_joints))
        accelerations = np.zeros((num_points-timestamp, num_joints))
        
        for i in range(num_joints):
            coeffs = self.cubic_spline_coefficients(initial_positions[i], target_positions[i],
                                            initial_velocities[i], target_velocities[i],
                                            tf, tt)
            for j, t in enumerate(time_steps):
                q, q_dot, q_ddot = self.cubic_spline_trajectory(coeffs, t)
                positions[j, i] = q
                velocities[j, i] = q_dot
                accelerations[j, i] = q_ddot
        
        return positions, velocities, accelerations, time_steps
        
        
    def cubic_motion_planning_syn(self, initial_positions: Tuple[np.ndarray,np.ndarray], target_positions: Tuple[np.ndarray,np.ndarray], t0: int, tf: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray]:   
        base_init_pos, gripper_init_pos = initial_positions
        base_target_pos, gripper_target_pos = target_positions
        qb, vb, ab, tb = self.basic_motion_planning(base_init_pos, base_target_pos, t0, tf, num_points)
        qg, vg, ag, tg = self.basic_motion_planning(gripper_init_pos, gripper_target_pos, t0, tf, num_points)
        return qb, vb, ab, tb, qg, vg, ag, tg
        
    def cubic_motion_planning_asyn(self, initial_positions: Tuple[np.ndarray,np.ndarray], target_positions: Tuple[np.ndarray,np.ndarray], via_position: np.ndarray, t0: int, tf: int, tt: int, num_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Moving the base earlier than the gripper
        Base:  -------------->__________
        Joint: ___________------------->
        """
        base_init_pos, gripper_init_pos = initial_positions
        base_target_pos, gripper_target_pos = target_positions
        num_base = len(base_init_pos)
        num_gripper = len(gripper_init_pos)
        qb_ = np.ones((num_points, num_base))*base_target_pos
        qj_ = np.ones((num_points, num_gripper))*gripper_init_pos
        
        vb_ = np.zeros((num_points, num_base))
        vj_ = np.zeros((num_points, num_gripper))
        
        ab_ = np.zeros((num_points, num_base))
        aj_ = np.zeros((num_points, num_gripper))
        
        # tb_ = np.linspace(t0, tf, num_points)
        # tj_ = np.linspace(t0, tt, num_points)
        
        
        qb, vb, ab, tb, timestamp, tv = self.basic_motion_planning_asynb(base_init_pos, base_target_pos, via_position, t0, tf, int(self.tm*(tf-t0)))
        # print("check joint setup", len(gripper_init_pos), len(gripper_target_pos), tv, tt, timestamp, num_points)
        qj, vj, aj, tj = self.basic_motion_planning_asynj(gripper_init_pos, gripper_target_pos, tv, tt, timestamp, num_points)    
        
        
        
        qb_[:int(self.tm*(tf-t0))], vb_[:int(self.tm*(tf-t0))], ab_[:int(self.tm*(tf-t0))] = qb, vb, ab
        
        qj_[timestamp:], vj_[timestamp:], aj_[timestamp:] = qj, vj, aj
        
        return qb_, vb_, ab_, tb, qj_, vj_, aj_, tj    
    
    def trajectory_generate_v2(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        time_itv = self.time_interval
        qb_0, vb_0, ab_0, tb_0, qj_0, vj_0, aj_0, tj_0 = self.cubic_motion_planning_asyn((self.initial_positions, self.initial_joint), (self.grasp_position, self.grasp_joint), self.pre_grasp, time_itv[0], time_itv[1], time_itv[2], int(self.tm*(time_itv[2]-time_itv[0])))
        qb_1, vb_1, ab_1, tb_1, qj_1, vj_1, aj_1, tj_1 = self.cubic_motion_planning_syn((self.grasp_position, self.grasp_joint), (self.liftup_position, self.grasp_joint), time_itv[2], time_itv[3], int(self.tm*(time_itv[3]-time_itv[2])))
        # print("check size", qb_0.shape, qb_1.shape, qj_0.shape, qj_1.shape, tb_0.shape, tb_1.shape, tj_0.shape, tj_1.shape, ab_0.shape, ab_1.shape, aj_0.shape, aj_1.shape)
        
        return np.concatenate((qb_0, qb_1)), np.concatenate((vb_0, vb_1)), np.concatenate((ab_0, ab_1)), np.concatenate((tb_0, tb_1)), np.concatenate((qj_0, qj_1)), np.concatenate((vj_0, vj_1)), np.concatenate((aj_0, aj_1)), np.concatenate((tj_0, tj_1))
    
    def trajectory_generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        from the pre_grasp to the grasp position:
            base: move from the pre_grasp to the grasp position
            gripper: stay
            
        grasp position:
            base: stay
            gripper: move from the initial joint to the grasp joint
            
        from the grasp position to the lift up position:
            base: move from the grasp position to the lift up position
            gripper: stay
            
            
        need add on more pre-grasp position    
        """
        time_interval = self.time_interval
        qb_0, vb_0, ab_0, tb_0 = self.cubic_motion_planning(self.initial_positions, self.grasp_position, time_interval[0], time_interval[1], self.tm*int(time_interval[1]-time_interval[0]))
        qg_0, vg_0, ag_0, tg_0 = self.cubic_motion_planning(self.initial_joint, self.initial_joint, time_interval[0], time_interval[1], self.tm*int(time_interval[1]-time_interval[0]))
        qb_1, vb_1, ab_1, tb_1 = self.cubic_motion_planning(self.grasp_position, self.grasp_position, time_interval[1], time_interval[2], self.tm*int(time_interval[2]-time_interval[1]))
        qg_1, vg_1, ag_1, tg_1 = self.cubic_motion_planning(self.initial_joint, self.grasp_joint, time_interval[1], time_interval[2], self.tm*int(time_interval[2]-time_interval[1]))
        qb_2, vb_2, ab_2, tb_2 = self.cubic_motion_planning(self.grasp_position, self.liftup_position, time_interval[2], time_interval[3], self.tm*int(time_interval[3]-time_interval[2]))
        qg_2, vg_2, ag_2, tg_2 = self.cubic_motion_planning(self.grasp_joint, self.grasp_joint, time_interval[2], time_interval[3], self.tm*int(time_interval[3]-time_interval[2]))
        
        return np.concatenate((qb_0, qb_1, qb_2)), np.concatenate((vb_0, vb_1, vb_2)), np.concatenate((ab_0, ab_1, ab_2)), np.concatenate((tb_0, tb_1, tb_2)), np.concatenate((qg_0, qg_1, qg_2)), np.concatenate((vg_0, vg_1, vg_2)), np.concatenate((ag_0, ag_1, ag_2)), np.concatenate((tg_0, tg_1, tg_2))
        
def normalize_quaternion(q):
    """Normalize a quaternion."""
    norm = np.linalg.norm(q)
    return q / norm

def euler_to_mat(euler, degrees=True):
    #euler = euler.cpu().numpy()
    r = R.from_euler('xyz', euler, degrees=degrees).as_matrix()
    return np.array(r)

def reverse_rotate_mat(v, q):
    q = normalize_quaternion(q)
    r_m = R.from_quat(q).as_matrix()
    return np.dot(r_m.T, v)
    


class Environment:
    def __init__(self,
                 robot: str = "Shadow",
                 mu: float = 0.9,
                 exp: str = "genhand",
                 repeat: int = 1,
                 render: str = "GUI",
                 data_sample: int = 20,
                 task: str = "test",
                 test_sdf: bool = False,
                 ) -> None:
        self.sim = PybulletBase(connection_mode=render)
        if robot == "Shadow":
            self.robot = shadow(self.sim)
        elif robot == "Allegro":
            self.robot = allegro(self.sim)
        elif robot == "Barrett":
            self.robot = barrett(self.sim)
        elif robot == "Robotiq":
            self.robot = robotiq(self.sim)
        self.robot.set_robot_info()
        self.object = objects(self.sim)
        self.base = MountingBase(self.sim, self.robot)
        self.mu = mu
        self.data_fetcher = ycb_opt_fetcher(robot_name=self.robot.body_name,
                                            mu = mu, 
                                            exp_name=exp, 
                                            repeat=repeat, 
                                            data_sample=data_sample, 
                                            task=task, 
                                            test_sdf=test_sdf)
        # self.data_fetcher = ycb_opt_fetcher(robot_name=self.robot.body_name,mu = mu, exp_name=exp, repeat=repeat, data_sample=None, task="test")

        self.re_orent = R.from_matrix(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])).as_quat()
        self.pre_grasp = self.robot.pre_grasp
        self.rob_joints = None
        self.obj_quat = None
        self.rob_glo_quat = None
        self.rob_glo_trans = None
        self.obj_quat = None
        self.obj_trans = None
        self.lift_up = np.array([0.0, 0.0, 0.3])

        if task == "test" and test_sdf is False:
            if exp == "genhand":
                self.record_dir = osp.join(osp.dirname(__file__), 'record') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result')
            elif exp == "nv":
                self.record_dir = osp.join(osp.dirname(__file__), 'record_nv') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_nv')
                self.image_dir = osp.join(osp.dirname(__file__), 'image_nv')
            else:
                self.record_dir = osp.join(osp.dirname(__file__), 'record_test')
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_test')
            if not osp.exists(self.record_dir):
                os.makedirs(self.record_dir)
            if not osp.exists(self.sim_result_dir):
                os.makedirs(self.sim_result_dir)
        if task == "test" and test_sdf is True:
            if exp == "genhand":
                self.record_dir = osp.join(osp.dirname(__file__), 'record_sdf') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_sdf')
            elif exp == "nv":
                self.record_dir = osp.join(osp.dirname(__file__), 'record_sdf_nv') 
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_sdf_nv')
                self.image_dir = osp.join(osp.dirname(__file__), 'image_sdf_nv')
            else:
                self.record_dir = osp.join(osp.dirname(__file__), 'record_test')
                self.sim_result_dir = osp.join(osp.dirname(__file__), 'sim_result_test')
            if not osp.exists(self.record_dir):
                os.makedirs(self.record_dir)
            if not osp.exists(self.sim_result_dir):
                os.makedirs(self.sim_result_dir)            

        self.image_dir = osp.join(osp.dirname(__file__), 'image')
        if not osp.exists(self.image_dir):
            os.makedirs(self.image_dir)
    def save_image(self, rgb, name):
        cv2.imwrite(osp.join(self.image_dir, name), rgb)
    
    def initialisation(self, data_pack) -> None:
        # data_pack = self.data_fetcher[idx]
        obj_quat = data_pack["obj_quat"]
        obj_trans = data_pack["obj_trans"]
        obj_centre = data_pack["obj_centre"]
        rob_mat = data_pack["rob_mat"]
        rob_trans = data_pack["rob_trans"]
        self.rob_joints = data_pack["joint_val"]
        self.obj_file = data_pack["obj_file"]
        print("joint values",self.rob_joints)
        rob_glo_quat, rob_glo_trans = self.robot.pre_transformation(rob_mat, rob_trans)
        self.rob_glo_quat = quat_mult(self.re_orent, rob_glo_quat)
        self.obj_quat = quat_mult(self.re_orent, obj_quat)
        self.rob_glo_trans = rotate_vector(rob_glo_trans, self.re_orent)
        self.obj_trans = rotate_vector(obj_trans-obj_centre, self.re_orent)
        self.pre_grasp_ = rotate_vector(self.pre_grasp, self.rob_glo_quat)
        
        self.traj = Trajectory(initial_positions=np.zeros(3), 
                          initial_joint=self.robot.init_joint, 
                          grasp_position=-1*self.robot.pre_grasp, 
                          grasp_joint=self.rob_joints, 
                          lift_up=rotate_vector_inverse((self.lift_up-self.pre_grasp_), self.rob_glo_quat))
        #self.record_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".mp4"
    
    def load(self) -> None:
        self.object.set_object_info(file_path=self.obj_file, 
                                    position=self.obj_trans, 
                                    orientation=self.obj_quat, 
                                    obj_mass=0.1)
        self.object.load_object(mu=self.mu)
        self.sim.load_plane()
        #self.object.create_world_contraint()
        self.robot.load_robot(base_position=self.rob_glo_trans+self.pre_grasp_, 
                              base_orientation=self.rob_glo_quat, 
                              joints_val=self.robot.init_joint, 
                              lateral_mu = self.mu, 
                              Fixbase=False)
        self.mount, _ = self.base.mounting_gripper()
        #self.robot.disable_motor()
    
    def load_grsap_position(self) -> None:
        self.sim.load_plane()
        self.robot.load_robot(base_position=self.rob_glo_trans,
                                base_orientation=self.rob_glo_quat,
                                joints_val=self.rob_joints,
                                lateral_mu=self.mu,
                                Fixbase=False)
        self.mount, _ = self.base.mounting_gripper()
        self.object.set_object_info(file_path=self.obj_file,
                                    position=self.obj_trans,
                                    orientation=self.obj_quat,
                                    obj_mass=0.1)
    
    def image_screenshot(self) -> None:
        self.robot.disable_motor()
        self.object.create_world_contraint()
        for t in range(2):
            self.sim.pc.stepSimulation()
            if t == 1:
                image = self.sim.get_camera_image()
        return image

    def start_image(self) -> None:
        qb, vb, ab, tb, qj, vj, aj, tj = self.traj.trajectory_generate_v2()
        self.robot.disable_motor()
        # self.object.create_mount_contraint()
        self.object.create_world_contraint()
        for t in range(1000*self.traj.get_reach_duration):

            for i in range(len(self.sim.control_joint)):
                self.sim.pc.setJointMotorControl2(
                                    bodyIndex=self.sim._bodies_idx[self.robot.body_name],
                                    jointIndex=self.sim.control_joint[i], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=qj[t, i],
                                    )                
            
            for i in range(self.base.get_num_joints()):
                self.sim.pc.setJointMotorControl2(
                bodyIndex=self.mount,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qb[t, i],
                # targetVelocity=v_0[t, i],
            )
            if t == 1000*self.traj.get_reach_duration-1:
                image = self.sim.get_camera_image()
                # time.sleep(3.0)
            self.sim.pc.stepSimulation()
            
            
        return image

                
    def start(self, record) -> None:
        qb, vb, ab, tb, qj, vj, aj, tj = self.traj.trajectory_generate_v2()
        self.robot.disable_motor()
        # self.object.create_mount_contraint()
        self.object.create_world_contraint()

        if record:
            self.sim.start_record(osp.join(self.record_dir, self.record_name))
        for t in range(1000*self.traj.get_reach_duration):
            for i in range(len(self.sim.control_joint)):
                self.sim.pc.setJointMotorControl2(
                                    bodyIndex=self.sim._bodies_idx[self.robot.body_name],
                                    jointIndex=self.sim.control_joint[i], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=qj[t, i],
                                    )                
            
            for i in range(self.base.get_num_joints()):
                self.sim.pc.setJointMotorControl2(
                bodyIndex=self.mount,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qb[t, i],
                # targetVelocity=v_0[t, i],
            )
            self.sim.pc.stepSimulation()
            if record:
                time.sleep(1.0 / 1000.0)
        
        self.object.remove_world_contraint()
        # rgb = self.sim.get_camera_image()
        # time.sleep(3.0)

        for t in range(1000*self.traj.get_reach_duration, 1000*self.traj.get_duration):
            for i in range(len(self.sim.control_joint)):
                self.sim.pc.setJointMotorControl2(
                                    bodyIndex=self.sim._bodies_idx[self.robot.body_name],
                                    jointIndex=self.sim.control_joint[i], 
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=qj[t, i],
                                    ) 
                
            for i in range(self.base.get_num_joints()):
                    self.sim.pc.setJointMotorControl2(
                    bodyIndex=self.mount,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=qb[t, i],
                )
            if t == 1000*self.traj.get_duration-1:
                contacts = self.sim.pc.getContactPoints(self.sim._bodies_idx[self.robot.body_name], self.sim._bodies_idx[self.object.body_name])
                if contacts:
                    contact_link = []
                    contact_position = []
                    contact_dist = []
                    for contact in contacts:
                        #print("contact", contact[8])
                        contact_dist.append(contact[8])
                        contact_link.append(contact[3])
                        contact_position.append(contact[6])
                        # contact_info = dict(contact_link = contact[3], contact_position = contact[6])
                    contact_info = dict(flag = True, contact_link = contact_link, contact_position = contact_position, contact_dist = contact_dist)
                
                else:
                    contact_info = dict(flag = False, contact_link = [], contact_position = [], contact_dist = [])


            self.sim.pc.stepSimulation()
            if record:
                time.sleep(1.0 / 1000.0)
            
        print("contact_info", contact_info["flag"])
        if record:
            self.sim.stop_record()
        return contact_info#, rgb
            

    def run_image(self, idx: int) -> None:
        self.sim._connect_()
        try:
            self.initialisation(self.data_fetcher[idx])
            self.load()
            image = self.start_image()
            self.remove_scene()
            self.image_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".png"

            self.save_image(image, self.image_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.remove_scene()
    
        self.sim.pc.disconnect()

    def screen_shot(self, idx: int) -> None:
        self.sim._connect_()
        try:
            self.initialisation(self.data_fetcher[idx])
            self.load_grsap_position()
            image = self.image_screenshot()
            self.image_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".png"
            self.save_image(image, self.image_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.remove_scene()
    
    def run_idx(self, idx: int, record: bool) -> None:
        self.sim._connect_()

        try:
            self.initialisation(self.data_fetcher[idx])
            self.record_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".mp4"
            self.load()
            contact_info = self.start(record)
            self.remove_scene()
            #self.save_image(im, self.image_name)
        except Exception as e:
            print(e)
            traceback.print_exc()
            contact_info = dict(flag = "Error", contact_link = [], contact_position = [])
            self.remove_scene()
        file_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".json"
        with open(osp.join(self.sim_result_dir,file_name), 'w') as f:
            json.dump(contact_info, f, indent=4)
    
        self.sim.pc.disconnect()

    def remove_scene(self) -> None:
        self.sim.remove_all()
        self.sim.reset_simulation()

        
    def run(self, record) -> None:
        # loader = DataLoader(self.data_fetcher, batch_size=1, shuffle=False)
        
        bar = tqdm(enumerate(self.data_fetcher), total=len(self.data_fetcher))
        for idx, data_pack in enumerate(bar):
            self.sim._connect_()
            try:
                self.initialisation(data_pack[1])
                self.record_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".mp4"
                self.image_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".png"

                self.load()
                contact_info = self.start(record)
                self.remove_scene()
                #self.save_image(im, self.image_name)
            except Exception as e:
                print(e)
                traceback.print_exc()
                contact_info = dict(flag = "Error", contact_link = [], contact_position = [], contact_dist = [])
                self.remove_scene()

            file_name = self.robot.body_name + "_" + str(self.mu) + "_" + str(idx) + ".json"
            with open(osp.join(self.sim_result_dir,file_name), 'w') as f:
                json.dump(contact_info, f, indent=4)
        
            self.sim.pc.disconnect()

    def run_refine(self, record, checker) -> None:
        # loader = DataLoader(self.data_fetcher, batch_size=1, shuffle=False)
        
        bar = tqdm(enumerate(checker), total=len(checker))
        for idx, data_pack in enumerate(bar):
            if data_pack[1] == 0:
                print("retake", idx, "simulating")
                self.run_idx(idx, record)
            else:
                print("skip", idx)
                continue
                            
def test():
    """
    test the basic motion planning asyn
    """
    rnd_init = np.zeros(3)
    rnd_target = np.array([0.5, 0.5, 0.5])
    rnd_via = rnd_target/2
    t0 = 0
    tf = 1
    rnd_init_joint = np.zeros(21)
    rnd_target_joint = np.random.rand(21)
    
    def cubic_spline_coefficients(q0, qf, v0, vf, t0, tf):
        """
        a0 + a1*t0 + a2*t0^2 + a3*t0^3 = q0
        """
        A = np.array([[1, t0, t0**2, t0**3],
                    [0, 1, 2*t0, 3*t0**2],
                    [1, tf, tf**2, tf**3],
                    [0, 1, 2*tf, 3*tf**2]])
        b = np.array([q0, v0, qf, vf])
        coefficients = np.linalg.solve(A, b)
        return coefficients   

    def cubic_spline_trajectory(coefficients, t):
        a0, a1, a2, a3 = coefficients
        q = a0 + a1*t + a2*t**2 + a3*t**3
        q_dot = a1 + 2*a2*t + 3*a3*t**2
        q_ddot = 2*a2 + 6*a3*t
        return q, q_dot, q_ddot


    def solve_cube_polynomial(coefficients, x, t_interval):
        """
        Solve the cubic polynomial for t given x.
        
        a0 + a1*t0 + a2*t0^2 + a3*t0^3 = x
        
        (a0-x) + a1*t0 + a2*t0^2 + a3*t0^3 = 0
        
        coes = [a3, a2, a1, a0-x]         
        
        
        """
        
        a0, a1, a2, a3 = coefficients
        new_coe = [a3, a2, a1, a0 - x]
        roots = np.roots(new_coe)
        real_roots = [t.real for t in roots if np.isreal(t) and t_interval[0] <= t.real <= t_interval[1]]
        print("roots", roots)
        if real_roots:
            return min(real_roots)
        else:
            return None
        
    for i in range(len(rnd_init)):
        coeffs = cubic_spline_coefficients(rnd_init[i], rnd_target[i], 0, 0, 0, 1)
        print("coeffs", coeffs)
        via_time = solve_cube_polynomial(coeffs, rnd_via[i], [0, 1])
        print("via_time", via_time)
                  
def main_():
    """5
        #2 load the result from the optimization record
        #3 load the object
        #4 reset the robot
        #5 start the simulation
            #6 move the robot to the target
            #7 set the object mass
            #8 record the data
        records.remove()
    
    """
    sim = PybulletBase()
    # rbt = shadow(sim)
    # rbt = allegro(sim)
    rbt = barrett(sim)
    # rbt = robotiq(sim)
    obj = objects(sim)
    base = MountingBase(sim, rbt)
    rbt.set_robot_info()

    re_orent = R.from_matrix(np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])).as_quat()
    # re_orent = R.from_matrix(np.eye(3)).as_quat()


    data_fetcher = ycb_opt_fetcher(robot_name=rbt.body_name,mu=0.9,data_sample=20,repeat=1)
    
    data_dict = data_fetcher[1]
    print(data_dict)
    obj_quat = data_dict["obj_quat"]
    obj_trans = data_dict["obj_trans"]
    obj_centre = data_dict["obj_centre"]
    
    rob_mat = data_dict["rob_mat"]
    rob_trans = data_dict["rob_trans"]
    
    print(rob_mat.shape, rob_trans.shape)
    pre_grasp = rbt.pre_grasp

    rob_glo_quat, rob_glo_trans = rbt.pre_transformation(rob_mat, rob_trans)
    
    rob_joints = data_dict["joint_val"]

    
    obj_file = data_dict["obj_file"]
    
    rob_glo_quat = quat_mult(re_orent, rob_glo_quat)
    obj_quat = quat_mult(re_orent, obj_quat)
    rob_glo_trans = rotate_vector(rob_glo_trans, re_orent)
    obj_trans_ = rotate_vector(obj_trans-obj_centre, re_orent)
    
    pre_grasp = rotate_vector(pre_grasp, rob_glo_quat)
    

    
    obj.set_object_info(file_path=obj_file, position=obj_trans_, orientation=obj_quat, obj_mass=0.1)
    
    obj.load_object(mu=0.9)
    
    sim.load_plane()

    obj.create_world_contraint()
    
    rbt.load_robot(base_position=rob_glo_trans+pre_grasp, base_orientation=rob_glo_quat, joints_val=rbt.init_joint, lateral_mu=0.9, Fixbase=False)
    rbt.disable_motor()
    m_id, c_id = base.mounting_gripper()
    
    # print("check inverse", reverse_rotate_mat(np.array([0.0, 0.0, 0.1]), base.get_base_orientation()) )

    traj = Trajectory(
        initial_positions=np.array([0.0, 0.0, 0.0]), 
        initial_joint=rbt.init_joint, 
        grasp_position = -1*rbt.pre_grasp,
        grasp_joint=rob_joints,
        lift_up = rotate_vector_inverse(np.array([0.0, 0.0, 0.3]-pre_grasp), base.get_base_orientation()))
        # lift_up = reverse_rotate_mat(np.array([0.0, 0.0, 0.1]), base.get_base_orientation()))
        # lift_up = np.array([0.1, 0.0, 0.1]),)
        # lift_up = rotate_vector(np.array([0.0, 0.0, 0.1]), obj_quat))
    
    # qb, vb, ab, tb, qg, vg, ag, tg = traj.trajectory_generate_v2()
    qb, vb, ab, tb, qg, vg, ag, tg = traj.trajectory_generate_v2()

    # print(qb.shape, vb.shape, ab.shape, tb.shape, qg.shape, vg.shape, ag.shape, tg.shape)
    # q_0, v_0, a_0, t_ = cubic_motion_planning(np.array([0.0, 0.0, 0.0]), move_0, 0, 1, 1000*1)
    

    for t in range(1000*traj.get_reach_duration):
    # while sim.pc.isConnected():
        # print("start control")
        for i in range(len(sim.control_joint)):
            sim.pc.setJointMotorControl2(
                                bodyIndex=sim._bodies_idx[rbt.body_name],
                                jointIndex=sim.control_joint[i], 
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=qg[t, i],
                                ) #rob_joints[i],)  #rbt.init_joint[i],
            
        # move_0 = sim.get_base_position(base.body_name) - rob_glo_trans 
        for i in range(base.get_num_joints()):
            sim.pc.setJointMotorControl2(
                bodyIndex=m_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qb[t, i],
                # targetVelocity=v_0[t, i],
            )

        # contacts = sim.pc.getContactPoints(sim._bodies_idx[rbt.body_name], sim._bodies_idx[obj.body_name])
        # if contacts:
        #     print("contact")
        #     for contact in contacts:

        #         print(f"🔥 Contact Detected! Link {contact[3]} - Force: {contact[9]} at Position: {contact[6]}")


        
        sim.pc.stepSimulation()
        time.sleep(1.0 / 1000.0)
        
    obj.remove_world_contraint()
    # print("base", rbt.get_base_position())
    # while sim.pc.isConnected():
        # sim.pc.stepSimulation()
        # time.sleep(1.0 / 1000.0)
    
    for t in range(1000*traj.get_reach_duration, 1000*traj.get_duration):
        for i in range(len(sim.control_joint)):
            sim.pc.setJointMotorControl2(
                                bodyIndex=sim._bodies_idx[rbt.body_name],
                                jointIndex=sim.control_joint[i], 
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=qg[t, i],
                                ) #rob_joints[i],)  #rbt.init_joint[i],
            
        # move_0 = sim.get_base_position(base.body_name) - rob_glo_trans 
        for i in range(base.get_num_joints()):
            sim.pc.setJointMotorControl2(
                bodyIndex=m_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qb[t, i],
                # targetVelocity=v_0[t, i],
            )


        sim.pc.stepSimulation()
        time.sleep(1.0 / 1000.0)
    
    # print("base", rbt.get_base_position())
    # print("base_base", base.get_base_position())
    # print("check inverse", rotate_vector_inverse(np.array([0.0, 0.0, 0.1]), base.get_base_orientation()))
    # print("check joint value", base.get_joint_angles())

def test_loading_robot():
    sim = PybulletBase()
    rbt = robotiq(sim)
    # rbt = shadow(sim) [-0.2, 0.0, -0.2]
    base = MountingBase(sim, rbt)
    sim.pc.setGravity(0, 0, 0)
    rbt.set_robot_info()
    rbt.load_robot(base_position=np.array([0.0, 0.0, 0.0]), base_orientation=np.array([0.0, 0.0, 0.0, 1.0]), joints_val=rbt.init_joint, lateral_mu=0.9, Fixbase=False)
    
    while sim.pc.isConnected():
        sim.pc.stepSimulation()
        # print(rbt.get_base_orientation())
        # print(rbt.get_base_position())
        time.sleep(1.0 / 1000.0)

def test_quaternion():
    q = np.array([0.0, 0.0, 0.7071, 0.7071])  # 90° rotation
    # print("Normalized Quaternion:", normalize_quaternion(q))

def test_Environment():
    for robot in ["Barrett", "Robotiq"]:#,"Allegro","Barrett","Robotiq"]:#, "Allegro", "Barrett"]:

        data_sample = 7
        # for mu in [0.9]:
        #     checker = sim_result_loader(robot=robot, mu=mu,)

        for mu in [0.9]:
            env = Environment(robot=robot, mu=mu, exp="genhand", repeat=1, render="GUI", data_sample=data_sample, test_sdf=True)
            env.run(record=False)
            # env.run_refine(record=False, checker=checker)
    # for mu in [0.1, 0.3, 0.5, 0.7, 0.9]:
    # for mu in [0.5,0.7,0.9]:
    #     env = Environment(robot="Allegro", mu=mu, exp="nv", repeat=1, render="DIRECT", data_sample=20)
    #     env.run(record=False)
    
    #TODO: Robotiq for gh

def viz_env():
    for idx_ in [3,4,13,24,33,34,46,52,55,56,78,81]:
        for robot in ["Shadow", "Allegro", "Barrett", "Robotiq"]:
        # for robot in ["Robotiq"]:
            if robot == "Shadow":
                data_sample = 5
                idx = idx_
            else:
                data_sample = 20
                i_1 = idx_//5
                i_2 = idx_%5
                idx = i_1*data_sample + i_2
                print("idx", idx)
            env = Environment(robot=robot, mu=0.5, exp="genhand", repeat=1, render="GUI", data_sample=data_sample)
            env.run_image(idx)

def viz_env_image():
    idx_= 2728
    for robot in ["Shadow", "Allegro", "Barrett", "Robotiq"]:
    # for robot in ["Robotiq"]:

        env = Environment(robot=robot, mu=0.9, exp=None, repeat=1, render="GUI", data_sample=None, task="test")
        env.run_image(idx_)

def idx():
    idx = 251
    for robot in ["Barrett"]:
        if robot == "Shadow":
            data_sample = 5
            # idx = idx_
        else:
            data_sample = 20
            # i_1 = idx_//5
            # i_2 = idx_%5
            # idx = i_1*data_sample + i_2
        env = Environment(robot=robot, mu=0.9, exp="genhand", repeat=1, render="GUI", data_sample=data_sample, task="test")
        env.run_idx(idx, record=False)


def simulation_worker(robot,mu,data_sample, exp, q):
    try:
        env = Environment(robot=robot, mu=mu, exp= exp, repeat=1, render="DIRECT", data_sample=data_sample, task = 'test', test_sdf=False)
        env.run(record=False)
        q.put((robot, mu, exp, 'Done'))
    except Exception as e:
        q.put((robot, mu, exp, f"❌ Error: {str(e)}"))
        
def run_simulation_mp():
    robot = ["Shadow","Barrett", "Robotiq","Allegro"]#"Shadow",,"Barrett", "Robotiq"
    mus = [0.9]
    data_sample = 5
    cores = 10
    exp = ["genhand"] #genhand
    processes = []
    q = Queue()
    task_args = list(product(robot, mus, exp))
    for robot, mu, exp in task_args:
        p = Process(target=simulation_worker, args=(robot, mu, data_sample, exp, q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    while not q.empty():
        robot, mu, exp, status = q.get()
        print(f"Robot: {robot}, Mu: {mu}, Exp: {exp}, Status: {status}")
    

if __name__ == "__main__":
    run_simulation_mp()
    # idx()
    # test_Environment()
    # viz_env_image()
    # viz_env()
    
    # test_loading_robot()
    # main()
    # test_quaternion()
    # test()
    # from manopth.rodrigues_layer import batch_rodrigues   
    # rand = np.random.rand(3)
    # rand_tensor = torch.tensor(rand, dtype=torch.float32)
    # print(rand)
    # print(rand_tensor)
    # print(rodrigues(rand))
    # print(batch_rodrigues(rand_tensor.unsqueeze(0)))