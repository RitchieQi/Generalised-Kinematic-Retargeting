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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

import time
class PybulletBase:
    def __init__(
        self,
        connection_mode: str = "GUI",
    ):
        self.connect = p.DIRECT if connection_mode == "DIRECT" else p.GUI
        self.pc = bc.BulletClient(connection_mode=self.connect)
        self.pc.setAdditionalSearchPath(pd.getDataPath())
        self.pc.loadURDF("plane.urdf", [0, 0, -0.5], useFixedBase=True)
        self.pc.setTimeStep(1 / 1000)
        self.pc.setGravity(0, 0, -9.8)
        self._bodies_idx = {}
        self.pc.setRealTimeSimulation(False)
        self.n_steps = 0
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
        
    def load_obj_as_mesh(self, body_name: str, obj_path: str, position: np.ndarray, orientation: np.ndarray, obj_mass: float=0.0) -> None:
        """Load obj file and create mesh/collision shape from it.
            ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createVisualShape.py
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            obj_path (str): Path to the obj file.
        """
        obj_visual_shape_id = self.pc.createVisualShape(
                                                    shapeType=p.GEOM_MESH, 
                                                    fileName=obj_path, 
                                                    rgbaColor=[1, 1, 1, 1],
                                                    meshScale=[1, 1, 1])

        obj_collision_shape_id = self.pc.createCollisionShape(
                                                shapeType = p.GEOM_MESH,
                                                fileName = obj_path,
                                                flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                                p.GEOM_CONCAVE_INTERNAL_EDGE,
                                                meshScale=[1, 1, 1])
        
        self._bodies_idx[body_name] = self.pc.createMultiBody(
                                baseMass=obj_mass,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=obj_collision_shape_id,
                                baseVisualShapeIndex=obj_visual_shape_id,
                                basePosition=position,
                                baseOrientation=orientation,
                                useMaximalCoordinates=True)

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
        for i in self.control_joint:
            self.pc.setJointMotorControl2(
                bodyIndex=self._bodies_idx[body],
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0,
                force=0,
            )
            

class robot(ABC):
    def __init__(
        self,
        sim: PybulletBase,
        FixedBase: bool = True,
    )-> None:
        self.sim = sim
        self.body_name = None
        self.urdf_path = None
        self.base_position = None
        self.base_orientation = None
        self.endeffect = None
        self.init_joint = None
        self.calibration_rot = None
        self.calibration_trans = None
        self.set_robot_info()
        self.load_robot(FixedBase)
        self.fixedbase = FixedBase
        
    @abstractmethod
    def set_robot_info(self) -> None:
        pass
    
    def load_robot(self, FixedBase) -> None:
        if self.urdf_path is None:
            self.set_robot_info()
        
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=self.urdf_path,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=FixedBase,
        )
        
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
    

class mountingBase():
    def __init__(
        self,
        sim: PybulletBase,
        Position: List[float],
        Orientation: List[float],
    ) -> None:
        self.sim = sim
        self.body_name = "xyz_base"
        urdf_path = osp.join(osp.dirname(__file__), 'xyz.urdf')
        self.sim.loadURDF(
            urdf_path,
            [0, 0, 0],
            useFixedBase=True,
        )
        self.num_joints = self.sim.get_num_joints(self.body_name)
        
        self.sim.setJointMotorControlArray(
            self.body_name,
            list(range(self.num_joints)),
            self.sim.POSITION_CONTROL,
            targetPositions=[0] * self.num_joints,
            forces=[1000] * self.num_joints,
        )
        self.JOINT_TYPES = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

        joint_info = {}
        for joint in range(self.num_joints):
            joint = self.parse_joint_info(self.body_name, joint)
            joint_info[joint['link_name']] = joint
        self.joint_info = joint_info
    def parse_joint_info(self, body_id: int, joint_id: int) -> Dict[str, Any]:
        info = self.sim.getJointInfo(body_id, joint_id)
        joint_info = {
            'id': info[0],
            'link_name': info[12].decode("utf-8"),
            'joint_name': info[1].decode("utf-8"),
            'type': self.JOINT_TYPES[info[2]],
            'friction': info[7],
            'lower_limit': info[8],
            'upper limit': info[9],
            'max_force': info[10],
            'max_velocity': info[11],
            'joint_axis': info[13],
            'parent_pos': info[14],
            'parent_orn': info[15]
        }
        return joint_info
    
    @property
    def ee_link_id(self) -> int:
        return self.joint_info['end_effector_link']['id']




if __name__ == "__main__":
    def sort_joint(body: int, joint: int):
        Dof = 0
        q_i = []

        for i in range(joint):
            joint_info = p.getJointInfo(body, i)
            print(joint_info)
            if joint_info[2] == 0:
                Dof += 1
                q_i.append(i)
        return Dof, q_i
    def parse_joint_info(body_id: int, joint_id: int) -> Dict[str, Any]:
        info = p.getJointInfo(body_id, joint_id)
        joint_info = {
            'id': info[0],
            'link_name': info[12].decode("utf-8"),
            'joint_name': info[1].decode("utf-8"),
            'friction': info[7],
            'lower_limit': info[8],
            'upper limit': info[9],
            'max_force': info[10],
            'max_velocity': info[11],
            'joint_axis': info[13],
            'parent_pos': info[14],
            'parent_orn': info[15]
        }
        return joint_info
    def mounting_gripper(gripper_id: int):
        pose_mount, quat_mount = p.getBasePositionAndOrientation(gripper_id)
        #mount = mountingBase(p, pose_mount, quat_mount)
        urdf_path = osp.join(osp.dirname(__file__), 'xyz.urdf')
        mount = p.loadURDF(urdf_path, basePosition=pose_mount, baseOrientation=quat_mount, useFixedBase=True)
        num_joints = p.getNumJoints(mount)
        joint_info = {} 
        for joint in range(num_joints):
            joint_ = parse_joint_info(mount, joint)
            joint_info[joint_['link_name']] = joint_
            
        constraint_id = p.createConstraint(mount, joint_info['end_effector_link']['id'], gripper_id, -1, 
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1], childFrameOrientation=[0, 0, 0, 1]
        )
        
        return mount, constraint_id
    
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pd.getDataPath())
    urdf_path = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'barrett_adagrasp','model.urdf')
    sdID = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=False)
    p.changeDynamics(sdID, -1, mass=1.0)

    
    mounted_gripper, c_id = mounting_gripper(sdID)
    # sdID = mounted_gripper
    
    constraint_info = p.getConstraintInfo(c_id)
    print("constraint info",constraint_info)
    
    
    desired_position = [0.5, 0, 0]
    desired_orientation = p.getQuaternionFromEuler([0, 0, 1.57])
    
    
    num_joints = p.getNumJoints(mounted_gripper) # 30 no matter the hand is fixed or not
    print("num_j",num_joints)
    
    _,_ = sort_joint(sdID, p.getNumJoints(sdID))
    dof, q_i = sort_joint(mounted_gripper, p.getNumJoints(mounted_gripper))
    print(dof, q_i)
    """
    fixed:   24 [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]
    floating:24 [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]
    
    """
    # jointStates = p.getJointStates(sdID, q_i)
    # q1 = []
    # for i in range(len(q_i)):
    #     q1.append(jointStates[i][0])
    # q = np.array(q1)
    # M = p.calculateMassMatrix(sdID, q1)
    # print(M)
    
    # base_joint = p.createConstraint(sdID, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])

    while True:
        # p.changeConstraint(base_joint, desired_position, desired_orientation)
        for i in range(p.getNumJoints(mounted_gripper)):
            p.setJointMotorControl2(bodyIndex=mounted_gripper,
                                    jointIndex=i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=desired_position[i],
                                    )
            # print(p.getJointState(mounted_gripper, i))    
        p.stepSimulation()
        time.sleep(1/240)
        
    p.disconnect()
    