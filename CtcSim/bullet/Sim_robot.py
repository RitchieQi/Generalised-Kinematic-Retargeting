from env_base import PBenv_base, PBrobot_base, PBtask_base
from pybullet_base import PybulletBase
import numpy as np
import os
osp = os.path
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
import pybullet as p

class sr_hand(PBrobot_base):
    def __init__(
            self,
            sim: PybulletBase,
            base_position: np.ndarray,
            base_orientation: np.ndarray,
            finger_friction: float,
    ) -> None:
        """ Shadow Robot hand.

        Args:
            sim (PybulletBase): The Pybullet simulation.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
            base_orientation (np.ndarray): The orientation of the robot, as (x, y, z, w).
            finger_friction (float): The friction coefficient of the fingers.
        """
        self.controllable_joints = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28])
        self.fingertip_joints = np.array([7, 12, 17, 23, 29])
        self.fingertip_links = np.array([7, 12, 17, 23, 29])
        self.motion = None
        self.force = None
        super().__init__(
            sim=sim,
            body_name='sr_hand',
            file_name = osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'sr_description','urdf','shadow_hand.urdf'),
            base_position=base_position,
            base_orientation=base_orientation,
            joint_indices = self.controllable_joints,
            joint_forces= np.array([150] * 24),

        )
    
    def setup(self) -> None:
        #self.set_joint_angles(joint_angle)
        pass
        
    def set_action(self, target, mode) -> None:
        if mode == "motion":

            self.motion_control_expilict(target)
            
    def reset(self) -> None:
        
        pass
        
    
    # def set_action(self, target_position: np.ndarray, target_force: np.ndarray) -> None:
    #     """
    #     customise the action space for the robot

    #     Args:
    #         target_position (np.ndarray): The target position of the robot, as (x, y, z).
    #         target_force (np.ndarray): The target force of the robot, as (x, y, z).
    #     """
    #     motion_command = []
    #     for p in target_position:
    #         motion_command.append(p)
    #     p = np.array(motion_command)
    #     force_command = []
    #     for f in target_force:
    #         force_command.append(f)
    #     f = np.array(force_command)
    #     self.motion = p
    #     self.force = f
        
        
    # def get_observation(self, body_id: int) -> np.ndarray:
    #     """Get the observation of the robot.

    #     Args:
    #         body_id (int): The body index for observation.

    #     Returns:
    #         np.ndarray: The observation of the robot.
    #     """
    #     return self.get_observation(body_id)

    # def reset(self, position: np.ndarray, orientation: np.ndarray, joints_val: np.ndarray) -> None:
    #     """Initialize the robot.

    #     Args:
    #         position (np.ndarray): The position of the robot, as (x, y, z).
    #         orientation (np.ndarray): The orientation of the robot, as (x, y, z, w).
    #         joints_val (np.ndarray): The initial joint values.
    #     """
    #     self.reset(position=position, orientation=orientation, joint_angles=joints_val)

    # def motion_control_expilict(self, target_angles: np.ndarray, Kp: float = 3, Kd: float = 1) -> None:
    #     """Control the robot to reach the target angles by Inverse Dynamics Control/Computed Torque Control.

    #     Args:
    #         target_angles (np.ndarray): The target joint angles.
        
    #     Reference: 11.37 of the book "Modern Robotics: Mechanics, Planning, and Control" by Kevin M. Lynch and Frank C. Park
        
    #     """
    #     self.motion_control_expilict(target_angles, Kp, Kd)

    
    # def motion_control_simplified(self, target_angles: np.ndarray, Kp: float = 3, Kd: float = 1) -> None:
    #     """Control the robot to reach the target angles by Inverse Dynamics Control/Computed Torque Control.

    #     Args:
    #         target_angles (np.ndarray): The target joint angles.
    #     """
    #     self.motion_control_simplified(target_angles, Kp, Kd)
            
    # def force_control(self, target_forces: np.ndarray, Kp: float, Ki: float) -> None:
    #     """Control the robot to reach the target forces.

    #     Args:
    #         link (np.ndarray): The link index for jacobian calculation.
    #         joints (np.ndarray): The joint indices for force sensing.
    #         target_forces (np.ndarray): The wrenches in cartesian space.

    #     #TODO: add velocity damping term!!
    #     """
    #     self.force_control(self.fingertip_joints, self.fingertip_links, target_forces, Kp, Ki)
            
            

if __name__ == "__main__":
    #from pybullet_base import PybulletBase
    sim = PybulletBase(connection_mode="GUI", bg_color=np.array([255, 255, 255]))
    base_position = np.array([0, 0, 0])
    base_orientation = np.array([0.5, -0.5, 0.5, -0.5])
    friction = 0.5
    robot = sr_hand(sim, base_position, base_orientation, friction)
    robot.set_action(np.ones(24), "motion")
    #robot.motion_control_simplified(np.ones(24))
    
    
        


