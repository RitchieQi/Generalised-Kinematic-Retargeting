from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from pybullet_base import PybulletBase
class PBrobot_base(ABC):
    """Base class for robots.


    """
    def __init__(
            self,
            sim: PybulletBase,
            body_name: str,
            file_name: str,
            base_position: np.ndarray,
            base_orientation: np.ndarray,
            joint_indices: np.ndarray,
            joint_forces: np.ndarray,
    ) -> None:
        self.sim = sim
        self.body_name = body_name
        with self.sim.no_rendering():
            self._load_robot(file_name, base_position, base_orientation)
            self.setup()

        self.joint_indices = joint_indices
        self.joint_forces = joint_forces
        self.joint_friction = np.array([0.01] * len(joint_indices))
    
    def _load_robot(self, file_name: str, base_position: np.ndarray, base_orientation: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): The URDF file name of the robot.
            base_position (np.ndarray): The position of the robot, as (x, y, z).
            base_orientation (np.ndarray): The orientation of the robot, as (x, y, z, w).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True,
        )
    
    @abstractmethod
    def setup(self) -> None:
        """Called after robot loading."""
        
    
    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action.

        Args:
            action (np.ndarray): The action to set.
        """
        

    # @abstractmethod
    # def get_observation(self) -> np.ndarray:
    #     """Get the observation.

    #     Returns:
    #         np.ndarray: The observation.
    #     """
        

    # @abstractmethod
    # def reset(self) -> None:
    #     """Reset the robot."""
    
    def step(self) -> None:
        """Step the robot."""
        self.sim.step()
    
    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_velocity(self, link: int) -> np.ndarray:
        """Returns the velocity of a link as (vx, vy, vz)

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: Velocity as (vx, vy, vz)
        """
        return self.sim.get_link_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): The joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): The joint index.

        Returns:
            np.ndarray: Joint velocity as (wx, wy, wz)
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def get_joint_angles(self) -> np.ndarray:
        """Returns the angles of all joints.

        Returns:
            np.ndarray: Joint angles
        """
        return self.sim.get_joint_angles(self.body_name, self.joint_indices)

    def get_joint_velocities(self) -> np.ndarray:
        """Returns the velocities of all joints.

        Returns:
            np.ndarray: Joint velocities
        """
        return self.sim.get_joint_velocities(self.body_name, self.joint_indices)

    def motion_control(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): The target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=target_angles,
            forces=self.joint_forces,
        )
    
    def torque_control(self, target_forces: np.ndarray) -> None:
        """Control the joints of the robot with forces.
        
        Args:
            target_forces (np.ndarray): The target forces. The length of the array must equal to the number of joints.
        """
        self.sim.torque_control(
            body=self.body_name,
            joints=self.joint_indices,
            forces=target_forces,
            joint_fric=self.joint_friction,
        )


    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (list): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)

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

    def get_jacobian(self, link: int) -> np.ndarray:
        """Returns the Jacobian of the robot.

        Args:
            link (int): The link index.

        Returns:
            np.ndarray: The Jacobian.
        """
        return self.sim.get_jacobian(self.body_name, link)

    def inverse_dynamics(self, positions: np.ndarray, velocities:np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        """Compute the inverse dynamics.

        Args:
            link (int): The link index.
            acceleration (np.ndarray): The acceleration.

        Returns:
            np.ndarray: The torques.
        """
        return self.sim.inverse_dynamics(self.body_name, positions, velocities, acceleration)
    
    def disable_motor(self) -> None:
        """Disable the motor."""
        self.sim.disable_motor(body=self.body_name, joint=self.joint_indices)
    
    def reset(self, position: np.ndarray, orientation: np.ndarray, joint_angles: np.ndarray) -> None:
        """Reset the robot.

        Args:
            position (np.ndarray): The position of the robot, as (x, y, z).
            orientation (np.ndarray): The orientation of the robot, as (x, y, z, w).
        """
        self.sim.set_base_pose(self.body_name, position, orientation)
        self.sim.set_joint_angles(self.body_name, self.joint_indices, joint_angles)

    def get_dynamics_matrices(self, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the dynamics matrices.

        Args:
            joints (np.ndarray): The controlable joints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The mass matrix, the h matrix, and the gravity matrix.
        """
        return self.sim.get_dynamics_matrices(self.body_name, joints)
    
    def enable_force_sensor(self, joints: np.ndarray) -> None:
        """Enable the force sensor.

        Args:
            joints (np.ndarray): The joints to enable the force sensor.
        """
        self.sim.enable_force_sensor(self.body_name, joints)

    def get_joints_force_sensing(self, joints: np.ndarray) -> np.ndarray:
        """Returns the forces sensed by the joints.

        Args:
            joints (np.ndarray): The joints to sense the force.

        Returns:
            np.ndarray: The sensed forces.
        """
        return self.sim.get_joints_force_sensing(self.body_name, joints)
    
    def get_joint_force_sensing(self, joint: int) -> np.ndarray:
        """Returns the force sensed by the joint.

        Args:
            joint (int): The joint to sense the force.

        Returns:
            np.ndarray: The sensed force.
        """
        return self.sim.get_joint_force_sensing(self.body_name, joint)
    
    def motion_control_expilict(self, target_angles: np.ndarray, Kp: float = 3, Kd: float = 1) -> None:
        """Control the robot to reach the target angles by Inverse Dynamics Control/Computed Torque Control.

        Args:
            target_angles (np.ndarray): The target joint angles.
        
        Reference: 11.37 of the book "Modern Robotics: Mechanics, Planning, and Control" by Kevin M. Lynch and Frank C. Park
        
        """
        self.disable_motor()
        current_angles = self.get_joint_angles()
        angles_traj, vels_traj, accs_traj, time_steps = self.cubic_motion_planning(current_angles, target_angles, 0, 1)
        for i in range(len(time_steps)):
            self.step()
            current_angles = self.get_joint_angles()
            current_vels = self.get_joint_velocities()
            M, h, G = self.get_dynamics_matrices(self.joint_indices)
            error = angles_traj[i] - current_angles
            error_dot = vels_traj[i] - current_vels
            error_dot_dot = accs_traj[i] 
            Error = Kp * error + Kd * error_dot + error_dot_dot
            control = np.dot(M, Error) + h
            self.torque_control(control)

    def motion_control_simplified(self, target_angles: np.ndarray, Kp: float = 3, Kd: float = 1) -> None:
        """Control the robot to reach the target angles by Inverse Dynamics Control/Computed Torque Control.

        Args:
            target_angles (np.ndarray): The target joint angles.
        """
        self.disable_motor()
        current_angles = self.get_joint_angles()
        angles_traj, vels_traj, accs_traj, time_steps = self.cubic_motion_planning(current_angles, target_angles, 0, 1)
        for i in range(len(time_steps)):
            current_angles = self.get_joint_angles()
            current_vels = self.get_joint_velocities()
            error = angles_traj[i] - current_angles
            error_dot = vels_traj[i] - current_vels
            error_dot_dot = accs_traj[i]
            control = Kp * error + Kd * error_dot + error_dot_dot
            tau = self.inverse_dynamics(current_angles, current_vels, control)
            self.torque_control(tau)
            self.step()
    
    def force_control(self, joints: np.ndarray, links: np.ndarray, target_forces: np.ndarray, Kp: float, Ki: float) -> None:
        """Control the robot to reach the target forces.

        Args:
            link (np.ndarray): The link index for jacobian calculation.
            joints (np.ndarray): The joint indices for force sensing.
            target_forces (np.ndarray): The wrenches in cartesian space.

        #TODO: add velocity damping term!!
        """
         
        self.disable_motor()
        self.enable_force_sensor(joints)
        num_targets = len(joints)

        M, h, G = self.get_dynamics_matrices(self.joint_indices)
        F_i = np.zeros((num_targets,6))
        #initial_wrench = np.array([0., 0., 0., 0., 0., 0.]*num_targets)
        for i,j in enumerate(joints):
            J = self.get_jacobian(links[i])
            F_tip = self.get_joint_force_sensors(j)
            F_e = target_forces[i] - F_tip
            F_d = target_forces[i]
            F_i[i] = F_i[i] + F_e* 1./240  # default time step
            current_vels = self.get_joint_velocities()
            vels_norm = np.linalg.norm(current_vels)
            Error = F_d + Kp * F_e + Ki * F_i[i] 
            damping = (Error*vels_norm)/(vels_norm + 1e-6) 
            Total_error = Error - damping
            tau = np.dot(J.T, Total_error) + G
            self.torque_control(tau)
            self.step()
    
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
    
class PBtask_base(ABC):
    """Base class for tasks.

    """
    def __init__(
            self,
            sim: PybulletBase,

    ) -> None:
        self.sim = sim    
    def _load_task(self, body_name, file_name: str, base_position: np.ndarray, base_orientation: np.ndarray) -> None:
        """Load the object
        """
        self.sim.load_obj_as_mesh(
            body_name=body_name,
            fileName=file_name,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
    
    def setup(self) -> None:
        """Called after task loading."""
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Get the observation.

        Returns:
            np.ndarray: The observation.
        """
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Get the achieved goal.

        Returns:
            np.ndarray: The achieved goal.
        """

    def get_obj_position(self) -> np.ndarray:
        """Returns the position of the object as (x, y, z)

        Returns:
            np.ndarray: Position as (x, y, z)
        """
        return self.sim.get_base_position(self.body_name)
    
    def get_base_orientation(self) -> np.ndarray:
        """Returns the orientation of the object as (x, y, z, w)

        Returns:
            np.ndarray: Orientation as (x, y, z, w)
        """
        return self.sim.get_obj_orientation(self.body_name)
    
    def remove_obj(self, body_name: str) -> None:
        """Remove the object."""
        self.sim.remove_body(body_name)

class PBenv_base(ABC):
    def __init__(
            self,
            robot: PBrobot_base,
            task: PBtask_base,
            render_target_position: Optional[np.ndarray] = None,
            render_distance: float = 1.4,
            render_yaw: float = 45,
            render_pitch: float = -30,
            render_roll: float = 0,
    ):
        assert robot.sim == task.sim
        self.sim = robot.sim
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0.0, 0.0, 0.0])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll
        with self.sim.no_rendering():
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )
    @abstractmethod
    def _get_obs(self) -> Dict[str, Any]:
        """Get the observation.

        Returns:
            Dict[str, Any]: The observation.
        """
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment.

        Returns:
            Dict[str, Any]: The observation.
        """
        pass
    
    def close(self):
        """Close the environment."""
        self.sim.close()
    
    
    @abstractmethod
    def step(self, action: np.ndarray):
        """Step the environment.

        Args:
            action (np.ndarray): The action.

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: The observation, the reward, whether the episode is done, and additional information.
        """
        pass