import os
osp = os.path
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc
from contextlib import contextmanager

class PybulletBase:
    '''
    Pybullet wrapper class
    '''
    def __init__(
            self,
            connection_mode: str = 'DIRECT',
            render_mode: str = None,
            n_substeps: int = 20,
            bg_color: Optional[np.ndarray] = None,
    ):
        self.connection_mode = p.DIRECT if connection_mode == 'DIRECT' else p.GUI
        self.render_mode = render_mode
        self.n_substeps = n_substeps
        bg_color = bg_color if bg_color is not None else np.array([223.0, 54.0, 45.0])
        self.bg_color = bg_color.astype(np.float32) / 255.0
        options = '--background_color_red={} --background_color_green={} --background_color_blue={}'.format(
            self.bg_color[0], self.bg_color[1], self.bg_color[2]
        )
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pd.getDataPath())
        self.physics_client.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}       

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self) -> None:
        """Step the simulation."""
        self.physics_client.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        if self.physics_client.isConnected():
            self.physics_client.disconnect()

    def save_state(self) -> int:
        """Save the current simulation state.

        Returns:
            int: A state id assigned by PyBullet, which is the first non-negative
            integer available for indexing.
        """
        return self.physics_client.saveState()

    def restore_state(self, state_id: int) -> None:
        """Restore a simulation state.

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.restoreState(state_id)

    def remove_state(self, state_id: int) -> None:
        """Remove a simulation state. This will make this state_id available again for returning in save_state().

        Args:
            state_id: The simulation state id returned by save_state().
        """
        self.physics_client.removeState(state_id)

    def render(
        self,
        width: int = 720,
        height: int = 480,
        target_position: Optional[np.ndarray] = None,
        distance: float = 1.4,
        yaw: float = 45,
        pitch: float = -30,
        roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Args:
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Roll of the camera. Defaults to 0.
            mode (str, optional): Deprecated: This argument is deprecated and will be removed in a future
                version. Use the render_mode argument of the constructor instead.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if self.render_mode == "rgb_array":
            target_position = target_position if target_position is not None else np.zeros(3)
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, rgba, _, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                shadow=True,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            # With Python3.10, pybullet return flat tuple instead of array. So we need to build create the array.
            rgba = np.array(rgba, dtype=np.uint8).reshape((height, width, 4))
            return rgba[..., :3]

    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)

    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as quaternion (x, y, z, w).
        """
        orientation = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[1]
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
            rotation = self.physics_client.getEulerFromQuaternion(quaternion)
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
        velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)

    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.physics_client.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]
    
    def get_joint_angles(self, body: str, joints: np.ndarray) -> np.ndarray:
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
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[1]
    
    def get_joint_velocities(self, body: str, joints: np.ndarray) -> np.ndarray:
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
            orientation = self.physics_client.getQuaternionFromEuler(orientation)
        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )
    
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
        joint_state = self.physics_client.calculateInverseKinematics(
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
        return self.physics_client.getNumJoints(self._bodies_idx[body])
    
    def inverse_dynamics(self, body: str, joint_positions: np.ndarray, joint_velocities: np.ndarray, desired_joint_accelerations: np.ndarray) -> np.ndarray:
        """Compute the inverse dynamics and return the joint forces.

        Args:
            body (str): Body unique name.
            joint_positions (np.ndarray): Joint positions.
            joint_velocities (np.ndarray): Joint velocities.
            desired_joint_accelerations (np.ndarray): Desired joint accelerations.

        Returns:
            np.ndarray: The joint forces.
        """
        joint_state = self.physics_client.calculateInverseDynamics(
            bodyUniqueId=self._bodies_idx[body],
            objPositions=joint_positions,
            objVelocities=joint_velocities,
            objAccelerations=desired_joint_accelerations,
        )
        return np.array(joint_state)

    def get_jacobian(self, body: str, link: int, joints: np.ndarray) -> np.ndarray:
        """Get the Jacobian of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            np.ndarray: The Jacobian.
        """
        joint_positions = self.get_joint_angles(body, joints)
        link_position = self.get_link_position(body, link)
        zero_vec = np.array([0.0]*len(joint_positions))
        jacobian_translation, jacobian_rotation = self.physics_client.calculateJacobian(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            localPosition=link_position,
            objPositions=joint_positions,
            objVelocities=zero_vec,
            objAccelerations=zero_vec)
        return np.vstack((jacobian_translation, jacobian_rotation))

    def get_dynamics_matrices(self, body: str, joints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the mass matrix, h matrix (which is sum of coriolis and gravity), and gravity vector.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The mass matrix, h matrix, and gravity vector.
        """
        #num_joints = len(joints)
        joint_positions = self.get_joint_angles(body, joints)
        joint_velocities = self.get_joint_velocities(body, joints)
        zero_vec = np.array([0.0]*len(joint_positions))

        # core dump error
        mass_matrix = self.physics_client.calculateMassMatrix(
            bodyUniqueId=self._bodies_idx[body],
            objPositions=joint_positions,
            )
        
        
        gravity_matrix = self.physics_client.calculateInverseDynamics(
            bodyUniqueId=self._bodies_idx[body],
            objPositions=joint_positions,
            objVelocities=zero_vec,
            objAccelerations=zero_vec
            )        


        h_matrix = self.physics_client.calculateInverseDynamics(
            bodyUniqueId=self._bodies_idx[body],
            objPositions=joint_positions,
            objVelocities=joint_velocities,
            objAccelerations=zero_vec
            )
        return np.array(mass_matrix), np.array(h_matrix), np.array(gravity_matrix)

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def load_obj_as_mesh(self, body_name: str, obj_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Load obj file and create mesh/collision shape from it.
            ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createVisualShape.py
        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            obj_path (str): Path to the obj file.
        """
        obj_mass = 1
        obj_visual_shape_id = self.physics_client.createVisualShape(
                                                    shapeType=p.GEOM_MESH, 
                                                    fileName=obj_path, 
                                                    rgbaColor=[1, 1, 1, 1],
                                                    meshScale=[1, 1, 1])

        obj_collision_shape_id = self.physics_client.createCollisionShape(
                                                shapeType = p.GEOM_MESH,
                                                fileName = obj_path,
                                                flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                                p.GEOM_CONCAVE_INTERNAL_EDGE,
                                                meshScale=[1, 1, 1])
        
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
                                baseMass=obj_mass,
                                baseInertialFramePosition=[0, 0, 0],
                                baseCollisionShapeIndex=obj_collision_shape_id,
                                baseVisualShapeIndex=obj_visual_shape_id,
                                basePosition=position,
                                baseOrientation=orientation,
                                useMaximalCoordinates=True)

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )

    def disable_motor(self, body: str, joint: np.ndarray, force_lim: float = 0.0) -> None:
        """Disable the motor of a joint.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
        """
        zero_vec = [force_lim]*len(joint)
        self.physics_client.setJointMotorControlArray(
            bodyUniqueId=self._bodies_idx[body],
            jointIndices=joint,
            controlMode=p.VELOCITY_CONTROL,
            forces=zero_vec,
        )
    
    def remove_body(self, body: str) -> None:
        """Remove the body.

        Args:
            body (str): Body unique name.
        """
        self.physics_client.removeBody(self._bodies_idx[body])

    def torque_control(self, body: str, joint: np.ndarray, forces: np.ndarray) -> None:
        """Apply forces to the joints.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            forces (np.ndarray): Forces to apply.
        """
        #self.disable_motor(body, joint, joint_fric)
        self.physics_client.setJointMotorControlArray(
            bodyUniqueId=self._bodies_idx[body],
            jointIndex=joint,
            controlMode=p.TORQUE_CONTROL,
            forces=forces,
        )

    def enable_force_sensor(self, body: str, joints: np.ndarray) -> None:
        """Enable the force sensor for the link.

        Args:
            body (str): Body unique name.
            joints (np.ndarray) : Link index in the body.
        """
        for joint in joints:
            self.physics_client.enableJointForceTorqueSensor(self._bodies_idx[body], joint, True)
    
    def get_joint_force_sensing(self, body: str, joint: int) -> np.ndarray:
        """Get the force sensing data for the joint.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.

        Returns:
            np.ndarray: The force sensing data.
        """
        return np.array(self.physics_client.getJointState(self._bodies_idx[body], joint)[2]) 
    
    def get_joints_force_sensing(self, body: str, joints: np.ndarray) -> np.ndarray:
        """Get the force sensing data for the joints.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.

        Returns:
            np.ndarray: The force sensing data.
        """
        return np.array(self.physics_client.getJointStates(self._bodies_idx[body], joints)[2])

if __name__ == '__main__':
    p = PybulletBase(connection_mode=p.GUI)