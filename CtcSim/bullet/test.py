import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
import os.path as osp
import numpy as np
from typing import List, Optional, Tuple, Union, Iterator, Any, Dict

"""
the process of setting up the simulation environment

1. connect to the client
    eg: p.connect(p.GUI)
2.load the stuff: plane, robot, cubes, etc.
    eg: p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
3. set the initial state of the robot
    eg: p.resetJointState(kukaId, i, rp[i])
4. start the simulation
    p.stepSimulation()
    while 1:
        p.setJointMotorControl2(kukaId, 1, p.POSITION_CONTROL, targetPosition=0, force=500)
"""

def disable_motor(body: int, joint: List[int]):
    """Disable the motor of a joint.

    Args:
        body (str): Body unique name.
        joint (int): Joint index in the body.
    """
    for i in joint:
        p.setJointMotorControl2(
            bodyIndex=body,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,
            force=0.0,
        )

def disable_motor2(body: int, joint: List[int]):
    """Disable the constraint-based motor of a joint.
    
    Args:
        body (str): Body unique name.
        joint (int): Joint index in the body.
    """
    for i in joint:
        p.setJointMotorControl2(
            bodyIndex=body,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0,
            force=0.0, 
        )

class cubic_spline:
    def __init__(self, pd):
        self._pd = pd
    
    def get_traj(self, bodyUniqueId, jointIndices, target_positions, t0, time_step, num_points):
        jointStates = self._pd.getJointStates(bodyUniqueId, jointIndices)
        q1 = []
        for i in range(len(jointIndices)):
            q1.append(jointStates[i][0])
        q = np.array(q1)
        q_desired = np.array(target_positions)
        q_t, q_dot_t, q_ddot_t, time_steps = cubic_motion_planning(q, q_desired, t0, t0 + time_step, num_points)
        return q_t, q_dot_t, q_ddot_t, time_steps

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

def get_joint_positions(robot_id: int, num_joints: int) -> List[float]:
    """
    Get the joint positions of the robot.
    
    robot_id: The robot's body index.
    num_joints: The number of joints of the robot.
    
    Returns:
    joint_positions: The joint positions of the robot.
    """
    joint_states = p.getJointStates(robot_id, range(num_joints))
    joint_positions =   [state[0] for state in joint_states]
    return joint_positions

def get_joint_limits(robot_id: int, num_joints: List[int]) -> Tuple[List[float], List[float]]:
    """
    Get the lower and upper joint limits of the robot.
    """
    joint_limits = [p.getJointInfo(robot_id, i) for i in num_joints]
    lower_limits = [limit[8] for limit in joint_limits]
    upper_limits = [limit[9] for limit in joint_limits]
    return lower_limits, upper_limits

def get_joint_velocities(robot_id: int, num_joints: int) -> List[float]:
    """
    Get the joint velocities of the robot.
    """
    joint_states = p.getJointStates(robot_id, range(num_joints))
    joint_velocities = [state[1] for state in joint_states]
    return joint_velocities

def sort_joint(body: int, joint: int):
    Dof = 0
    q_i = []

    for i in range(joint):
        joint_info = p.getJointInfo(body, i)
        print("joint_info", joint_info)
        if joint_info[2] == 0:
            Dof += 1
            q_i.append(i)

            
    return Dof, q_i

def get_max_force(body: int, joint: List[int]) -> List[float]:
    """
    Get the maximum force of the robot.
    """
    joint_states = [p.getJointInfo(body, i) for i in joint]
    max_force = [state[10] for state in joint_states]
    return max_force
    
class PDController:
    """
    "stable proportional-derivative controller" DOI: 10.1109/MCG.2011.30
    """
    def __init__(self, pd):
        self._pd = pd
    
    def computePD(self, bodyUniqueId, jointIndices, desiredPosition, 
                  desiredVelocity, kps, kds, maxForces, timeStep):
        # numJoints = self._pd.getNumJoints(bodyUniqueId)
        jointStates = self._pd.getJointStates(bodyUniqueId, jointIndices)
        q1 = []
        qdot1 = []
        zeroAccelerations = []
        for i in range(len(jointIndices)):
            # print("i & idx", i, jointIndices[i])
            # idx = jointIndices[i]
            q1.append(jointStates[i][0])
            qdot1.append(jointStates[i][1])
            zeroAccelerations.append(0)
        q = np.array(q1)
        qdot = np.array(qdot1)
        q_desired = np.array(desiredPosition)
        qdot_desired = np.array(desiredVelocity)
        
        qError = q_desired - q
        qdotError = qdot_desired - qdot
        # print("qError", qError)
        # print("qdotError", qdotError)
        
        Kp = np.diagflat(kps)
        Kd = np.diagflat(kds)
        
        p_term = np.dot(Kp, qError)
        d_term = np.dot(Kd, qdotError)
        
        M = self._pd.calculateMassMatrix(bodyUniqueId, q1)
        M = np.array(M)
        
        G = self._pd.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations)
        G = np.array(G)
        qddot = np.linalg.solve(a=(M + Kd * timeStep),
                                b=(-G + p_term + d_term))        
        tau = p_term + d_term - (Kd.dot(qddot) * timeStep)
        # Clip generalized forces to actuator limits
        maxF = np.array(maxForces)
        generalized_forces = np.clip(tau, -maxF, maxF)
        # print("generalized_forces", generalized_forces)
        return generalized_forces

class PDControllerExplicit(object):

  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):
    # numJoints = self._pb.getNumJoints(bodyUniqueId)
    jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)
    q1 = []
    qdot1 = []
    for i in range(len(jointIndices)):

        q1.append(jointStates[i][0])
        qdot1.append(jointStates[i][1])
    q = np.array(q1)
    qdot = np.array(qdot1)
    qdes = np.array(desiredPositions)
    qdotdes = np.array(desiredVelocities)
    qError = qdes - q
    qdotError = qdotdes - qdot
    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)
    forces = Kp.dot(qError) + Kd.dot(qdotError)
    maxF = np.array(maxForces)
    forces = np.clip(forces, -maxF, maxF)
    print("forces", forces)
    return forces   
    
# p.connect(p.GUI)
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
kukaId = p.loadURDF(osp.join(osp.dirname(__file__), '..', '..', '..', 'model', 'urdf', 'sr_description','urdf','shadow_hand.urdf'), [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
numJoints = p.getNumJoints(kukaId)

t = 0
positions = get_joint_positions(kukaId, numJoints)
velocities = get_joint_velocities(kukaId, numJoints)
print("link info", p.getLinkStates(kukaId, range(numJoints)))
timeStepId = p.addUserDebugParameter("timeStep", 0.001, 0.1, 0.001)
kpCartId = p.addUserDebugParameter("kpCart", 0, 100, 50)
kdCartId = p.addUserDebugParameter("kdCart", 0, 20, 15)
dof, q_i = sort_joint(kukaId, numJoints)
joint_limits = get_joint_limits(kukaId, q_i)
j_00 = p.addUserDebugParameter("joint_0", joint_limits[0][0], joint_limits[1][0], 0.0)
j_01 = p.addUserDebugParameter("joint_1", joint_limits[0][1], joint_limits[1][1], 0.0)
j_02 = p.addUserDebugParameter("joint_2", joint_limits[0][2], joint_limits[1][2], 0.0)
j_03 = p.addUserDebugParameter("joint_3", joint_limits[0][3], joint_limits[1][3], 0.0)
j_04 = p.addUserDebugParameter("joint_4", joint_limits[0][4], joint_limits[1][4], 0.0)
j_05 = p.addUserDebugParameter("joint_5", joint_limits[0][5], joint_limits[1][5], 0.0)
j_06 = p.addUserDebugParameter("joint_6", joint_limits[0][6], joint_limits[1][6], 0.0)
j_07 = p.addUserDebugParameter("joint_7", joint_limits[0][7], joint_limits[1][7], 0.0)
j_08 = p.addUserDebugParameter("joint_8", joint_limits[0][8], joint_limits[1][8], 0.0)
j_09 = p.addUserDebugParameter("joint_9", joint_limits[0][9], joint_limits[1][9], 0.0)
j_10 = p.addUserDebugParameter("joint_10", joint_limits[0][10], joint_limits[1][10], 0.0)
j_11 = p.addUserDebugParameter("joint_11", joint_limits[0][11], joint_limits[1][11], 0.0)
j_12 = p.addUserDebugParameter("joint_12", joint_limits[0][12], joint_limits[1][12], 0.0)
j_13 = p.addUserDebugParameter("joint_13", joint_limits[0][13], joint_limits[1][13], 0.0)
j_14 = p.addUserDebugParameter("joint_14", joint_limits[0][14], joint_limits[1][14], 0.0)
j_15 = p.addUserDebugParameter("joint_15", joint_limits[0][15], joint_limits[1][15], 0.0)
j_16 = p.addUserDebugParameter("joint_16", joint_limits[0][16], joint_limits[1][16], 0.0)
j_17 = p.addUserDebugParameter("joint_17", joint_limits[0][17], joint_limits[1][17], 0.0)
j_18 = p.addUserDebugParameter("joint_18", joint_limits[0][18], joint_limits[1][18], 0.0)
j_19 = p.addUserDebugParameter("joint_19", joint_limits[0][19], joint_limits[1][19], 0.0)
j_20 = p.addUserDebugParameter("joint_20", joint_limits[0][20], joint_limits[1][20], 0.0)
j_21 = p.addUserDebugParameter("joint_21", joint_limits[0][21], joint_limits[1][21], 0.0)
j_22 = p.addUserDebugParameter("joint_22", joint_limits[0][22], joint_limits[1][22], 0.0)
j_23 = p.addUserDebugParameter("joint_23", joint_limits[0][23], joint_limits[1][23], 0.0)

sPD = PDController(p)
spline = cubic_spline(p)

p.setGravity(0, 0, -10)

useRealTimeSim = False

p.setRealTimeSimulation(useRealTimeSim)

timeStep = 0.001


dof, q_i = sort_joint(kukaId, numJoints)
# print("dof", dof, "q_i", q_i)
# print(type(q_i))
zero_vec = [0.0]*dof
# print("dof", dof)
position_dof = [positions[i] for i in q_i]
velocity_dof = [velocities[i] for i in q_i]
# print("position_dof", position_dof)
# q, q_dot, q_ddot, time_steps = cubic_motion_planning(np.array(position_dof), np.array([1] * dof), 0, 60, 100)
# print("q", len(q[1]))
# print("q_dot", q_dot)
# print("q_ddot", q_ddot)
maxforce = get_max_force(kukaId, q_i)
# maxforce = [500]*dof
# print("maxforce", maxforce)

disable_motor2(kukaId, q_i)

for i in q_i:
    print(i, p.getJointInfo(kukaId, i))

while p.isConnected():
    timeStep = p.readUserDebugParameter(timeStepId)
    p.setTimeStep(timeStep)
    kp = p.readUserDebugParameter(kpCartId)
    kd = p.readUserDebugParameter(kdCartId)
    joint_00 = p.readUserDebugParameter(j_00)
    joint_01 = p.readUserDebugParameter(j_01)
    joint_02 = p.readUserDebugParameter(j_02)
    joint_03 = p.readUserDebugParameter(j_03)
    joint_04 = p.readUserDebugParameter(j_04)
    joint_05 = p.readUserDebugParameter(j_05)
    joint_06 = p.readUserDebugParameter(j_06)
    joint_07 = p.readUserDebugParameter(j_07)
    joint_08 = p.readUserDebugParameter(j_08)
    joint_09 = p.readUserDebugParameter(j_09)
    joint_10 = p.readUserDebugParameter(j_10)
    joint_11 = p.readUserDebugParameter(j_11)
    joint_12 = p.readUserDebugParameter(j_12)
    joint_13 = p.readUserDebugParameter(j_13)
    joint_14 = p.readUserDebugParameter(j_14)
    joint_15 = p.readUserDebugParameter(j_15)
    joint_16 = p.readUserDebugParameter(j_16)
    joint_17 = p.readUserDebugParameter(j_17)
    joint_18 = p.readUserDebugParameter(j_18)
    joint_19 = p.readUserDebugParameter(j_19)
    joint_20 = p.readUserDebugParameter(j_20)
    joint_21 = p.readUserDebugParameter(j_21)
    joint_22 = p.readUserDebugParameter(j_22)
    joint_23 = p.readUserDebugParameter(j_23)
    desiredPos = [joint_00, joint_01, joint_02, joint_03, joint_04, joint_05, joint_06, joint_07, joint_08, joint_09, joint_10, joint_11, joint_12, joint_13, joint_14, joint_15, joint_16, joint_17, joint_18, joint_19, joint_20, joint_21, joint_22, joint_23]
    desiredVal = [0.0]*dof
    
    # q, q_dot, q_ddot, time_steps = cubic_motion_planning(np.array(position_dof), np.array(desiredPos), 0, 60, 100)
    
    tau = sPD.computePD(kukaId, q_i, desiredPos, desiredVal, 
                        [kp]*dof, [kd]*dof, maxforce, timeStep)
    #print("tau", tau)
    # print("tau", tau)
    for i in range(dof):
        p.setJointMotorControl2(bodyIndex=kukaId,
                                jointIndex=q_i[i],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=desiredPos[i],)        
        # p.setJointMotorControlMultiDof(kukaId,
        #                                q_i[i],
        #                                p.TORQUE_CONTROL,
        #                                [tau[i]])
        # print("1", p.getJointStates(kukaId, q_i))
        # print("2", p.getJointStates(kukaId, range(numJoints)))
    if (not useRealTimeSim):
        p.stepSimulation()
        time.sleep(timeStep)
        
        
        
        
        
        
        
        
        
        
# disable_motor(kukaId, numJoints, 0.0)
# i = 0
# print("time_steps", time_steps)
# while 1:
#     p.stepSimulation()
#     print("step", i)
#     t = t + 60/100
#     error = q[i] - position_dof
#     error_dot = q_dot[i] - velocity_dof
#     error_dot_dot = q_ddot[i] 
    
#     m = p.calculateMassMatrix(kukaId, position_dof)
#     h = p.calculateInverseDynamics(kukaId, position_dof, velocity_dof, zero_vec)
#     g = p.calculateInverseDynamics(kukaId, position_dof, zero_vec, zero_vec)
#     # print("Mass matrix: ", m, "Inverse dynamics: ", h, "Gravity: ", g)
#     E = 3*error + 1*error_dot + error_dot_dot
#     print("error", len(E))
#     for j in range(dof):
#         p.setJointMotorControl2(bodyIndex=kukaId,
#                                 jointIndex=j,
#                                 controlMode=p.POSITION_CONTROL,
#                                 targetPosition=q[i][j],
#                                 targetVelocity=0,
#                                 force=500,
#                                 positionGain=0.03,
#                                 velocityGain=1)
    
    
    
#     # control = p.calculateInverseDynamics(kukaId, position_dof, velocity_dof, E)
#     # # control = np.dot(m, E) + h
#     # print("control", control)
#     # p.setJointMotorControlArray(
#     #     bodyUniqueId=kukaId,
#     #     jointIndices=q_i,
#     #     controlMode=p.TORQUE_CONTROL,
#     #     forces=control.tolist(),
#     # )
    
#     position_dof = [get_joint_positions(kukaId, numJoints)[i] for i in q_i]
#     velocity_dof = [get_joint_velocities(kukaId, numJoints)[i] for i in q_i]
#     if i < 99:
#         i += 1

