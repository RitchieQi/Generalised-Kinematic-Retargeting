import numpy as np
import pybullet as p
import time
import pybullet_data

class PDControllerStableMultiDof(object):

  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):

    numJoints = len(jointIndices)  #self._pb.getNumJoints(bodyUniqueId)
    curPos, curOrn = self._pb.getBasePositionAndOrientation(bodyUniqueId)
    #q1 = [desiredPositions[0],desiredPositions[1],desiredPositions[2],desiredPositions[3],desiredPositions[4],desiredPositions[5],desiredPositions[6]]
    q1 = [curPos[0], curPos[1], curPos[2], curOrn[0], curOrn[1], curOrn[2], curOrn[3]]

    #qdot1 = [0,0,0, 0,0,0,0]
    baseLinVel, baseAngVel = self._pb.getBaseVelocity(bodyUniqueId)

    qdot1 = [
        baseLinVel[0], baseLinVel[1], baseLinVel[2], baseAngVel[0], baseAngVel[1], baseAngVel[2], 0
    ]
    qError = [0, 0, 0, 0, 0, 0, 0]

    qIndex = 7
    qdotIndex = 7
    zeroAccelerations = [0, 0, 0, 0, 0, 0, 0]
    for i in range(numJoints):
      js = self._pb.getJointStateMultiDof(bodyUniqueId, jointIndices[i])

      jointPos = js[0]
      jointVel = js[1]
      q1 += jointPos

      if len(js[0]) == 1:
        desiredPos = desiredPositions[qIndex]

        qdiff = desiredPos - jointPos[0]
        qError.append(qdiff)
        zeroAccelerations.append(0.)
        qdot1 += jointVel
        qIndex += 1
        qdotIndex += 1
      if len(js[0]) == 4:
        desiredPos = [
            desiredPositions[qIndex], desiredPositions[qIndex + 1], desiredPositions[qIndex + 2],
            desiredPositions[qIndex + 3]
        ]
        axis = self._pb.getAxisDifferenceQuaternion(desiredPos, jointPos)
        jointVelNew = [jointVel[0], jointVel[1], jointVel[2], 0]
        qdot1 += jointVelNew
        qError.append(axis[0])
        qError.append(axis[1])
        qError.append(axis[2])
        qError.append(0)
        desiredVel = [
            desiredVelocities[qdotIndex], desiredVelocities[qdotIndex + 1],
            desiredVelocities[qdotIndex + 2]
        ]
        zeroAccelerations += [0., 0., 0., 0.]
        qIndex += 4
        qdotIndex += 4

    q = np.array(q1)
    qdot = np.array(qdot1)

    qdotdesired = np.array(desiredVelocities)
    qdoterr = qdotdesired - qdot

    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)

    # Compute -Kp(q + qdot - qdes)
    p_term = Kp.dot(qError - qdot*timeStep)
    # Compute -Kd(qdot - qdotdes)
    d_term = Kd.dot(qdoterr)

    # Compute Inertia matrix M(q)
    M = self._pb.calculateMassMatrix(bodyUniqueId, q1, flags=1)
    M = np.array(M)
    # Given: M(q) * qddot + C(q, qdot) = T_ext + T_int
    # Compute Coriolis and External (Gravitational) terms G = C - T_ext
    G = self._pb.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations, flags=1)
    G = np.array(G)
    # Obtain estimated generalized accelerations, considering Coriolis and Gravitational forces, and stable PD actions
    qddot = np.linalg.solve(a=(M + Kd * timeStep),
                            b=p_term + d_term - G)
    # Compute control generalized forces (T_int)
    tau = p_term + d_term - Kd.dot(qddot) * timeStep
    # Clip generalized forces to actuator limits
    maxF = np.array(maxForces)
    generalized_forces = np.clip(tau, -maxF, maxF)
    return generalized_forces

class PDControllerStable(object):
  """
  Implementation based on: Tan, J., Liu, K., & Turk, G. (2011). "Stable proportional-derivative controllers"
  DOI: 10.1109/MCG.2011.30
  """
  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):
    numJoints = self._pb.getNumJoints(bodyUniqueId)
    jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)
    q1 = []
    qdot1 = []
    zeroAccelerations = []
    for i in range(numJoints):
      q1.append(jointStates[i][0])
      qdot1.append(jointStates[i][1])
      zeroAccelerations.append(0)

    q = np.array(q1)
    qdot = np.array(qdot1)
    qdes = np.array(desiredPositions)
    qdotdes = np.array(desiredVelocities)

    qError = qdes - q
    qdotError = qdotdes - qdot

    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)

    # Compute -Kp(q + qdot - qdes)
    p_term = Kp.dot(qError - qdot*timeStep)
    # Compute -Kd(qdot - qdotdes)
    d_term = Kd.dot(qdotError)

    # Compute Inertia matrix M(q)
    M = self._pb.calculateMassMatrix(bodyUniqueId, q1)
    M = np.array(M)
    # Given: M(q) * qddot + C(q, qdot) = T_ext + T_int
    # Compute Coriolis and External (Gravitational) terms G = C - T_ext
    G = self._pb.calculateInverseDynamics(bodyUniqueId, q1, qdot1, zeroAccelerations)
    G = np.array(G)
    # Obtain estimated generalized accelerations, considering Coriolis and Gravitational forces, and stable PD actions
    qddot = np.linalg.solve(a=(M + Kd * timeStep),
                            b=(-G + p_term + d_term))
    # Compute control generalized forces (T_int)
    tau = p_term + d_term - (Kd.dot(qddot) * timeStep)
    # Clip generalized forces to actuator limits
    maxF = np.array(maxForces)
    generalized_forces = np.clip(tau, -maxF, maxF)
    return generalized_forces

class PDControllerExplicitMultiDof(object):

  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):

    numJoints = len(jointIndices)  #self._pb.getNumJoints(bodyUniqueId)
    curPos, curOrn = self._pb.getBasePositionAndOrientation(bodyUniqueId)
    q1 = [curPos[0], curPos[1], curPos[2], curOrn[0], curOrn[1], curOrn[2], curOrn[3]]
    baseLinVel, baseAngVel = self._pb.getBaseVelocity(bodyUniqueId)
    qdot1 = [
        baseLinVel[0], baseLinVel[1], baseLinVel[2], baseAngVel[0], baseAngVel[1], baseAngVel[2], 0
    ]
    qError = [0, 0, 0, 0, 0, 0, 0]
    qIndex = 7
    qdotIndex = 7
    zeroAccelerations = [0, 0, 0, 0, 0, 0, 0]
    for i in range(numJoints):
      js = self._pb.getJointStateMultiDof(bodyUniqueId, jointIndices[i])

      jointPos = js[0]
      jointVel = js[1]
      q1 += jointPos

      if len(js[0]) == 1:
        desiredPos = desiredPositions[qIndex]

        qdiff = desiredPos - jointPos[0]
        qError.append(qdiff)
        zeroAccelerations.append(0.)
        qdot1 += jointVel
        qIndex += 1
        qdotIndex += 1
      if len(js[0]) == 4:
        desiredPos = [
            desiredPositions[qIndex], desiredPositions[qIndex + 1], desiredPositions[qIndex + 2],
            desiredPositions[qIndex + 3]
        ]
        axis = self._pb.getAxisDifferenceQuaternion(desiredPos, jointPos)
        jointVelNew = [jointVel[0], jointVel[1], jointVel[2], 0]
        qdot1 += jointVelNew
        qError.append(axis[0])
        qError.append(axis[1])
        qError.append(axis[2])
        qError.append(0)
        desiredVel = [
            desiredVelocities[qdotIndex], desiredVelocities[qdotIndex + 1],
            desiredVelocities[qdotIndex + 2]
        ]
        zeroAccelerations += [0., 0., 0., 0.]
        qIndex += 4
        qdotIndex += 4

    q = np.array(q1)
    qdot = np.array(qdot1)
    qdotdesired = np.array(desiredVelocities)
    qdoterr = qdotdesired - qdot
    Kp = np.diagflat(kps)
    Kd = np.diagflat(kds)
    p = Kp.dot(qError)
    d = Kd.dot(qdoterr)
    forces = p + d
    maxF = np.array(maxForces)
    forces = np.clip(forces, -maxF, maxF)
    return forces

class PDControllerExplicit(object):

  def __init__(self, pb):
    self._pb = pb

  def computePD(self, bodyUniqueId, jointIndices, desiredPositions, desiredVelocities, kps, kds,
                maxForces, timeStep):
    numJoints = self._pb.getNumJoints(bodyUniqueId)
    jointStates = self._pb.getJointStates(bodyUniqueId, jointIndices)
    q1 = []
    qdot1 = []
    for i in range(numJoints):
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
    return forces

if __name__ == "__main__":
    useMaximalCoordinates = False
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    pole = p.loadURDF("cartpole.urdf", [0, 0, 0], useMaximalCoordinates=useMaximalCoordinates)
    pole2 = p.loadURDF("cartpole.urdf", [0, 1, 0], useMaximalCoordinates=useMaximalCoordinates)
    pole3 = p.loadURDF("cartpole.urdf", [0, 2, 0], useMaximalCoordinates=useMaximalCoordinates)
    pole4 = p.loadURDF("cartpole.urdf", [0, 3, 0], useMaximalCoordinates=useMaximalCoordinates)
    
    exPD = PDControllerExplicitMultiDof(p)
    sPD = PDControllerStable(p)
    
    for i in range(p.getNumJoints(pole2)):
        #disable default constraint-based motors
        p.setJointMotorControl2(pole, i, p.POSITION_CONTROL, targetPosition=0, force=0)
        p.setJointMotorControl2(pole2, i, p.POSITION_CONTROL, targetPosition=0, force=0)
        p.setJointMotorControl2(pole3, i, p.POSITION_CONTROL, targetPosition=0, force=0)
        p.setJointMotorControl2(pole4, i, p.POSITION_CONTROL, targetPosition=0, force=0)

    timeStepId = p.addUserDebugParameter("timeStep", 0.001, 0.1, 0.01)
    desiredPosCartId = p.addUserDebugParameter("desiredPosCart", -10, 10, 2)
    desiredVelCartId = p.addUserDebugParameter("desiredVelCart", -10, 10, 0)
    kpCartId = p.addUserDebugParameter("kpCart", 0, 500, 1300)
    kdCartId = p.addUserDebugParameter("kdCart", 0, 300, 150)
    maxForceCartId = p.addUserDebugParameter("maxForceCart", 0, 5000, 1000)
    
    textColor = [1, 1, 1]
    shift = 0.05
    p.addUserDebugText("explicit PD", [shift, 0, .1],
                    textColor,
                    parentObjectUniqueId=pole,
                    parentLinkIndex=1)
    p.addUserDebugText("explicit PD plugin", [shift, 0, -.1],
                    textColor,
                    parentObjectUniqueId=pole2,
                    parentLinkIndex=1)
    p.addUserDebugText("stablePD", [shift, 0, .1],
                    textColor,
                    parentObjectUniqueId=pole4,
                    parentLinkIndex=1)
    p.addUserDebugText("position constraint", [shift, 0, -.1],
                    textColor,
                    parentObjectUniqueId=pole3,
                    parentLinkIndex=1)
    
    desiredPosPoleId = p.addUserDebugParameter("desiredPosPole", -10, 10, 0)
    desiredVelPoleId = p.addUserDebugParameter("desiredVelPole", -10, 10, 0)
    kpPoleId = p.addUserDebugParameter("kpPole", 0, 500, 1200)
    kdPoleId = p.addUserDebugParameter("kdPole", 0, 300, 100)
    maxForcePoleId = p.addUserDebugParameter("maxForcePole", 0, 5000, 1000)
    
    pd = p.loadPlugin("pdControlPlugin")
    
    p.setGravity(0, 0, -10)
    
    useRealTimeSim = False

    p.setRealTimeSimulation(useRealTimeSim)

    timeStep = 0.001
    for i in [0,1]:
      print("joint dynamics info", p.getDynamicsInfo(pole4, i))
    while p.isConnected():
        timeStep = p.readUserDebugParameter(timeStepId)
        p.setTimeStep(timeStep)
        
        desiredPosCart = p.readUserDebugParameter(desiredPosCartId)
        desiredVelCart = p.readUserDebugParameter(desiredVelCartId)
        kpCart = p.readUserDebugParameter(kpCartId)
        kdCart = p.readUserDebugParameter(kdCartId)
        maxForceCart = p.readUserDebugParameter(maxForceCartId)

        desiredPosPole = p.readUserDebugParameter(desiredPosPoleId)
        desiredVelPole = p.readUserDebugParameter(desiredVelPoleId)
        kpPole = p.readUserDebugParameter(kpPoleId)
        kdPole = p.readUserDebugParameter(kdPoleId)
        maxForcePole = p.readUserDebugParameter(maxForcePoleId)
        
        basePos, baseOrn = p.getBasePositionAndOrientation(pole)
        
        baseDof = 7
        taus = exPD.computePD(pole, [0, 1], [basePos[0], basePos[1], basePos[2], baseOrn[0], baseOrn[1], baseOrn[2], baseOrn[3],
        desiredPosCart, desiredPosPole], [0, 0, 0, 0, 0, 0, 0, desiredVelCart, desiredVelPole], [0, 0, 0, 0, 0, 0, 0, kpCart, kpPole],
        [0, 0, 0, 0, 0, 0, 0, kdCart, kdPole],[0, 0, 0, 0, 0, 0, 0, maxForceCart, maxForcePole], timeStep)
        
        for j in [0,1]:
            p.setJointMotorControlMultiDof(pole, 
                                           j, 
                                           p.TORQUE_CONTROL, 
                                           force=[taus[j + baseDof]])
        if (pd >= 0):
            link = 0
            p.setJointMotorControl2(bodyUniqueId=pole2, 
                                    jointIndex=link,
                                    controlMode=p.PD_CONTROL,
                                    targetPosition=desiredPosCart,
                                    targetVelocity=desiredVelCart,
                                    force=maxForceCart,
                                    positionGain=kpCart,
                                    velocityGain=kdCart)
            link = 1
            p.setJointMotorControl2(bodyUniqueId=pole2, 
                                    jointIndex=link,
                                    controlMode=p.PD_CONTROL,
                                    targetPosition=desiredPosPole,
                                    targetVelocity=desiredVelPole,
                                    force=maxForcePole,
                                    positionGain=kpPole,
                                    velocityGain=kdPole)
        
        taus = sPD.computePD(pole4, [0, 1], [desiredPosCart, desiredPosPole],
                            [desiredVelCart, desiredVelPole], [kpCart, kpPole], [kdCart, kdPole],
                            [maxForceCart, maxForcePole], timeStep)
        for j in [0,1]:
            p.setJointMotorControlMultiDof(pole4, 
                                           j, 
                                           p.TORQUE_CONTROL, 
                                           force=[taus[j]])
        p.setJointMotorControl2(pole3,
                                0,
                                p.POSITION_CONTROL,
                                targetPosition=desiredPosCart,
                                targetVelocity=desiredVelCart,
                                positionGain=timeStep * (kpCart / 150.),
                                velocityGain=0.5,
                                force=maxForceCart)
        p.setJointMotorControl2(pole3,
                                1,
                                p.POSITION_CONTROL,
                                targetPosition=desiredPosPole,
                                targetVelocity=desiredVelPole,
                                positionGain=timeStep * (kpPole / 150.),
                                velocityGain=0.5,
                                force=maxForcePole)
        
        if (not useRealTimeSim):
            p.stepSimulation()
            time.sleep(timeStep)