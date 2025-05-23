import os
import math

import torch

import pytorch_kinematics as pk

TEST_DIR = os.path.dirname(__file__)


def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot


def quaternion_equality(a, b):
    # negative of a quaternion is the same rotation
    return torch.allclose(a, b) or torch.allclose(a, -b)


# test more complex robot and the MJCF parser
def test_fk_mjcf():
    chain = pk.build_chain_from_mjcf(open(os.path.join(TEST_DIR, "ant.xml")).read())
    chain = chain.to(dtype=torch.float64)
    print(chain)
    print(chain.get_joint_parameter_names())
    th = {'hip_1': 1.0, 'ankle_1': 1}
    ret = chain.forward_kinematics(th)
    tg = ret['aux_1']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, torch.tensor([0.87758256, 0., 0., 0.47942554], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([0.2, 0.2, 0.75], dtype=torch.float64))
    tg = ret['front_left_foot']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, torch.tensor([0.77015115, -0.4600326, 0.13497724, 0.42073549], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([0.13976626, 0.47635466, 0.75], dtype=torch.float64))
    print(ret)


def test_fk_serial_mjcf():
    chain = pk.build_serial_chain_from_mjcf(open(os.path.join(TEST_DIR, "ant.xml")).read(), 'front_left_foot')
    chain = chain.to(dtype=torch.float64)
    tg = chain.forward_kinematics([1.0, 1.0])
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, torch.tensor([0.77015115, -0.4600326, 0.13497724, 0.42073549], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([0.13976626, 0.47635466, 0.75], dtype=torch.float64))


def test_fkik():
    data = '<robot name="test_robot">' \
           '<link name="link1" />' \
           '<link name="link2" />' \
           '<link name="link3" />' \
           '<joint name="joint1" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link1"/>' \
           '<child link="link2"/>' \
           '</joint>' \
           '<joint name="joint2" type="revolute">' \
           '<origin xyz="1.0 0.0 0.0"/>' \
           '<parent link="link2"/>' \
           '<child link="link3"/>' \
           '</joint>' \
           '</robot>'
    chain = pk.build_serial_chain_from_urdf(data, 'link3')
    th1 = torch.tensor([0.42553542, 0.17529176])
    tg = chain.forward_kinematics(th1)
    pos, rot = quat_pos_from_transform3d(tg)
    assert torch.allclose(pos, torch.tensor([[1.91081784, 0.41280851, 0.0000]]))
    assert quaternion_equality(rot, torch.tensor([[0.95521418, 0.0000, 0.0000, 0.2959153]]))
    N = 20
    th_batch = torch.rand(N, 2)
    tg_batch = chain.forward_kinematics(th_batch)
    m = tg_batch.get_matrix()
    for i in range(N):
        tg = chain.forward_kinematics(th_batch[i])
        assert torch.allclose(tg.get_matrix().view(4, 4), m[i])

    # check that gradients are passed through
    th2 = torch.tensor([0.42553542, 0.17529176], requires_grad=True)
    tg = chain.forward_kinematics(th2)
    pos, rot = quat_pos_from_transform3d(tg)
    # note that since we are using existing operations we are not checking grad calculation correctness
    assert th2.grad is None
    pos.norm().backward()
    assert th2.grad is not None


def test_urdf():
    chain = pk.build_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read())
    chain.to(dtype=torch.float64)
    th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]
    ret = chain.forward_kinematics(th)
    tg = ret['lbr_iiwa_link_7']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, torch.tensor([7.07106781e-01, 0, -7.07106781e-01, 0], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([-6.60827561e-01, 0, 3.74142136e-01], dtype=torch.float64))


def test_urdf_serial():
    chain = pk.build_serial_chain_from_urdf(open(os.path.join(TEST_DIR, "kuka_iiwa.urdf")).read(), "lbr_iiwa_link_7")
    chain.to(dtype=torch.float64)
    print(chain)
    print(chain.get_joint_parameter_names())
    th = [0.0, -math.pi / 4.0, 0.0, math.pi / 2.0, 0.0, math.pi / 4.0, 0.0]
    ret = chain.forward_kinematics(th, end_only=False)
    tg = ret['lbr_iiwa_link_7']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, torch.tensor([7.07106781e-01, 0, -7.07106781e-01, 0], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([-6.60827561e-01, 0, 3.74142136e-01], dtype=torch.float64))

    N = 1000
    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)
    chain = chain.to(dtype=dtype, device=d)

    import time
    start = time.time()
    tg_batch = chain.forward_kinematics(th_batch)
    m = tg_batch.get_matrix()
    elapsed = time.time() - start
    print("elapsed {}s for N={} when parallel".format(elapsed, N))

    start = time.time()
    elapsed = 0
    for i in range(N):
        tg = chain.forward_kinematics(th_batch[i])
        elapsed += time.time() - start
        start = time.time()
        assert torch.allclose(tg.get_matrix().view(4, 4), m[i])
    print("elapsed {}s for N={} when serial".format(elapsed, N))


# test robot with prismatic and fixed joints
def test_fk_simple_arm():
    chain = pk.build_chain_from_sdf(open(os.path.join(TEST_DIR, "simple_arm.sdf")).read())
    chain = chain.to(dtype=torch.float64)
    # print(chain)
    # print(chain.get_joint_parameter_names())
    ret = chain.forward_kinematics({'arm_elbow_pan_joint': math.pi / 2.0, 'arm_wrist_lift_joint': -0.5})
    tg = ret['arm_wrist_roll']
    pos, rot = quat_pos_from_transform3d(tg)
    assert quaternion_equality(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=torch.float64))
    assert torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=torch.float64))

    N = 100
    ret = chain.forward_kinematics({'arm_elbow_pan_joint': torch.rand(N, 1), 'arm_wrist_lift_joint': torch.rand(N, 1)})
    tg = ret['arm_wrist_roll']
    assert list(tg.get_matrix().shape) == [N, 4, 4]


def test_cuda():
    if torch.cuda.is_available():
        d = "cuda"
        dtype = torch.float64
        chain = pk.build_chain_from_sdf(open(os.path.join(TEST_DIR, "simple_arm.sdf")).read())
        chain = chain.to(dtype=dtype, device=d)

        # NOTE: do it twice because we previously had an issue with default arguments
        #  like joint=Joint() causing spooky behavior
        chain = pk.build_chain_from_sdf(open(os.path.join(TEST_DIR, "simple_arm.sdf")).read())
        chain = chain.to(dtype=dtype, device=d)

        ret = chain.forward_kinematics({'arm_elbow_pan_joint': math.pi / 2.0, 'arm_wrist_lift_joint': -0.5})
        tg = ret['arm_wrist_roll']
        pos, rot = quat_pos_from_transform3d(tg)
        assert quaternion_equality(rot, torch.tensor([0.70710678, 0., 0., 0.70710678], dtype=dtype, device=d))
        assert torch.allclose(pos, torch.tensor([1.05, 0.55, 0.5], dtype=dtype, device=d))

        data = '<robot name="test_robot">' \
               '<link name="link1" />' \
               '<link name="link2" />' \
               '<link name="link3" />' \
               '<joint name="joint1" type="revolute">' \
               '<origin xyz="1.0 0.0 0.0"/>' \
               '<parent link="link1"/>' \
               '<child link="link2"/>' \
               '</joint>' \
               '<joint name="joint2" type="revolute">' \
               '<origin xyz="1.0 0.0 0.0"/>' \
               '<parent link="link2"/>' \
               '<child link="link3"/>' \
               '</joint>' \
               '</robot>'
        chain = pk.build_serial_chain_from_urdf(data, 'link3')
        chain = chain.to(dtype=dtype, device=d)
        N = 20
        th_batch = torch.rand(N, 2).to(device=d, dtype=dtype)
        tg_batch = chain.forward_kinematics(th_batch)
        m = tg_batch.get_matrix()
        for i in range(N):
            tg = chain.forward_kinematics(th_batch[i])
            assert torch.allclose(tg.get_matrix().view(4, 4), m[i])


# FIXME: comment out because compound joints are no longer implemented
# def test_fk_mjcf_humanoid():
#     chain = pk.build_chain_from_mjcf(open(os.path.join(TEST_DIR, "humanoid.xml")).read())
#     print(chain)
#     print(chain.get_joint_parameter_names())
#     th = {'left_knee': 0.0, 'right_knee': 0.0}
#     ret = chain.forward_kinematics(th)
#     print(ret)


def test_mjcf_slide_joint_parsing():
    # just testing that we can parse it without error
    # the slide joint is not actually of a link to another link, but instead of the base to the world
    # which we do not represent
    chain = pk.build_chain_from_mjcf(open(os.path.join(TEST_DIR, "hopper.xml")).read())
    print(chain.get_joint_parameter_names())
    print(chain.get_frame_names())


if __name__ == "__main__":
    test_fkik()
    test_fk_simple_arm()
    test_fk_mjcf()
    test_cuda()
    test_urdf()
    test_urdf_serial()
    # test_fk_mjcf_humanoid()
    test_mjcf_slide_joint_parsing()
