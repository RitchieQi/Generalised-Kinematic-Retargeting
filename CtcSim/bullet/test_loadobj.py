import pybullet as p
import time
import pybullet_data
import math
import os.path as osp
import os
import numpy as np
from typing import List, Tuple
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
import json
from scipy.spatial.transform import Rotation as R

from torch.utils.data import Dataset

body_idx = {}

def load_obj_as_mesh(body_name: str, obj_path: str, position: np.ndarray, orientation: np.ndarray) -> None:
    """Load obj file and create mesh/collision shape from it.
        ref: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createVisualShape.py
    Args:
        body_name (str): The name of the body. Must be unique in the sim.
        obj_path (str): Path to the obj file.
    """
    obj_mass = 0
    obj_visual_shape_id = p.createVisualShape(
                                                shapeType=p.GEOM_MESH, 
                                                fileName=obj_path, 
                                                rgbaColor=[1, 1, 1, 1],
                                                meshScale=[1, 1, 1])

    obj_collision_shape_id = p.createCollisionShape(
                                            shapeType = p.GEOM_MESH,
                                            fileName = obj_path,
                                            flags=p.GEOM_FORCE_CONCAVE_TRIMESH |
                                            p.GEOM_CONCAVE_INTERNAL_EDGE,
                                            meshScale=[1, 1, 1])
    
    body_idx[body_name] = p.createMultiBody(
                            baseMass=obj_mass,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=obj_collision_shape_id,
                            baseVisualShapeIndex=obj_visual_shape_id,
                            basePosition=position,
                            baseOrientation=orientation,
                            useMaximalCoordinates=True)

class dexycb_obj(Dataset):
    def __init__(self, dexycb_dir = '/home/liyuan/DexYCB/', task = "train", data_dir = osp.join(osp.dirname(osp.abspath(__file__)),"..","..", "CtcSDF", "data")):
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset("s0", task)
        config_file = osp.join(data_dir, "dexycb_{}_s0.json".format(task))
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
    def __len__(self):
        return len(self.config)
    
    def __getitem__(self, idx):
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
        return obj_file, pose, translation, quat
        
obj_dataset = dexycb_obj()
obj_file, pose, translation, quat = obj_dataset[0]


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -10)
useRealTimeSim = False

p.setRealTimeSimulation(useRealTimeSim)
timeStep = 0.001

load_obj_as_mesh("obj", obj_file, translation, quat)
while p.isConnected():
    if (not useRealTimeSim):
        p.stepSimulation()
        time.sleep(timeStep)
        