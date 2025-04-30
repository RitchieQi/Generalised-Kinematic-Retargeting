import json
#import pickle
from tqdm import tqdm
import os
import cv2
from manopth.manolayer import ManoLayer
from dex_ycb_toolkit.dex_ycb import DexYCBDataset
from torch.utils.data import Dataset
import torch
import shutil
import numpy as np
import json
from dex_ycb_toolkit.factory import get_dataset
import open3d as o3d
import matplotlib.pyplot as plt
import pyrender
import traceback
import trimesh
osp = os.path

class dexycb_data(Dataset):
    def __init__(self, dexycb_dir = '/home/liyuan/DexYCB/', sdf_dir = osp.join(osp.dirname(__file__), "data"), task = "train"):
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset('s0', task)
        pre_process_json = osp.join(sdf_dir, "dexycb_{}_s0.json".format(task))
        with open(pre_process_json, 'r') as f:
            self.pre_process = json.load(f)
        self.sdf_dir = osp.join(sdf_dir, "sdf_data")

    def __len__(self):
        return len(self.pre_process['images'])
    
    def __getitem__(self, idx):
        #sdf_file sdf_hand/sdf_obj: pos,neg,lab_pos,lab_neg,pos_other,neg_other,lab_pos_other,lab_neg_other

        file_name = self.pre_process['images'][idx]['file_name'] + '.npz'
        sdf_hand = np.load(osp.join(self.sdf_dir, "sdf_hand", file_name))
        sdf_obj = np.load(osp.join(self.sdf_dir, "sdf_obj", file_name))  
        sdf_norm = np.load(osp.join(self.sdf_dir, "norm", file_name))
        s0_id = self.pre_process['images'][idx]['id']
        try:
            pc = self.create_pc(s0_id,sample=1024)

            hand_sdf,hand_label = self.unpack_sdf(sdf_hand,idx, samples = 1024, clamp = 0.05, source = 'hand')
            obj_sdf,obj_label = self.unpack_sdf(sdf_obj, idx,samples = 1024, clamp = 0.05, source = 'obj')
            return pc,{"hand_sdf":hand_sdf, "hand_label":hand_label, "obj_sdf":obj_sdf, "obj_label":obj_label}
        except Exception as e:
            print(idx,e)
            return self.__getitem__(idx+1)

    def create_pc(self,s0_id, sample = None):
        depth_file = self.getdata[s0_id]['depth_file']
        label_file = self.getdata[s0_id]['label_file']
        label = np.load(label_file)
        seg_map = label['seg']
        obj_idx = self.getdata[s0_id]['ycb_ids'][self.getdata[s0_id]['ycb_grasp_ind']]
        obj_centre = label['pose_y'][self.getdata[s0_id]['ycb_grasp_ind']][:,3]
        mask = (seg_map == obj_idx)| (seg_map == 255)

        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

        filter_depth = np.where(mask, depth, 0)

        depth = o3d.geometry.Image(filter_depth)
        camera = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=self.getdata[s0_id]['intrinsics']['fx'], fy = self.getdata[s0_id]['intrinsics']['fy'], cx = self.getdata[s0_id]['intrinsics']['ppx'], cy = self.getdata[s0_id]['intrinsics']['ppy'])
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera)

        points = np.asarray(pcd.points)
        filter_points = points[(points[:, 2] >= obj_centre[2]-0.1) & (points[:, 2] <= obj_centre[2]+0.1)]
        pcd.points = o3d.utility.Vector3dVector(filter_points)

        pcd.estimate_normals()
        pcd.normalize_normals()
        pc = torch.tensor(np.asarray(pcd.points))
        n = torch.tensor(np.asarray(pcd.normals))
        pc = torch.cat([pc, n], dim = 1)
        if sample is not None:
            rand_idx = (torch.rand(sample) * pc.shape[0]).long()
            sampled_pc = torch.index_select(pc, 0, rand_idx)
            return sampled_pc
        else:
            return pc

    def create_pc_separate(self, s0_id, sample = None):
        depth_file = self.getdata[s0_id]['depth_file']
        label_file = self.getdata[s0_id]['label_file']
        label = np.load(label_file)
        seg_map = label['seg']
        obj_idx = self.getdata[s0_id]['ycb_ids'][self.getdata[s0_id]['ycb_grasp_ind']]
        obj_centre = label['pose_y'][self.getdata[s0_id]['ycb_grasp_ind']][:,3]
        
        obj_mask = (seg_map == obj_idx)
        hand_mask = (seg_map == 255)
        
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        
        filter_obj = np.where(obj_mask, depth, 0)
        filter_hand = np.where(hand_mask, depth, 0)
        
        camera = o3d.camera.PinholeCameraIntrinsic(width=640, 
                                                   height=480, 
                                                   fx=self.getdata[s0_id]['intrinsics']['fx'], 
                                                   fy = self.getdata[s0_id]['intrinsics']['fy'], 
                                                   cx = self.getdata[s0_id]['intrinsics']['ppx'], 
                                                   cy = self.getdata[s0_id]['intrinsics']['ppy'])
        
        depth_obj = o3d.geometry.Image(filter_obj)
        depth_hand = o3d.geometry.Image(filter_hand)
        
        pcd_obj = o3d.geometry.PointCloud.create_from_depth_image(depth_obj, camera)
        pcd_hand = o3d.geometry.PointCloud.create_from_depth_image(depth_hand, camera)
        
        points_obj = np.asarray(pcd_obj.points)
        points_hand = np.asarray(pcd_hand.points)
        filter_points_obj = points_obj[(points_obj[:, 2] >= obj_centre[2]-0.1) & (points_obj[:, 2] <= obj_centre[2]+0.1)]
        filter_points_hand = points_hand[(points_hand[:, 2] >= obj_centre[2]-0.1) & (points_hand[:, 2] <= obj_centre[2]+0.1)]
        
        pcd_obj.points = o3d.utility.Vector3dVector(filter_points_obj)
        pcd_hand.points = o3d.utility.Vector3dVector(filter_points_hand)
        
        pcd_obj.estimate_normals()
        pcd_obj.normalize_normals()
        pcd_hand.estimate_normals()
        pcd_hand.normalize_normals()
        
        pc_obj = torch.tensor(np.asarray(pcd_obj.points))
        n_obj = torch.tensor(np.asarray(pcd_obj.normals))
        
        pc_hand = torch.tensor(np.asarray(pcd_hand.points))
        n_hand = torch.tensor(np.asarray(pcd_hand.normals))
        
        pc_obj = torch.cat([pc_obj, n_obj], dim = 1)
        pc_hand = torch.cat([pc_hand, n_hand], dim = 1)
        
        if sample is not None:
            rand_idx_obj = (torch.rand(sample) * pc_obj.shape[0]).long()
            rand_idx_hand = (torch.rand(sample) * pc_hand.shape[0]).long()
            sampled_pc_obj = torch.index_select(pc_obj, 0, rand_idx_obj)
            sampled_pc_hand = torch.index_select(pc_hand, 0, rand_idx_hand)
            return sampled_pc_obj, sampled_pc_hand
        else:
            return pc_obj, pc_hand
    
    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)

        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1        
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label

class dexycb_points(Dataset):
    def __init__(self, data_dir = osp.join(osp.dirname(__file__), "data"), task = "train"):
        config_file = osp.join(data_dir,"dexycb_{}_s0.json".format(task))
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.point_dir = osp.join(data_dir, "points")
        self.sdf_dir = osp.join(data_dir, "sdf_data")
    
    def __len__(self):
        return len(self.meta['images'])
    
    def __getitem__(self,idx):
        sdf_name = self.meta['images'][idx]['file_name'] + '.npz'
        points_name = self.meta['images'][idx]['file_name'] + '.npy'
        sdf_hand = np.load(osp.join(self.sdf_dir, "sdf_hand", sdf_name))
        sdf_obj = np.load(osp.join(self.sdf_dir, "sdf_obj", sdf_name))
        hand_sdf, hand_label = self.unpack_sdf(sdf_hand, idx, samples = 1024, clamp = 0.05, source = 'hand')
        obj_sdf, obj_label = self.unpack_sdf(sdf_obj, idx, samples = 1024, clamp = 0.05, source = 'obj')
        hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'])
        obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'])
        hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'])
        mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'])
        mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'])
        obj_centre = obj_centre - hand_trans
        hand_joint = hand_joint - hand_trans
        try:
            pointcloud = np.load(osp.join(self.point_dir, points_name))
            pointcloud = torch.tensor(pointcloud) 
            pointcloud[:, :3] = pointcloud[:, :3] - hand_trans
            hand_sdf[:, :3] = hand_sdf[:, :3] - hand_trans
            obj_sdf[:, :3] = obj_sdf[:, :3] - hand_trans
            vote = pointcloud
            vote[:, 3:] = vote[:, :3] - obj_centre
            input_pack = {'pc':pointcloud, 'obj_xyz':obj_sdf[:, :3]}
            mano_pack = {"pose":mano_pose, "shape":mano_shape, "joint":hand_joint, "trans": hand_trans}
            sdf_pack = {"hand_sdf":hand_sdf, "hand_label":hand_label, "obj_sdf":obj_sdf, "obj_label":obj_label, "obj_centre":obj_centre, "vote":vote}
            return input_pack, mano_pack, sdf_pack
        except Exception as e:
            print(idx, e)
            return self.__getitem__(idx+1)

    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)

        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1        
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label

class dexycb_fullfeed(Dataset):
    def __init__(self, dexycb_dir = '/home/liyuan/DexYCB/', sdf_dir = osp.join(osp.dirname(__file__), "data"), task = "train"):
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset('s0', task)
        pre_process_json = osp.join(sdf_dir, "dexycb_{}_s0.json".format(task))
        with open(pre_process_json, 'r') as f:
            self.pre_process = json.load(f)
        self.sdf_dir = osp.join(sdf_dir, "sdf_data")
        self.point_dir = osp.join(sdf_dir, "points_2048")

    def __len__(self):
        return len(self.pre_process['images'])
    def load_mesh(self, idx):
        sample = self.getdata[idx]
        ycbi = sample['ycb_grasp_ind']
        label = np.load(sample['label_file'])
        # for component in label:
        #     print(component)
        #     print(label[component])
        pose_y = label['pose_y'][ycbi]
        pose_m = label['pose_m']
        mesh = trimesh.load_mesh(self.getdata.obj_file[sample['ycb_ids'][ycbi]])
        pose = np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32)))
        # pose[1] *= -1
        # pose[2] *= -1
        return mesh, pose

    def __getitem__(self, idx):
        #sdf_file sdf_hand/sdf_obj: pos,neg,lab_pos,lab_neg,pos_other,neg_other,lab_pos_other,lab_neg_other
        file_name = self.pre_process['images'][idx]['file_name'] + '.npz'
        points_name = self.pre_process['images'][idx]['file_name'] + '.npy'
        
        sdf_hand = np.load(osp.join(self.sdf_dir, "sdf_hand", file_name))
        sdf_obj = np.load(osp.join(self.sdf_dir, "sdf_obj", file_name))  
        #hand_sdf,hand_label = self.unpack_sdf(sdf_hand,idx, samples = 1024, clamp = 0.05, source = 'hand')
        obj_sdf,obj_label = self.unpack_sdf(sdf_obj, idx,samples = 2048, clamp = 0.05, source = 'obj')
        #sdf_norm = np.load(osp.join(self.sdf_dir, "norm", file_name))
        s0_id = self.pre_process['images'][idx]['id']
        color = cv2.imread(self.getdata[s0_id]['color_file'])
        
        hand_trans = torch.tensor(self.pre_process['annotations'][idx]['hand_trans'])
        obj_centre = torch.tensor(self.pre_process['annotations'][idx]['obj_center_3d'])
        hand_joint = torch.tensor(self.pre_process['annotations'][idx]['hand_joints_3d'])
        mano_pose = torch.tensor(self.pre_process['annotations'][idx]['hand_poses'])
        mano_shape = torch.tensor(self.pre_process['annotations'][idx]['hand_shapes'])   
        scale = torch.tensor(self.pre_process['annotations'][idx]['sdf_scale'])
        offset = torch.tensor(self.pre_process['annotations'][idx]['sdf_offset'])     

        obj_sdf[:, :3] = obj_sdf[:, :3]/scale - offset
        obj_centre = obj_centre - hand_trans
        hand_joint = hand_joint - hand_trans
        mesh, pose = self.load_mesh(s0_id)
        try:
            mask = self.create_pc(s0_id)
            pointcloud = np.load(osp.join(self.point_dir, points_name))
            pointcloud = torch.tensor(pointcloud) 

            pointcloud[:, :3] = pointcloud[:, :3] - hand_trans
            obj_sdf[:, :3] = obj_sdf[:, :3] - hand_trans
            
    
            #obj_sdf[:, :3] = obj_sdf[:, :3] - hand_trans
            vote = pointcloud
            vote[:, 3:] = vote[:, :3] - obj_centre
            pose[:3,3] = pose[:3,3] - hand_trans.numpy()
            input_pack = {'mesh_vert':torch.tensor(np.array(mesh.vertices)), 'mesh_face':torch.tensor(np.array(mesh.faces)), 'pose':pose,'color':color,'mask':mask,'pc':pointcloud, 'obj_xyz':obj_sdf[:, :3]}
            mano_pack = {"pose":mano_pose, "shape":mano_shape, "joint":hand_joint, "trans": hand_trans}
            sdf_pack = {"obj_sdf":obj_sdf, "obj_label":obj_label, "obj_centre":obj_centre, "vote":vote}
            return input_pack, mano_pack, sdf_pack
        except Exception as e:
            print(idx,e)
            return self.__getitem__(idx+1)

    def create_pc(self,s0_id, sample = None):
        depth_file = self.getdata[s0_id]['depth_file']
        label_file = self.getdata[s0_id]['label_file']
        label = np.load(label_file)
        seg_map = label['seg']
        obj_idx = self.getdata[s0_id]['ycb_ids'][self.getdata[s0_id]['ycb_grasp_ind']]
        mask = (seg_map == obj_idx)| (seg_map == 255)

        # depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

        # filter_depth = np.where(mask, depth, 0)

        # depth = o3d.geometry.Image(filter_depth)
        # camera = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=self.getdata[s0_id]['intrinsics']['fx'], fy = self.getdata[s0_id]['intrinsics']['fy'], cx = self.getdata[s0_id]['intrinsics']['ppx'], cy = self.getdata[s0_id]['intrinsics']['ppy'])
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera)
        # pcd.estimate_normals()
        # pcd.normalize_normals()
        # pc = torch.tensor(np.asarray(pcd.points))
        # n = torch.tensor(np.asarray(pcd.normals))
        # pc = torch.cat([pc, n], dim = 1)
        # if sample is not None:
        #     rand_idx = (torch.rand(sample) * pc.shape[0]).long()
        #     sampled_pc = torch.index_select(pc, 0, rand_idx)
        #     return mask,sampled_pc
        # else:
        #     return mask,pc
        return mask
    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)

        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1        
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label

class dexycb_objsdf(Dataset):
    def __init__(self, sdf_scaler, scale_input, double_input = False, data_dir = osp.join(osp.dirname(__file__), "data"), task = "train", pc_sample = 2048, clamp = 0.05):
        config_file = osp.join(data_dir,"dexycb_{}_s0.json".format(task))
        self.clamp = clamp
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.point_dir = osp.join(data_dir, "points_{}".format(pc_sample))
        self.sdf_dir = osp.join(data_dir, "sdf_data")
        self.sdf_scaler = sdf_scaler
        self.sample_num = pc_sample 
        self.scale_input = scale_input
        self.double_input = double_input
    def __len__(self):
        return len(self.meta['images'])
    
    def __getitem__(self,idx):
        sdf_name = self.meta['images'][idx]['file_name'] + '.npz'
        points_name = self.meta['images'][idx]['file_name'] + '.npy'
  
        
        sdf_obj = np.load(osp.join(self.sdf_dir, "sdf_obj", sdf_name))
        obj_sdf, obj_label = self.unpack_sdf(sdf_obj, idx, samples = self.sample_num, clamp = self.clamp, source = 'obj')

        hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
        obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
        hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32)
        mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32)
        mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32)
        scale = torch.tensor(self.meta['annotations'][idx]['sdf_scale'], dtype = torch.float32)
        offset = torch.tensor(self.meta['annotations'][idx]['sdf_offset'], dtype = torch.float32)
        print("value check", scale, offset, hand_trans, obj_centre, hand_joint)
        obj_sdf[:, :3] = obj_sdf[:, :3]/scale - offset #transform the sdf pointcloud back to the camera frame
        obj_centre = obj_centre - hand_trans #translate origin to the hand wrist
        hand_joint = hand_joint - hand_trans #translate origin to the hand wrist
        obj_sdf[:, :3] = (obj_sdf[:, :3] - hand_trans) * self.sdf_scaler #scale again to avoid numerical issues
        obj_sdf[:, 3:] = obj_sdf[:, 3:] / scale * self.sdf_scaler #scale the sdf value, also avoiding numerical issues
        #obj_sdf[:, 0:5] = obj_sdf[:, 0:5] / 2
        try: 
            pointcloud = np.load(osp.join(self.point_dir, points_name))
            pointcloud = torch.tensor(pointcloud, dtype = torch.float32)

            pointcloud[:, :3] = pointcloud[:, :3] - hand_trans
            if self.double_input:
                pointcloud_original = pointcloud.clone()
                obj_centre_original = obj_centre.clone()
            if self.scale_input:
                pointcloud[:, :3] = pointcloud[:, :3]*self.sdf_scaler
                obj_centre = obj_centre * self.sdf_scaler
            vote = torch.zeros(pointcloud.shape, dtype = torch.float32)
            vote[:,:3] = pointcloud[:,:3]
            vote[:, 3:] = obj_centre - pointcloud[:, :3] 

            input_pack = {'pc':pointcloud, 'obj_xyz':obj_sdf[:, :3]}
            mano_pack = {"pose":mano_pose, "shape":mano_shape, "joint":hand_joint, "trans": hand_trans}
            sdf_pack = {"obj_sdf":obj_sdf, "obj_label":obj_label, "obj_centre":obj_centre, "vote":vote}
            if self.double_input:
                input_pack = {'pc':pointcloud, 'obj_xyz':obj_sdf[:, :3], 'pc_original':pointcloud_original, 'obj_xyz_original':obj_centre_original}
            return input_pack, mano_pack, sdf_pack
        except Exception as e:
            #print(idx, e)
            return self.__getitem__(idx+1)

    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)
        if clamp is not None:
            mask_pos = (pos[...,-1] <= clamp)
            mask_neg = (neg[...,-1] >= -clamp)
            pos = pos[mask_pos]
            neg = neg[mask_neg]
            lab_pos = lab_pos[mask_pos]
            lab_neg = lab_neg[mask_neg]


        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:    
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label

class dexycb_test(Dataset):
    def __init__(self, data_dir = osp.join(osp.dirname(__file__), "data"), load_mesh = False, pc_sample = 1024, scale = None):
        config_file = osp.join(data_dir,"dexycb_test_s0.json")
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.load_mesh = load_mesh
        self.mesh_dir = osp.join(data_dir, 'mesh_data')
        self.point_dir = osp.join(data_dir, 'points_{}'.format(pc_sample))
        self.scale = scale

    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self, idx):
        points_name = self.meta['images'][idx]['file_name'] + '.npy'

        hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
        obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
        hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32)
        # mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32)
        # mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32)
        # offset = torch.tensor(self.meta['annotations'][idx]['sdf_offset'], dtype = torch.float32)   
        obj_centre = obj_centre - hand_trans
        hand_joint = hand_joint - hand_trans    
        try:
            pointcloud = np.load(osp.join(self.point_dir, points_name))
            pointcloud = torch.tensor(pointcloud, dtype = torch.float32) 
            pointcloud[:, :3] = pointcloud[:, :3] - hand_trans
            if self.scale is not None:
                pointcloud[:, :3] = pointcloud[:, :3] * self.scale
                obj_centre = obj_centre * self.scale


            
            if self.load_mesh:
                mesh_name = self.meta['images'][idx]['file_name'] + '.obj'
                mesh = trimesh.load_mesh(osp.join(self.mesh_dir, mesh_name))
                pose = torch.tensor(self.meta['annotations'][idx]['obj_transform'])
                pose[:3,3] = pose[:3,3] - hand_trans
                out_pack = {'pc':pointcloud, 'mesh_vert':torch.tensor(np.array(mesh.vertices))- hand_trans, 'mesh_face':torch.tensor(np.array(mesh.faces)), 'pose':pose, 'hand_joint':hand_joint, 'obj_centre':obj_centre}
                return out_pack
            else:
                out_pack = {'pc':pointcloud, 'hand_joint':hand_joint, 'obj_centre':obj_centre, 'filename':self.meta['images'][idx]['file_name']}
                return out_pack
        except Exception as e:
            print(idx, e)
            return self.__getitem__(idx+1)

class dexycb_testfullfeed(Dataset):  
    def __init__(self, data_dir = osp.join(osp.dirname(__file__), "data"),mano_root = os.path.join(os.path.dirname(__file__),'..', 'manopth','mano','models'), load_mesh = False, pc_sample = 1024, data_sample = None, scale_input=False, scale = None, precept = False):
        dexycb_dir = '/home/liyuan/DexYCB/'
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset('s0', "test")
        self.scale_input = scale_input
        self.scale = scale
        config_file = osp.join(data_dir,"dexycb_test_s0.json")
        self.hand_pred_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'hand_pose_results')
        self.obj_pred_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'obj_pose_results')
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.data_sample = data_sample
        if data_sample is not None:
            sample_dir = osp.join(data_dir, "test_sample_{}.json".format(data_sample))
            with open(sample_dir, 'r') as f:
                sample = json.load(f)
                all_indices = [torch.tensor(i) for i in sample.values()]
                self.sample = torch.cat(all_indices)
            self.len = len(self.sample)
        else:
            self.len = len(self.meta['images'])
            self.sample = None

        self.load_mesh = load_mesh
        self.mano_root = mano_root
        self.precept = precept
        
        self.pred_obj_mesh_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'mesh')
        self.pred_hand_mesh_dir = osp.join(osp.dirname(__file__), '..', 'CtcSDF_v2', 'hmano_osdf', 'mesh_hand')
        self.mesh_dir = osp.join(data_dir, 'mesh_data', 'mesh_obj')
        self.point_dir = osp.join(data_dir, 'points_{}'.format(pc_sample))
        self.mano_layer = ManoLayer(
                    ncomps = 45,
                    side = 'right',
                    mano_root= self.mano_root,
                    use_pca=False,
                    flat_hand_mean=True
                    )
        self.mano_layer_pca = ManoLayer(
                    ncomps = 15,
                    side = 'right',
                    center_idx = 0,
                    mano_root= self.mano_root,
                    use_pca=True,
                    flat_hand_mean=True
                    )
        self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)),'closed_fmano.npy')))
    def __len__(self):
        return self.len

    def __getitem__(self, idx_):
        if self.sample is not None:
            idx = self.sample[idx_].item()
        else:
            idx = idx_
        s0_id = self.meta['images'][idx]['id']
        file_name = self.meta['images'][idx]['file_name']
        print("s0_id", self.meta['images'][idx]['file_name'])
        color = cv2.imread(self.getdata[s0_id]['color_file'])
        ycb_id = torch.tensor(self.meta['annotations'][idx]['ycb_id'], dtype = torch.long)
        dexycb_s0_id = torch.tensor(s0_id, dtype = torch.long)
        try:
            if self.precept:
                with open(osp.join(self.hand_pred_dir, file_name + '.json'), 'r') as f:
                    hand_pred_pose = json.load(f)
                cam_extr = torch.tensor(hand_pred_pose['cam_extr'], dtype = torch.float32)
                hand_mesh = trimesh.load_mesh(osp.join(self.pred_hand_mesh_dir, file_name + '_hand.ply'))
                obj_mesh = trimesh.load_mesh(osp.join(self.pred_obj_mesh_dir, file_name + '_obj.ply'))
                obj_centre = torch.tensor(obj_mesh.center_mass, dtype = torch.float32)
                hand_joint = torch.tensor(hand_pred_pose['joints'], dtype = torch.float32)
                hand_trans = 0
                global_trans = torch.tensor(hand_pred_pose['global_trans'], dtype = torch.float32)
                # mano_pose = torch.tensor(hand_pred_pose['pcas'], dtype = torch.float32).view(1, -1)
                # mano_shape = torch.tensor(hand_pred_pose['shape'], dtype = torch.float32).view(1, -1)
                # verts, th_jtr, th_full_pose, th_results_global, center_joint, th_trans = self.mano_layer_pca(th_pose_coeffs = mano_pose[:,0:18], th_betas = mano_shape, th_trans=torch.tensor([0.0,0.0,0.0]), root_palm=False)  
                # transform = torch.eye(4)
                # transform[:3, :3] = cam_extr
                # th_trans_ = (transform @ th_trans.squeeze(0).transpose(1,0)).transpose(1,0)
                # # print(th_trans_)
                # # new_trans = torch.eye(4)
                # # new_trans[:3, :3] = th_trans_[:3, :3]
                new_trans = global_trans.reshape(16,4,4)[0]
                new_trans[:3,3] = (cam_extr @ hand_joint[0].reshape(1,-1).transpose(1,0)).transpose(1,0)

                pose = torch.eye(4, dtype = torch.float32)
                return dict(color_img=color, s0 = dexycb_s0_id, file_name = file_name), dict(verts=torch.tensor(np.array(obj_mesh.vertices), dtype=torch.float32)-hand_trans, faces=torch.tensor(np.array(obj_mesh.faces)), pose=pose, centre=obj_centre, hand_trans = hand_trans), dict(verts=torch.tensor(np.array(hand_mesh.vertices, dtype=np.float32)), faces=self.faces, joint=hand_joint, transformation=new_trans.squeeze(0))
            else:

                hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
                hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32)
                hand_joint = hand_joint - hand_trans
                obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
                mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32).view(1, -1)
                mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32).view(1, -1)
                verts, th_jtr, th_full_pose, th_results_global, center_joint, th_trans = self.mano_layer(th_pose_coeffs = mano_pose[:,0:48], th_betas = mano_shape, th_trans=torch.tensor([0.0,0.0,0.0]), root_palm=False)  
                obj_centre = obj_centre - hand_trans


                mesh_name = self.meta['images'][idx]['file_name'] + '.obj' #already transformed
                mesh = trimesh.load_mesh(osp.join(self.mesh_dir, mesh_name))
                pose = torch.tensor(self.meta['annotations'][idx]['obj_transform'])
                return dict(color_img=color, ycb = ycb_id, s0 = dexycb_s0_id, file_name = file_name), dict(verts=torch.tensor(np.array(mesh.vertices))-hand_trans, faces=torch.tensor(np.array(mesh.faces)), pose=pose, centre=obj_centre, hand_trans = hand_trans), dict(verts=verts.squeeze(0), faces=self.faces, joint=hand_joint, transformation=th_trans.squeeze(0))

        except Exception as e:
            print(idx_, e)
            return self.__getitem__(idx_+1)
def write_json(data, filename):
    
    def convert_tensor_to_list(data):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.tolist()
        return data
    file_name = osp.join(osp.dirname(osp.abspath(__file__)), "data", filename+".json")
    with open(file_name, 'w') as f:
        json.dump(convert_tensor_to_list(data), f, indent=4)

def image_inspection(images):
    global im_id
    im_id = 0
    fig, ax = plt.subplots()
    im = images[im_id][:,:,::-1]
    im_display = ax.imshow(im)

    def update(event):
        global im_id
        if event.key == 'right':
            im_id = (im_id + 1) % len(images)
        elif event.key == 'left':
            im_id = (im_id - 1) % len(images)
        im = images[im_id][:,:,::-1]
        im_display.set_data(im)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('key_press_event', update)
    plt.show()
        
    
class dexycb_new(Dataset):
    def __init__(self, sdf_scaler, double_input = False, data_dir = osp.join(osp.dirname(__file__), "data"), task = "train", pc_sample = 2048, clamp = 0.05, dexycb_backend = False):
        config_file = osp.join(data_dir,"dexycb_{}_s0.json".format(task))
        self.clamp = clamp
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.point_dir = osp.join(data_dir, "points_{}".format(pc_sample))
        self.sdf_dir = osp.join(data_dir, "sdf_data")
        self.sdf_scaler = sdf_scaler
        self.sample_num = pc_sample 
        self.double_input = double_input
        self.backend = dexycb_backend
        if self.backend:
            os.environ["DEX_YCB_DIR"] = '/home/liyuan/DexYCB/'
            self.getdata = DexYCBDataset('s0', task)
    
    def __len__(self):
        return len(self.meta['images'])
    
    def __getitem__(self,idx):
        sdf_name = self.meta['images'][idx]['file_name'] + '.npz'
        points_name = self.meta['images'][idx]['file_name'] + '.npy'
  
        sdf_obj = np.load(osp.join(self.sdf_dir, "sdf_obj", sdf_name))
        obj_sdf, obj_label = self.unpack_sdf(sdf_obj, idx, samples = self.sample_num, clamp = self.clamp, source = 'obj')
        if self.backend:
            s0_id = self.meta['images'][idx]['id']
            sample = self.getdata[s0_id]
            label = np.load(sample['label_file'])
            pose_y = label['pose_y'][sample['ycb_grasp_ind']]
            # obj_index = sample['ycb_ids'][sample['ycb_grasp_ind']]
            obj_centre = torch.tensor(pose_y[:3,3], dtype = torch.float32)
            hand_joint = torch.tensor(label['joint_3d'], dtype = torch.float32).squeeze(0)
            pose_m = label['pose_m']
            hand_trans = torch.tensor(pose_m[:, 48:51], dtype = torch.float32).squeeze(0)
            mano_pose = torch.tensor(pose_m[:, 0:48], dtype = torch.float32).squeeze(0)
            mano_shape = torch.tensor(sample['mano_betas'], dtype = torch.float32).squeeze(0)
        else:            
            hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
            obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
            hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32).squeeze(0)
            mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32)
            mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32)
        scale = torch.tensor(self.meta['annotations'][idx]['sdf_scale'], dtype = torch.float32)
        offset = torch.tensor(self.meta['annotations'][idx]['sdf_offset'], dtype = torch.float32)

        obj_sdf[:, :3] = obj_sdf[:, :3]/scale - offset #transform the sdf pointcloud back to the camera frame
        #print("value check", obj_sdf[:,:3].max(), obj_sdf[:,:3].min())         
        try:
            pointcloud = np.load(osp.join(self.point_dir, points_name))
            pointcloud = torch.tensor(pointcloud, dtype = torch.float32)
            pcd_centre = pointcloud[:, :3].mean(dim = 0)
            if self.double_input:
                pointcloud_original = pointcloud.clone()
                obj_centre_original = obj_centre.clone()
                pointcloud_original[:, :3] = (pointcloud_original[:, :3] - pcd_centre)
                obj_centre_original = obj_centre_original - pcd_centre
            #centre the pointcloud
            pointcloud[:, :3] = (pointcloud[:, :3] - pcd_centre)*self.sdf_scaler/2
            obj_centre = (obj_centre - pcd_centre)*self.sdf_scaler/2
            hand_joint = (hand_joint - pcd_centre)#*self.sdf_scaler
            obj_sdf[:, :3] = (obj_sdf[:, :3] - pcd_centre)*self.sdf_scaler/2
            obj_sdf[:, 3:] = obj_sdf[:, 3:]/scale*(self.sdf_scaler/2)
            hand_trans = (hand_trans - pcd_centre)#*self.sdf_scaler
            
            vote = torch.zeros(pointcloud.shape, dtype = torch.float32)
            vote[:,:3] = pointcloud[:,:3]
            vote[:, 3:] = obj_centre - pointcloud[:, :3]
            
            input_pack = {'pc':pointcloud, 'obj_xyz':obj_sdf[:, :3]}
            mano_pack = {"pose":mano_pose, "shape":mano_shape, "joint":hand_joint, "trans": hand_trans}
            sdf_pack = {"obj_sdf":obj_sdf, "obj_label":obj_label, "obj_centre":obj_centre, "vote":vote, "pcd_centre":pcd_centre}
            if self.double_input:
                input_pack = {'pc':pointcloud, 'obj_xyz':obj_sdf[:, :3], 'pc_original':pointcloud_original, 'obj_xyz_original':obj_centre_original}
            return input_pack, mano_pack, sdf_pack
        except Exception as e:
            return self.__getitem__(idx+1)
            
    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)
        if clamp is not None:
            mask_pos = (pos[...,-1] <= clamp)
            mask_neg = (neg[...,-1] >= -clamp)
            pos = pos[mask_pos]
            neg = neg[mask_neg]
            lab_pos = lab_pos[mask_pos]
            lab_neg = lab_neg[mask_neg]


        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:    
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label    

        
class dexycb_separate(Dataset):
    def __init__(self, sdf_scaler, data_dir = osp.join(osp.dirname(__file__), "data"), task = "train", pc_sample = 1024, sdf_sample = 1024, clamp = 0.1, dexycb_backend = False, xyz_scale = True):
        config_file = osp.join(data_dir,"dexycb_{}_s0.json".format(task))
        self.clamp = clamp
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.point_dir = osp.join(data_dir, "points_separate_{}".format(pc_sample))
        self.sdf_dir = osp.join(data_dir, "sdf_data")
        self.sdf_scaler = sdf_scaler
        self.sample_num = sdf_sample 
        self.backend = dexycb_backend
        self.xyz_scale = xyz_scale
        if self.backend:
            os.environ["DEX_YCB_DIR"] = '/home/liyuan/DexYCB/'
            self.getdata = DexYCBDataset('s0', task)
    
    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self,idx):
        sdf_name = self.meta['images'][idx]['file_name'] + '.npz'
        points_name = self.meta['images'][idx]['file_name'] + '.npy'
        
        sdf_obj = np.load(osp.join(self.sdf_dir, "sdf_obj", sdf_name))
        obj_sdf, obj_label = self.unpack_sdf(sdf_obj, idx, samples = self.sample_num, clamp = self.clamp, source = 'obj')
        if self.backend:
            s0_id = self.meta['images'][idx]['id']
            sample = self.getdata[s0_id]
            label = np.load(sample['label_file'])
            pose_y = label['pose_y'][sample['ycb_grasp_ind']]
            # obj_index = sample['ycb_ids'][sample['ycb_grasp_ind']]
            obj_centre = torch.tensor(pose_y[:3,3], dtype = torch.float32)
            hand_joint = torch.tensor(label['joint_3d'], dtype = torch.float32).squeeze(0)
            pose_m = label['pose_m']
            hand_trans = torch.tensor(pose_m[:, 48:51], dtype = torch.float32).squeeze(0)
            mano_pose = torch.tensor(pose_m[:, 0:48], dtype = torch.float32).squeeze(0)
            mano_shape = torch.tensor(sample['mano_betas'], dtype = torch.float32).squeeze(0)
        else:            
            hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
            obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
            hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32).squeeze(0)
            mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32)
            mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32)
        scale = torch.tensor(self.meta['annotations'][idx]['sdf_scale'], dtype = torch.float32)
        offset = torch.tensor(self.meta['annotations'][idx]['sdf_offset'], dtype = torch.float32)

        obj_sdf[:, :3] = obj_sdf[:, :3]/scale - offset

        mano_pose_ = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32)
        print('if this is same',torch.allclose(mano_pose, mano_pose_, atol = 1e-2))
        obj_centre_ = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
        print(obj_centre, obj_centre_)
        print('if this is same,obj',torch.allclose(obj_centre, obj_centre_, atol = 1e-2))
        hand_joint_ = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32).squeeze(0)
        print('if this is same,hand',torch.allclose(hand_joint, hand_joint_, atol = 1e-5))
        pose_y_ = torch.tensor(self.meta['annotations'][idx]['obj_transform'], dtype = torch.float32)
        print('if this is same,pose',torch.allclose(pose_y_[:3,:], torch.tensor(pose_y,dtype = torch.float32).squeeze(0), atol = 1e-1))
        print(pose_y, pose_y_[:3,:])
        try:
            obj_pc_ = np.load(osp.join(self.point_dir, "obj", points_name))
            obj_pc = torch.tensor(obj_pc_, dtype = torch.float32)
            # print("unique", obj_pc[:, 3].unique())
            obj_pc[:, :3],obj_pc_centre,xyz_scale = self.normalize(obj_pc[:, :3])
            if self.xyz_scale:
                self.sdf_scaler = 1/xyz_scale
            self.sdf_scaler = torch.clamp(self.sdf_scaler, None, 15)

            obj_centre = (obj_centre - obj_pc_centre)*self.sdf_scaler

            
            obj_sdf[:, :3] = (obj_sdf[:, :3] - obj_pc_centre)*self.sdf_scaler
            obj_sdf[:, 3:] = (obj_sdf[:, 3:]/scale)*(self.sdf_scaler*10)
            

            #hand 
            hand_pc_ = np.load(osp.join(self.point_dir, "hand", points_name))
            hand_pc = torch.tensor(hand_pc_, dtype = torch.float32)
            hand_pc[:, :3], hand_pc_centre, hand_scale = self.normalize(hand_pc[:, :3], set_back = False)

            hand_joint = (hand_joint - hand_trans)

            hand_trans = (hand_trans- hand_pc_centre)

            

            
            input_pack = {'obj_pc':obj_pc, 'hand_pc':hand_pc, 'obj_xyz':obj_sdf[:, :3], 
                          "obj_pc_centre":obj_pc_centre, "hand_pc_centre":hand_pc_centre, "scale":self.sdf_scaler, "hand_scale":hand_scale}
            
            mano_pack = {"pose":mano_pose, "shape":mano_shape, "joint":hand_joint, "trans": hand_trans}
            sdf_pack = {"obj_sdf":obj_sdf, "obj_label":obj_label, "obj_centre":obj_centre}
            
            return input_pack, mano_pack, sdf_pack
        except Exception as e:
            # traceback.print_exc()
            return self.__getitem__(idx+1)
    
    def normalize(self, tensor, set_back = False):
        centre = tensor.mean(dim = 0)
        normalized_points = tensor - centre
        max_dist = torch.norm(normalized_points, dim=-1).max()
        unit_sphere_points = normalized_points / (max_dist + 1e-6)  # Avoid division by zero
        if set_back:
            unit_sphere_points = unit_sphere_points + centre
        return unit_sphere_points, centre, max_dist + 1e-6

    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)
        if clamp is not None:
            mask_pos = (pos[...,-1] <= clamp)
            mask_neg = (neg[...,-1] >= -clamp)
            pos = pos[mask_pos]
            neg = neg[mask_neg]
            lab_pos = lab_pos[mask_pos]
            lab_neg = lab_neg[mask_neg]


        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:    
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label    

class dexycb_separate_test(Dataset):
    def __init__(self, sdf_scaler, data_dir = osp.join(osp.dirname(__file__), "data"), task = "test", pc_sample = 1024, sdf_sample = 1024, clamp = 0.1, dexycb_backend = False, xyz_scale = True):
        config_file = osp.join(data_dir,"dexycb_{}_s0.json".format(task))
        self.clamp = clamp
        with open(config_file, 'r') as f:
            self.meta = json.load(f)
        self.point_dir = osp.join(data_dir, "points_separate_{}".format(pc_sample))
        self.sdf_dir = osp.join(data_dir, "sdf_data")
        self.sdf_scaler = sdf_scaler
        self.sample_num = sdf_sample 
        self.backend = dexycb_backend
        self.xyz_scale = xyz_scale
        if self.backend:
            os.environ["DEX_YCB_DIR"] = '/home/liyuan/DexYCB/'
            self.getdata = DexYCBDataset('s0', task)
    
    def __len__(self):
        return len(self.meta['images'])

    def __getitem__(self,idx):
        points_name = self.meta['images'][idx]['file_name'] + '.npy'
        

        if self.backend:
            s0_id = self.meta['images'][idx]['id']
            sample = self.getdata[s0_id]
            label = np.load(sample['label_file'])
            pose_y = label['pose_y'][sample['ycb_grasp_ind']]
            # obj_index = sample['ycb_ids'][sample['ycb_grasp_ind']]
            obj_centre = torch.tensor(pose_y[:3,3], dtype = torch.float32)
            hand_joint = torch.tensor(label['joint_3d'], dtype = torch.float32).squeeze(0)
            pose_m = label['pose_m']
            hand_trans = torch.tensor(pose_m[:, 48:51], dtype = torch.float32).squeeze(0)
            mano_pose = torch.tensor(pose_m[:, 0:48], dtype = torch.float32).squeeze(0)
            mano_shape = torch.tensor(sample['mano_betas'], dtype = torch.float32).squeeze(0)
        else:            
            hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
            obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
            hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32).squeeze(0)
            mano_pose = torch.tensor(self.meta['annotations'][idx]['hand_poses'], dtype = torch.float32)
            mano_shape = torch.tensor(self.meta['annotations'][idx]['hand_shapes'], dtype = torch.float32)
        scale = torch.tensor(self.meta['annotations'][idx]['sdf_scale'], dtype = torch.float32)
        offset = torch.tensor(self.meta['annotations'][idx]['sdf_offset'], dtype = torch.float32)


        try:
            obj_pc_ = np.load(osp.join(self.point_dir, "obj", points_name))
            obj_pc = torch.tensor(obj_pc_, dtype = torch.float32)
            
            _,_,xyz_scale = self.normalize(obj_pc[:, :3])
            if self.xyz_scale:
                self.sdf_scaler = 2/xyz_scale
            
            
            obj_pc_centre = obj_pc[:,:3].mean(dim = 0)
            obj_pc[:, :3] = (obj_pc[:, :3] - obj_pc_centre)*self.sdf_scaler/2
            obj_centre = (obj_centre - obj_pc_centre)*self.sdf_scaler/2
 
            hand_pc_ = np.load(osp.join(self.point_dir, "hand", points_name))
            hand_pc = torch.tensor(hand_pc_, dtype = torch.float32)
            hand_pc[:, :3], hand_pc_centre, hand_scale = self.normalize(hand_pc[:, :3], set_back = False)

            hand_joint = (hand_joint - hand_trans)

            hand_trans = (hand_trans- hand_pc_centre)


            
            input_pack = {'obj_pc':obj_pc, 'hand_pc':hand_pc, 
                          "obj_pc_centre":obj_pc_centre, "hand_pc_centre":hand_pc_centre, "scale":self.sdf_scaler, "hand_scale":hand_scale}
            
            mano_pack = {"pose":mano_pose, "shape":mano_shape, "joint":hand_joint, "trans": hand_trans}
            sdf_pack = {"obj_centre":obj_centre}
            
            return input_pack, mano_pack, sdf_pack
        except Exception as e:
            # traceback.print_exc()
            return self.__getitem__(idx+1)
    
    def normalize(self, tensor, set_back = False):
        centre = tensor.mean(dim = 0)
        normalized_points = tensor - centre
        max_dist = torch.norm(normalized_points, dim=-1).max()
        unit_sphere_points = normalized_points / (max_dist + 1e-6)  # Avoid division by zero
        if set_back:
            unit_sphere_points = unit_sphere_points + centre
        return unit_sphere_points, centre, max_dist + 1e-6

    def unpack_sdf(self, sdf_npz, idx,samples = None, clamp = None, source = 'obj'):
        def filter_invalid_sdf(tensor,lab_tenosr,dist):
          keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
          return tensor[keep,:], lab_tenosr[keep,:]
        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]
        
        try:
            pos = remove_nans(torch.from_numpy(sdf_npz['pos']))
            neg = remove_nans(torch.from_numpy(sdf_npz['neg']))
            pos_other = torch.from_numpy(sdf_npz['pos_other'])
            neg_other = torch.from_numpy(sdf_npz['neg_other'])
            if source == 'obj':
                lab_pos = torch.from_numpy(sdf_npz['lab_pos_other'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg_other'])
            else:
                lab_pos = torch.from_numpy(sdf_npz['lab_pos'])
                lab_neg = torch.from_numpy(sdf_npz['lab_neg'])
                
        except Exception as e:
            print('{}{}:failed to load sdf: {}'.format(source,idx, e))
            
        if source == 'hand':
            pos = torch.cat([pos, pos_other], dim = 1)
            neg = torch.cat([neg, neg_other], dim = 1)
        else:
            pos_xyz = pos[:,:3]
            pos_val = pos[:,3]
            pos = torch.cat([pos_xyz, pos_other, pos_val.unsqueeze(1)], dim = 1)

            neg_xyz = neg[:,:3]
            neg_val = neg[:,3]
            neg = torch.cat([neg_xyz, neg_other, neg_val.unsqueeze(1)], dim = 1)

        pos, lab_pos = filter_invalid_sdf(pos, lab_pos, 2.0)
        neg, lab_neg = filter_invalid_sdf(neg, lab_neg, 2.0)
        if clamp is not None:
            mask_pos = (pos[...,-1] <= clamp)
            mask_neg = (neg[...,-1] >= -clamp)
            pos = pos[mask_pos]
            neg = neg[mask_neg]
            lab_pos = lab_pos[mask_pos]
            lab_neg = lab_neg[mask_neg]


        half = int(samples/2)
        rand_idx_pos = (torch.rand(half) * pos.shape[0]).long()
        rand_idx_neg = (torch.rand(half) * neg.shape[0]).long()

        samples_pos = torch.index_select(pos, 0, rand_idx_pos)
        samples_neg = torch.index_select(neg, 0, rand_idx_neg)
        samples_lab_pos = torch.index_select(lab_pos, 0, rand_idx_pos)
        samples_lab_neg = torch.index_select(lab_neg, 0, rand_idx_neg)

        hand_part_pos = samples_lab_pos[:, 0]
        hand_part_neg = samples_lab_neg[:, 0]

        sample = torch.cat([samples_pos, samples_neg], dim = 0)
        label = torch.cat([hand_part_pos, hand_part_neg], dim = 0)
        
        if clamp is not None:    
            label[sample[:, 3] < -clamp] = -1
            label[sample[:, 3] > clamp] = -1
        #return sample, lable

        if source == 'obj':
            label[:] = -1
        return sample, label    

class dexycb_trainfullfeed(Dataset):
    def __init__(self, data_dir = osp.join(osp.dirname(__file__), "data"),mano_root = os.path.join(os.path.dirname(__file__),'..', 'manopth','mano','models')):
        dexycb_dir = '/home/liyuan/DexYCB/'
        os.environ["DEX_YCB_DIR"] = dexycb_dir
        self.getdata = DexYCBDataset('s0', "train")

        config_file = osp.join(data_dir,"dexycb_train_s0.json")
        with open(config_file, 'r') as f:
            self.meta = json.load(f)

        self.len = len(self.meta['images'])
        self.mano_root = mano_root
        self.data_sample = 0
        self.mano_layer = ManoLayer(
                    ncomps = 45,
                    side = 'right',
                    mano_root = os.path.join(os.path.dirname(__file__), '..', 'manopth','mano','models'),
                    use_pca=True,
                    flat_hand_mean=False
                    )
        self.faces = torch.LongTensor(np.load(osp.join(osp.dirname(osp.abspath(__file__)),'closed_fmano.npy')))
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx_):

        idx = idx_
        s0_id = self.meta['images'][idx]['id']
        sample = self.getdata[s0_id]
        color = cv2.imread(sample['color_file'])
        ycbi = sample['ycb_grasp_ind']
        label = np.load(sample['label_file'])
        pose_y = label['pose_y'][ycbi]
        pose_m = torch.tensor(label['pose_m'], dtype=torch.float32)
        bates = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)
        verts,_,th_trans = self.mano_layer( pose_m[:, 0:48],bates, torch.zeros_like(pose_m[:, 48:51]))
    
        hand_trans = torch.tensor(self.meta['annotations'][idx]['hand_trans'], dtype = torch.float32)
        obj_centre = torch.tensor(self.meta['annotations'][idx]['obj_center_3d'], dtype = torch.float32)
        hand_joint = torch.tensor(self.meta['annotations'][idx]['hand_joints_3d'], dtype = torch.float32)
        verts /= 1000
        verts = verts.view(778,3)
        pose = torch.tensor(np.vstack((pose_y, np.array([[0, 0, 0, 1]], dtype=np.float32))))
        pose[:3,3] = pose[:3,3] - hand_trans
        # ycb_id = torch.tensor(self.meta['annotations'][idx]['ycb_id'], dtype = torch.long)
        # dexycb_s0_id = torch.tensor(s0_id, dtype = torch.long)

        # print("joints", hand_joint-hand_trans)
        # print("mano_joints", _)
        # offset = torch.tensor(self.meta['annotations'][idx]['sdf_offset'], dtype = torch.float32)   
        obj_centre = obj_centre - hand_trans
        hand_joint = hand_joint - hand_trans
        #th_trans[:,:3,3] = th_trans[:,:3,3] #- hand_trans

        mesh = trimesh.load_mesh(self.getdata.obj_file[sample['ycb_ids'][ycbi]])
        mesh.apply_transform(pose)


            
        return dict(color_img=color), dict(verts=torch.tensor(np.array(mesh.vertices)), faces=torch.tensor(np.array(mesh.faces)), pose=pose, centre=obj_centre), dict(verts=verts, faces=self.faces, joint=hand_joint, transformation=th_trans.squeeze(0))


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # import matplotlib.pyplot as plt
    # dexycb = dexycb_testfullfeed(load_mesh=True, data_sample=None)
    # dl = DataLoader(dexycb, batch_size = 1, shuffle = False)
    # ycb_id_collect = []
    # image_stack = []
    # for i, (input_pack, mesh_pack, mano_pack) in enumerate(dl):
    #     # color = input_pack['color_img']
    #     # image_stack.append(color)
    #     ycb_id = input_pack['ycb']
    #     ycb_id_collect.append(ycb_id)
        
    # # image_stack = torch.stack(image_stack).squeeze().numpy()
    # # print(image_stack.shape)
    # # image_inspection(image_stack)
    # #     ycd_id = input_pack['ycb']
    # #     ycb_id_collect.append(ycd_id)
    # ycb_id_collect = torch.stack(ycb_id_collect)
    # ycb_unique = torch.unique(ycb_id_collect) 
    
    # print(ycb_id_collect)
    # print(ycb_unique)
    # print(torch.bincount(ycb_id_collect.squeeze()))
    # sample_num = 5
    # bin_indices = {}
    # for ycb in ycb_unique:
    #     value = torch.where(ycb_id_collect == ycb)[0]
    #     if value.shape[0] > sample_num:
    #         shuffled_indices = value[torch.randperm(value.shape[0])]
    #         bin_indices[ycb.item()] = shuffled_indices[:sample_num]
    #     else:
    #         bin_indices[ycb.item()] = value
    
    # for value, indices in bin_indices.items():
    #     print(f"Value: {value}, Indices: {indices.tolist()}")

    # write_json(bin_indices, "test_sample_{}".format(sample_num))


    # def furtheset_dist(points):
    #     squared_norms = torch.sum(points**2, dim = 1).reshape(-1,1)
    #     pairwise_distances_squared = squared_norms + squared_norms.T - 2 * torch.mm(points, points.T)
    #     pairwise_distances = torch.sqrt(torch.clamp(pairwise_distances_squared, min=0))
    #     max_distance = pairwise_distances.max()
    #     return max_distance


    # # from .utils import data_viz
    sdf_scaler = 6.2
    dexycb = dexycb_separate(task='train',sdf_scaler = sdf_scaler, pc_sample = 1024, clamp=0.05,dexycb_backend=True)
    dl = DataLoader(dexycb, batch_size = 1, shuffle = True, num_workers=4)
    input_pack,mano_pack,sdf_pack = next(iter(dl))
    print(sdf_pack['obj_sdf'][...,-1].max(), sdf_pack['obj_sdf'][...,-1].min())
    print(input_pack['scale'])
    d = []
    for i, (input_pack, mano_pack, sdf_pack) in tqdm(enumerate(dl), total = len(dexycb)):
        #pc = input_pack['pc'][...,:3].squeeze()
        # norm = sdf_pack['obj_sdf'][...,:-1].squeeze()
        # norm = torch.norm(pc, dim = 1)
        # norm = torch.norm(input_pack['hand_pc'][...,:3], dim = -1)
        # centre = torch.norm(sdf_pack['scale'], dim = -1)
        d.append(input_pack['scale'])
        # print(norm)
        # print(sdf_pack['obj_sdf'].size())   
        # if i == 1000:
        #     break
            #print(pc)
            #print(pc[:,:3].norm(-1).max(), pc[:,:3].norm(-1).min(), pc[:,:3].norm(-1).mean())
    d = torch.stack(d)
    d.squeeze()
    print(d)
    print(d.size())
    print(d.max(), d.min(), d.mean())
    print(d.argmax(), d.argmin())
    
    
 

    # dl = DataLoader(dexycb, batch_size = 1, shuffle = False)
    # input_pack,mano_pack,sdf_pack = next(iter(dl))
    # print('pc size',input_pack['pc'].size())
    # print('xyz size',input_pack['obj_xyz'].size())
    # print('mano pose size',mano_pack['pose'].size())
    # print('mano shape size',mano_pack['shape'].size())
    # print('hand joins size',mano_pack['joint'].size())
    # print('hand trans size',mano_pack['trans'].size())
    # print('obj sdf size',sdf_pack['obj_sdf'].size())
    # print('obj label size',sdf_pack['obj_label'].size())
    # print('obj centre size',sdf_pack['obj_centre'].size())
    # print('vote map size',sdf_pack['vote'].size())






    
        
