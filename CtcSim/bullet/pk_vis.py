"""Visualize the result from optimizer"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from CtcBot import CtcRobot, Shadow, Allegro, Barrett, Robotiq
import torch
from CtcObj import object_sdf
from plotly import graph_objects as go
import trimesh as tm
import numpy as np
from typing import List, Union
from PB_test import ycb_opt_fetcher
import matplotlib.pyplot as plt
import pyrender
import matplotlib.colors as mcolors
import OpenGL.GL as gl  # Import OpenGL for reading pixel data
import json
"""
what to visualize:
1. robot
2. object

how to visualize:
1. initialize the robot (and object)

"""

class Visualizer:
    def __init__ (self,
         robot: CtcRobot,
         ):
        
        self.robot = robot
        self.robot.init_q()

    
    def lines(self, 
              start: np.ndarray, 
              end: np.ndarray, 
              color: str = 'red', 
              width: float = 2.0,
              ):
        N, _ = start.shape
        x1, y1, z1 = start[:,0], start[:,1], start[:,2]
        x2, y2, z2 = end[:,0], end[:,1], end[:,2]
        x_lines = []
        y_lines = []
        z_lines = []
        for i in range(N):
            x_lines.extend([x1[i], x2[i], None])
            y_lines.extend([y1[i], y2[i], None])
            z_lines.extend([z1[i], z2[i], None])
        lines = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            line=dict(color=color, width=width)
        )
        return lines

    def mesh(self, 
             mesh_v: np.ndarray, 
             mesh_f: np.ndarray, 
             color: str = 'white', 
             opacity: float = 0.5):
        mesh = go.Mesh3d(
            x=mesh_v[:, 0],
            y=mesh_v[:, 1],
            z=mesh_v[:, 2],
            i=mesh_f[:, 0],
            j=mesh_f[:, 1],
            k=mesh_f[:, 2],
            color=color,
            opacity=opacity
        )
        return mesh

    def scatter(self, 
                points: np.ndarray, 
                size: float, 
                color: str = 'red'):
        scatter = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=size,  # Adjust marker size here
                color=color  # Adjust marker color here
            )
        )
        return scatter

    def go_graph(self, 
                 data: List[go.Scatter3d],
                 transparent_background: bool = False):
        fig = go.Figure(data = data)
        camera = dict(
            eye=dict(x=1.87, y=0.88, z=0.64),
            up = dict(x=0, y=0, z=1),
            center = dict(x=0, y=0, z=0)
        )
        fig.update_layout(scene_camera=camera)
        if transparent_background:
            fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False,  # Hides the x-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent x-axis ticks
                yaxis=dict(showbackground=False,  # Hides the y-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent y-axis ticks
                zaxis=dict(showbackground=False,  # Hides the z-axis background
                            tickfont=dict(color='rgba(0,0,0,0)')),  # Transparent z-axis ticks
            ),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
            legend=dict(font=dict(color='rgba(0,0,0,0)'))  # Transparent legend
            )
        fig.show() 
    
    def draw(self,
             joint_values: torch.tensor,
             data_dict: List[go.Scatter3d],
             transparent_background: bool = False
             ):
        if joint_values is not None:
            data = self.robot.get_mesh_updated(q =joint_values, opacity=0.8, color='lightblue')
        else:
            data = []
            
        if data_dict is not None:
            for mesh in data_dict:
                data.append(mesh)
        
        for go_mesh in data:
            x = go_mesh.x
            y = go_mesh.y
            z = go_mesh.z
            
            go_mesh.x = -z
            go_mesh.y = x
            go_mesh.z = -y
        self.go_graph(data=data, transparent_background=transparent_background)
    
    def convert_to_pyrender(self,
                            go_meshes: List[go.Mesh3d],
                            color: Union[np.ndarray, str],):
        trimesh_meshes = []

        for go_mesh in go_meshes:
            # vertices = np.array([-1*go_mesh.z, go_mesh.x, -1*go_mesh.y]).T
            x = go_mesh.x
            y = go_mesh.y
            z = go_mesh.z
            
            vertices = np.array([x, -y, -z]).T
            
            faces = np.array([go_mesh.i, go_mesh.j, go_mesh.k]).T
            trimesh_mesh = tm.Trimesh(vertices=vertices, faces=faces)

            if isinstance(color, str):
                color = mcolors.to_rgba(color, alpha=1.0)          
            if isinstance(color, np.ndarray):
                color = color.tolist() + [1.0]
            trimesh_mesh.visual.vertex_colors = color

            # mesh = pyrender.Mesh.from_trimesh(tm.Trimesh(vertices, faces), smooth=False)
            trimesh_meshes.append(trimesh_mesh)
        
        pyrender_meshes = [pyrender.Mesh.from_trimesh(trimesh_mesh) for trimesh_mesh in trimesh_meshes]
        return pyrender_meshes
        

    def pyrender_draw(self,
                      meshes: dict,
                      cam: dict,
                      pose: np.ndarray = np.eye(4),
                      filename: str = "screenshot.png"):
        scene = pyrender.Scene()
        cam = pyrender.IntrinsicsCamera(cam['fx'], cam['fy'], cam['cx'], cam['cy'])
        scene.add(cam, pose=pose)
        hand = self.convert_to_pyrender(meshes["hand"], color="skyblue")
        obj = self.convert_to_pyrender(meshes["obj"], color="silver")
        for mesh in hand:
            scene.add(mesh)
        for mesh in obj:
            scene.add(mesh)

        viewer = ScreenshotSaver(scene, use_raymond_lighting=True, viewport_size=(3840 , 2160))

class ScreenshotSaver(pyrender.Viewer):
    def __init__(self, scene, **kwargs):

        super().__init__(scene, **kwargs)  # Call the base class constructor

    def on_key_press(self, key, modifiers):
        if key == ord("s"):  # Press 'S' to save the screenshot
            width, height = self._viewport_size

            # Bind the framebuffer and read pixels
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            color_buffer = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            
            # Convert to NumPy array and reshape (Flip vertically to correct OpenGL coordinates)
            color = np.frombuffer(color_buffer, dtype=np.uint8).reshape(height, width, 3)[::-1]

            plt.imsave(sim_filename, color)

def vis_deepsdf():
    import json
    dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset_sdf")
    idx = 0
    json_file = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF", "data", "dexycb_test_s0.json")
    hand_mesh_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF_v2", "hmano_osdf", "mesh_hand")
    obj_mesh_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF_v2", "hmano_osdf", "mesh")
    with open(json_file, 'r') as f:
        data = json.load(f) 
    file_name = data['images'][idx]['file_name']
    hand_mesh = tm.load(os.path.join(hand_mesh_dir, file_name + "_hand.ply"), force="mesh", process=False)
    obj_mesh = tm.load(os.path.join(obj_mesh_dir, file_name + "_obj.ply"), force="mesh", process=False)
    
    

if __name__ == "__main__":
        dir_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asset_comp") 
        idx_ = 100  # 26 20
        robot = Robotiq(batch=1, device="cpu")
        expname = "genhand" 
        # for model in ["Shadow", "Allegro", "Barrett", "Robotiq"]:
        #     if model == "Shadow":
        #         robot = Shadow(batch=1, device="cuda:0")
        #         data_sample = 5
        #         idx = idx_
        #     if model == "Allegro":
        #         robot = Allegro(batch=1, device="cuda:0")
        #         data_sample = 20
        #         i_1 = idx_//5
        #         i_2 = idx_%5
        #         idx = i_1*data_sample + i_2
        #     if model == "Barrett":
        #         robot = Barrett(batch=1, device="cuda:0")
        #         data_sample = 20
        #         i_1 = idx_//5
        #         i_2 = idx_%5
        #         idx = i_1*data_sample + i_2
        #     if model == "Robotiq":
        #         robot = Robotiq(batch=1, device="cuda:0")
        #         data_sample = 20
        #         i_1 = idx_//5
        #         i_2 = idx_%5
        #         idx = i_1*data_sample + i_2
            
            # idx =39
            # robot = Allegro(batch=1, device="cuda:0")
        data_fetcher = ycb_opt_fetcher(mu=0.9,
                                       SDF_source=False,
                                       load_assert=True, 
                                       data_sample=20, 
                                       repeat=1, 
                                       robot_name=robot.robot_model, 
                                       exp_name=expname, 
                                       test_sdf=True,
                                       )
        # data_fetcher = ycb_opt_fetcher

        data_dict = data_fetcher[idx_]
        file_name = data_dict["file_name"]
        color = data_dict["color"]
        verts = data_dict["hand_verts"]
        faces = data_dict["hand_faces"]
        posey = data_dict["pose_y"]
        q = data_dict["full_q"]
        print("q", q)
        print(verts.shape)
        print(faces.shape)
        vizer = Visualizer(robot)

        #######################################################
        # hand_mesh_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF_v2", "hmano_osdf", "mesh_hand")
        # obj_mesh_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF_v2", "hmano_osdf", "mesh")
        # hand_mesh = tm.load(os.path.join(hand_mesh_dir, file_name + "_hand.ply"), force="mesh", process=False)
        # obj_mesh = tm.load(os.path.join(obj_mesh_dir, file_name + "_obj.ply"), force="mesh", process=False)
        # obj_pose_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF_v2", "hmano_osdf", "obj_pose_results")
        # hand_pose_dir = os.path.join(os.path.dirname(__file__), "..", "..", "CtcSDF_v2", "hmano_osdf", "hand_pose_results")
        # with open(os.path.join(obj_pose_dir, file_name + ".json"), 'r') as f:
        #     obj_pose = json.load(f)
        # with open(os.path.join(hand_pose_dir, file_name + ".json"), 'r') as f:
        #     hand_pose = json.load(f)
        # # cam_ext = np.array(hand_pose["cam_extr"])
        # # hand_centre = (cam_ext @ np.array(hand_pose["rot_center"]).reshape(-1,1).transpose(1,0)).transpose(1,0)
        # obj_center_est = np.array(obj_pose["center"])
        # obj_centre = obj_mesh.center_mass
        # print("obj_centre", obj_centre)
        # print("obj_center_est", obj_center_est)
        # # obj_mesh.vertices -= obj_centre
        # q[:3] = q[:3] + posey[:3, 3]
        # print("q", q)
        # hand_mesh = vizer.mesh(np.array(hand_mesh.vertices), np.array(hand_mesh.faces), color='lightblue')
        # obj_mesh = vizer.mesh(np.array(obj_mesh.vertices), np.array(obj_mesh.faces), color='lightblue')
        ##############################################
        obj_file = data_dict["obj_file"]
        se3 = data_dict["se3"]
        obj_centre = data_dict["obj_centre"]
        obj = tm.load(obj_file, force="mesh", process=False)
        se3[:3, 3] = se3[:3, 3] - obj_centre
        # q[:3] = q[:3] - obj_centre
        obj.apply_transform(se3)

        obj = tm.load(obj_file, force="mesh", process=False)
        obj.apply_transform(posey)

        meshes = [vizer.mesh(verts, faces, 'lightblue'), vizer.mesh(np.array(obj_.vertices), np.array(obj_.faces))]
        meshes.append()
        meshes.append(vizer.mesh(np.array(obj_.vertices), np.array(obj_.faces)))
        hand_mesh = vizer.mesh(verts, faces, color='lightblue')
        obj_mesh = vizer.mesh(np.array(obj.vertices), np.array(obj.faces))
        #if model == "Shadow":
        im_real = color[:, :, ::-1]
        width, height, _ = im_real.shape
        if robot.robot_model == "Shadow":
            plt.imshow(im_real)
            plt.tight_layout()
            plt.show()
            plt.imsave(os.path.join(dir_, str(idx_) + "_real.png"), im_real)
            print("obj_centre", obj_centre)
        robot_mesh = robot.get_mesh_updated(q =torch.tensor(q, dtype=torch.float32).unsqueeze(0), opacity=0.3, color='lightblue')
        #vizer.draw(torch.tensor(q, dtype=torch.float32).unsqueeze(0).to("cuda:0"), [mesh], True)

        #vizer.draw(None, meshes, False)
        cam = dict(fx = data_dict["fx"], fy = data_dict["fy"], cx = data_dict["cx"], cy = data_dict["cy"])
        sim_filename = os.path.join(dir_, str(idx_)+"sdf_sim_{}.png".format(robot.robot_model))
        pose = np.eye(4)
        pose[:3, 3] = pose[:3, 3] + obj_centre
        pose[2, 3] += 1
        vizer.pyrender_draw({"hand": [hand_mesh], "obj": [obj_mesh]}, cam, pose)