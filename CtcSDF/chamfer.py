from multiprocessing import process
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import warnings
import os
"""
This piece of code is directly adapted from DeepSDF repo without any modification.
"""
class ICP_T_S():
    def __init__(self, mesh_source, mesh_target):
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

        self.points_source = self.mesh_source.vertices.copy()
        self.points_target = self.mesh_target.vertices.copy()

    def sample_mesh(self, n=30000, mesh_id='both'):
        if mesh_id == 'source' or mesh_id == 'both':
            self.points_source, _ = trimesh.sample.sample_surface(self.mesh_source, n)
        if mesh_id == 'target' or mesh_id == 'both':
            self.points_target, _ = trimesh.sample.sample_surface(self.mesh_target, n)

        self.offset_source = self.points_source.mean(0)
        self.scale_source = np.sqrt(((self.points_source - self.offset_source)**2).sum() / len(self.points_source))
        self.offset_target = self.points_target.mean(0)
        self.scale_target = np.sqrt(((self.points_target - self.offset_target)**2).sum() / len(self.points_target))

        self.points_source = (self.points_source - self.offset_source) / self.scale_source * self.scale_target + self.offset_target

    def run_icp_f(self, max_iter = 10, stop_error = 1e-3, stop_improvement = 1e-5, verbose=0):
        # Difference with run_icp():
            # run_icp_1 build KDTree only once
            # run_icp() build KDTree in every iteration.
        # Build KDTree for both original target and source point cloud
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        previous_error = error
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            query_source_points = self.points_source*self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index[:,0], :]

            # Find closest source point for each target point:
            query_target_points = (self.points_target - self.trans)/self.scale
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:,0], :]
            closest_source_points = closest_source_points*self.scale + self.trans
            query_target_points = self.points_target

            # Compute current error:
            error = (((query_source_points-closest_target_points)**2).sum() +\
            ((query_target_points-closest_source_points)**2).sum())/(query_source_points.shape[0] + query_target_points.shape[0])
            error = error**0.5
            if verbose >= 1:
                print(i, "th iter, error: ", error)

            if previous_error - error < stop_improvement:
                break
            else:
                previous_error = error

            if error < stop_error:
                break

            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1,1),
                              self.points_source[closest_source_points_index[:,0], :].reshape(-1,1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3),\
                            dtype = np.float) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            # print(closest_target_points.reshape(-1,1).shape)
            # print(query_target_points.reshape(-1,1).shape)
            b = np.vstack([closest_target_points.reshape(-1,1),
                                query_target_points.reshape(-1, 1)])
            # print(A.shape)
            # print(b.shape)
            x = np.linalg.lstsq(A,b)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()

            # query_source_points = self.points_source*self.scale + self.trans
            # closest_source_points = self.points_source[closest_source_points_index[:,0], :]*self.scale + self.trans
            # error = (((query_source_points-closest_target_points)**2).sum() +\
            # ((query_target_points-closest_source_points)**2).sum())/(query_source_points.shape[0] + query_target_points.shape[0])
            # error = error**0.5
            # print(i, "th iter, error: ", error)
    
    def run_icp(self, max_iter = 10, stop_error = 1e-3):
        # Build KDTree for both original target and source point cloud
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            self.source_KDTree = KDTree(self.points_source*self.scale + self.trans)
            query_source_points = self.points_source*self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index[:,0], :]

            # Find closest source point for each target point:
            query_target_points = self.points_target
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:,0], :]*self.scale + self.trans

            # Compute current error:
            error = (((query_source_points-closest_target_points)**2).sum() +\
                    ((query_target_points-closest_source_points)**2).sum())\
                    /(query_source_points.shape[0] + query_target_points.shape[0])
            error = error**0.5
            # print(i, "th iter, error: ", error)

            if error < stop_error:
                break

            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1,1),
                              self.points_source[closest_source_points_index[:,0], :].reshape(-1,1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3),\
                            dtype = np.float) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            # print(closest_target_points.reshape(-1,1).shape)
            # print(query_target_points.reshape(-1,1).shape)
            b = np.vstack([closest_target_points.reshape(-1,1),
                                query_target_points.reshape(-1, 1)])
            # print(A.shape)
            # print(b.shape)
            x = np.linalg.lstsq(A,b)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()

            query_source_points = self.points_source*self.scale + self.trans
            closest_source_points = self.points_source[closest_source_points_index[:,0], :]*self.scale + self.trans
            error = (((query_source_points-closest_target_points)**2).sum() +\
            ((query_target_points-closest_source_points)**2).sum())/(query_source_points.shape[0] + query_target_points.shape[0])
            error = error**0.5
            # print(i, "th iter, error: ", error)
    
    def get_trans_scale(self):
        all_scale = self.scale_target * self.scale / self.scale_source 
        all_trans = self.trans + self.offset_target * self.scale - self.offset_source * self.scale_target * self.scale / self.scale_source
        return all_trans, all_scale

    def export_source_mesh(self, output_name):
        self.mesh_source.vertices = (self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        self.mesh_source.export(output_name)

def transform_points(points, matrix, translate=True):
    """
    Returns points rotated by a homogeneous
    transformation matrix.

    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)

    Parameters
    ----------
    points : (n, D) float
      Points where D is 2 or 3
    matrix : (3, 3) or (4, 4) float
      Homogeneous rotation matrix
    translate : bool
      Apply translation from matrix or not

    Returns
    ----------
    transformed : (n, d) float
      Transformed points
    """
    points = np.asanyarray(
        points, dtype=np.float64)
    # no points no cry
    if len(points) == 0:
        return points.copy()

    matrix = np.asanyarray(matrix, dtype=np.float64)
    if (len(points.shape) != 2 or
            (points.shape[1] + 1 != matrix.shape[1])):
        raise ValueError('matrix shape ({}) doesn\'t match points ({})'.format(
            matrix.shape,
            points.shape))

    # check to see if we've been passed an identity matrix
    identity = np.abs(matrix - np.eye(matrix.shape[0])).max()
    if identity < 1e-8:
        return np.ascontiguousarray(points.copy())

    dimension = points.shape[1]
    column = np.zeros(len(points)) + int(bool(translate))
    stacked = np.column_stack((points, column))
    transformed = np.dot(matrix, stacked.T).T[:, :dimension]
    transformed = np.ascontiguousarray(transformed)
    return transformed


def procrustes(a, b, reflection=True, translation=True, scale=True, return_cost=True):
    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError('a and b must contain same number of points!')

    # Remove translation component
    if translation:
        acenter = a.mean(axis=0)
        bcenter = b.mean(axis=0)
    else:
        acenter = np.zeros(a.shape[1])
        bcenter = np.zeros(b.shape[1])

    # Remove scale component
    if scale:
        ascale = np.sqrt(((a - acenter)**2).sum() / len(a))
        bscale = np.sqrt(((b - bcenter)**2).sum() / len(b))
    else:
        ascale = 1
        bscale = 1

    # Use SVD to find optimal orthogonal matrix R
    # constrained to det(R) = 1 if necessary.
    u, s, vh = np.linalg.svd(np.dot(((b - bcenter) / bscale).T, ((a - acenter) / ascale)))
    if reflection:
        R = np.dot(u, vh)
    else:
        R = np.dot(np.dot(u, np.diag([1, 1, np.linalg.det(np.dot(u, vh))])), vh)

    # Compute our 4D transformation matrix encoding
    # a -> (R @ (a - acenter)/ascale) * bscale + bcenter
    #    = (bscale/ascale)R @ a + (bcenter - (bscale/ascale)R @ acenter)
    translation = bcenter - (bscale / ascale) * np.dot(R, acenter)
    matrix = np.hstack((bscale / ascale * R, translation.reshape(-1, 1)))
    matrix = np.vstack((matrix, np.array([0.] * (a.shape[1]) + [1.]).reshape(1, -1)))

    if return_cost:
        transformed = transform_points(a, matrix)
        cost = ((b - transformed)**2).mean()
        return matrix, transformed, cost
    else:
        return matrix


def procrustes_without_rot(a, b):
    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError('a and b must contain same number of points!')

    b_vec = b.reshape((-1))
    dim = b_vec.shape[0]
    A_matrix = np.zeros((b.shape[0]*3, 4))
    A_matrix[0:dim:3, 1] = 1
    A_matrix[1:dim:3, 2] = 1
    A_matrix[2:dim:3, 3] = 1
    A_matrix[:, 0] = a.reshape((-1))
    solution = (np.linalg.inv((A_matrix.T)@A_matrix))@A_matrix.T@b_vec

    matrix = np.zeros((4, 4))
    matrix[:3, :3] = np.identity(3) * solution[0]
    matrix[:3, 3] = solution[1:4]
    matrix[3, 3] = 1

    transformed = transform_points(a, matrix)
    cost = ((b - transformed)**2).mean()
    return matrix, transformed, cost


def icp(a, b, initial=np.identity(4), threshold=1e-5, max_iterations=20, rot=False):
    a = np.asanyarray(a, dtype=np.float64)
    b = np.asanyarray(b, dtype=np.float64)
    atree = KDTree(a)
    btree = KDTree(b)

    # transform a under initial_transformation
    a = transform_points(a, initial)
    b = transform_points(b, initial)
    total_matrix_a = initial
    total_matrix_b = initial

    # start with infinite cost
    old_cost = np.inf

    # avoid looping forever by capping iterations
    for n_iteration in range(max_iterations):
        # Closest point in b to each point in a
        _, idx = btree.query(a, 1)
        closest = b[idx]
        # align a with closest points
        if rot:
            matrix_a, transformed_a, cost_pred = procrustes(a=a, b=closest)
        else:
            matrix_a, transformed_a, cost_pred = procrustes_without_rot(a=a, b=closest)

        # Closest point in a to each point in b
        _, idx = atree.query(b, 1)
        closest = a[idx]
        # align a with closest points
        if rot:
            matrix_b, transformed_b, cost_gt = procrustes(a=b, b=closest)
        else:
            matrix_b, transformed_b, cost_gt = procrustes_without_rot(a=b, b=closest)

        cost = cost_pred + cost_gt
        # update a with our new transformed points
        a = transformed_a
        b = transformed_b
        total_matrix_a = np.dot(matrix_a, total_matrix_a)
        total_matrix_b = np.dot(matrix_b, total_matrix_b)

        if old_cost - cost < threshold:
            break
        else:
            old_cost = cost

    return transformed_a, transformed_b, cost


def compute_trimesh_chamfer(gt_mesh_filename, pred_mesh_filename, optim=False, rot=False):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """
    warnings.filterwarnings("ignore")

    source_mesh = trimesh.load(pred_mesh_filename, process=False)
    target_mesh = trimesh.load(gt_mesh_filename, process=False)

    if optim:
        if rot:
            points_source, _ = trimesh.sample.sample_surface(source_mesh, 30000)
            points_target, _ = trimesh.sample.sample_surface(target_mesh, 30000)
            _, points_source, _ = trimesh.registration.icp(points_source, points_target)
        else:
            icp_solver = ICP_T_S(source_mesh, target_mesh)
            icp_solver.sample_mesh(30000, 'both')
            icp_solver.run_icp_f(max_iter = 100)
            optimized_trans = icp_solver.trans
            optimized_scale = icp_solver.scale

            points_source = icp_solver.points_source * optimized_scale + optimized_trans
            points_target = icp_solver.points_target
    else:
        points_source, _ = trimesh.sample.sample_surface(source_mesh, 30000)
        points_target, _ = trimesh.sample.sample_surface(target_mesh, 30000)

    # change unit from meter to centimeter
    points_source *= 100.
    points_target *= 100.

    # one direction
    gen_points_kd_tree = KDTree(points_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(points_target)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(points_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(points_source)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer
