import torch
import numpy as np
from torch.nn import ReLU

def rotation_vectors(a,b):
    """
    input:
    --------------------------------
    a: (3,)
    b: (3,)

    output:
    --------------------------------
    T: (3,3)

    Return the rotation matrix that rotates vector a to vector b

    """
    #orthogonal vector
    #Note this will equal to 0 if a and b are parallel or anti-parallel
    cross = torch.cross(a, b)


    sin = (cross.T/torch.linalg.norm(cross)).T
    cos = torch.dot(a, b)
    skew_symmetric_constructor = torch.tensor(np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])).float()
    I = torch.eye(3)
    cross_i = torch.matmul(cross, skew_symmetric_constructor).reshape([3,3])
    cross_i_2 = torch.matmul(cross_i, cross_i)
    T = I + cross_i + cross_i_2 * (1 / (1 + cos))
    return T

def multi_rotation_vectors_batch(a,b):
    """
    input:
    --------------------------------
    a: B*N*3
    b: B*N*3

    output:
    --------------------------------
    T: B*N*3*3

    Return the rotation matrix that rotates multiple vectors a to multiple vectors b

    """
    B, N, _ = a.shape
    cross = torch.cross(a, b, dim=2)
    sin = cross / torch.linalg.norm(cross, dim=2, keepdim=True)
    cos = torch.sum(a * b, dim=2, keepdim=True)
    skew_symmetric_constructor = torch.tensor(np.array([[0,0,0,0,0,-1,0,1,0], [0,0,1,0,0,0,-1,0,0], [0,-1,0,1,0,0,0,0,0]])).float().to(a.device)
    xi_cross = torch.matmul(cross, skew_symmetric_constructor).reshape([B,N,3,3])
    I = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1).to(a.device)
    xi_cross_2 = torch.matmul(xi_cross, xi_cross)
    T = I + xi_cross + xi_cross_2 * (1 / (1 + cos).unsqueeze(-1))
    return T



def linearized_cone(normal, w):
    """
    input:
    --------------------------------
    normal: B x N x 3
    w: B x N x 4

    output:
    --------------------------------
    f: B x N x 3

    Return the weights of the 4-edge friction pyramid
    """
    B = normal.shape[0]
    N = normal.shape[1]
    z = torch.tensor([0., 0., 1.]).view(1, 1, 3).repeat([B, N, 1])
    # construct 4-edge friction pyramid 
    mu = 0.1
    e1 = torch.tensor([mu, 0, 1]).view(1, 1, 3).repeat([B, N, 1])
    e2 = torch.tensor([0, mu, 1]).view(1, 1, 3).repeat([B, N, 1])
    e3 = torch.tensor([mu, 0, 1]).view(1, 1, 3).repeat([B, N, 1])
    e4 = torch.tensor([0, mu, 1]).view(1, 1, 3).repeat([B, N, 1])

    # f = w_0e_0 + , w > 0
    # fe^-1 > 0
    E = torch.stack([e1, e2, e3, e4], -2) # B x N x 4 x 3
    T = multi_rotation_vectors_batch(z, normal) # B x N x 3 x 3

    #apple T to E
    E_transed = torch.matmul(T, E.transpose(-2,-1)).transpose(-2,-1) # B x N x 4 x 3

    #E_pinv = torch.linalg.pinv(E_transed) # B x N x 4 x 3

    #print(E_pinv.shape)
    print(E_transed.shape)
    print(w.shape)
    f = torch.matmul(w.unsqueeze(-2), E_transed).squeeze()
    w_edges = (E_transed * w.unsqueeze(-1)).reshape(B, N*4, 3)

    
    

    return f, w_edges

if __name__ == "__main__":
    B = 1
    N = 1
    z = torch.tensor([0., 0., 1.]).view(1, 1, 3).repeat([B, N, 1])
    a = torch.rand(B, N, 3)
    print(multi_rotation_vectors_batch(z, a))
    print(rotation_vectors(z.squeeze(), a.squeeze()))


    # b = torch.rand(B, N, 4)

    # w ,e= linearized_cone(a, b)

    # print(w.shape)
    # print(e.shape)