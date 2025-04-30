from dexycb import dexycb
from CtcOpt.CtcObj import object_sdf
from Robot import human
import torch
from plotly import graph_objects as go
from test import mesh_plot, plot_3d_quiver
def mesh_plot(mesh, points):
    vertices = points

    # Create a 3D scatter plot for the vertices
    scatter = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(
            size=8,  # Adjust marker size here
            color='blue'  # Adjust marker color here
        )
    )

    # Initialize a list to store the text annotations
    annotations = []

    # Add an annotation for each vertex
    for i, vertex in enumerate(vertices):
        annotations.append(
            dict(
                showarrow=False,
                x=vertex[0], y=vertex[1], z=vertex[2],
                text=str(i),
                xanchor="left",
                xshift=8,
                opacity=1,
                font = dict(
                size=10,  # Specify the font size here
                color="black"  # You can also change the font color
            )
            )
        )

    mesh = go.Mesh3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color='lightblue', opacity=0.9)

    # Create the figure and add the scatter plot
    fig = go.Figure(data=[scatter,mesh])

    # Add annotations to the figure
    # fig.update_layout(scene=dict(
    #     annotations=annotations
        
    # ))
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
    # Show the figure
    fig.show()
if __name__ == "__main__":
    dex_train = dexycb('s1', 'train')
    obj, hand = dex_train.process(dex_train.getdata[9000])
    human = human(hand_mesh=hand)
    obj_model = object_sdf(obj, hand, "cuda")

    test_point = torch.randn((1,4,3),dtype=torch.float32,requires_grad=True, device="cuda")
    #test_point = obj_model.hand_object_contact().float().unsqueeze(0)
    w = torch.ones((1,4,4),dtype=torch.float32,requires_grad=True, device="cuda")

    def fc_loss(points, w):
        sdf,gf,gg,intfc,e_dist = obj_model.loss_fc(points,w)
        #sdf,gf,gg,intfc,e_dist,r = obj_model.grasp_quality(points,w)

        return sdf + gf + gg + intfc + e_dist, gf, sdf


    def grasp_quality(points, w):
        sdf,gf,gg,intfc,e_dist,r = obj_model.grasp_quality(points,w)

        return sdf + gf + gg + intfc + e_dist + r, r


    def sdf_loss_(points):
        sdf = obj_model.sdf_loss(points)

        return sdf


    # we = obj_model.get_weighted_edges(test_point, w)
    # we = we.squeeze(0)
    # print(we.size(),test_point.size())
    # x = test_point.squeeze(0).unsqueeze(1).repeat(1,we.size(1),1)
    # x_to_plot = x.reshape(20,3).cpu().detach().numpy()
    # we_to_plot = we.reshape(20,3).cpu().detach().numpy()
    # short_we = x_to_plot+(we_to_plot-x_to_plot)*0.01
    # plot_3d_quiver(x_to_plot,short_we)
    opt = torch.optim.Adam([test_point,w], lr=0.01)

    print(obj_model.grasp_quality(test_point, w))
    #while not torch.allclose(fc_loss(test_point, w)[1],torch.tensor(0, dtype=torch.float), atol=1e-3) and not torch.allclose(fc_loss(test_point, w)[2],torch.tensor(0, dtype=torch.float), atol=1e-6):
    condition = False
    while not condition:
        condition = torch.allclose(fc_loss(test_point, w)[2],torch.tensor(0, dtype=torch.float), atol=1e-5)
        opt.zero_grad()
        loss = fc_loss(test_point, w)[0]
        loss.backward()
        opt.step()
        print(loss)

    for _ in range(1000):
        opt.zero_grad()
        loss = grasp_quality(test_point, w)[0]
        loss.backward()
        opt.step()
        print(loss)
    


    mesh_plot(obj, test_point.detach().squeeze(0).cpu().numpy())
    #print("points",test_point, "w", w)
    print(obj_model.grasp_quality(test_point, w)[-1])

