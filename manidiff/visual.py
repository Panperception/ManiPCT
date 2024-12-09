
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
import trimesh
from pathlib import Path

'''
Custom visualization
'''
def pcwrite(filename, xyz, rgb=None):
    """Save a point cloud to a polygon .ply file.
    """
    if rgb is None:
        rgb = np.ones_like(xyz) * 128
    rgb = rgb.astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

# '''
# Matplotlib Visualization
# '''
#
# def visualize_pointcloud(points, normals=None,
#                          out_file=None, show=False, elev=30, azim=225):
#     r''' Visualizes point cloud data.
#     Args:
#         points (tensor): point data
#         normals (tensor): normal data (if existing)
#         out_file (string): output file
#         show (bool): whether the plot should be shown
#     '''
#     # Create plot
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.scatter(points[:, 2], points[:, 0], points[:, 1])
#     if normals is not None:
#         ax.quiver(
#             points[:, 2], points[:, 0], points[:, 1],
#             normals[:, 2], normals[:, 0], normals[:, 1],
#             length=0.1, color='k'
#         )
#     ax.set_xlabel('Z')
#     ax.set_ylabel('X')
#     ax.set_zlabel('Y')
#     # ax.set_xlim(-0.5, 0.5)
#     # ax.set_ylim(-0.5, 0.5)
#     # ax.set_zlim(-0.5, 0.5)
#     ax.view_init(elev=elev, azim=azim)
#     if out_file is not None:
#         plt.savefig(out_file)
#     if show:
#         plt.show()
#     plt.close(fig)
#
#
# def visualize_pointcloud_batch(path, pointclouds, pred_labels, labels, categories, vis_label=False, target=None,  elev=30, azim=225):
#     batch_size = len(pointclouds)
#     fig = plt.figure(figsize=(20,20))
#
#     ncols = int(np.sqrt(batch_size))
#     nrows = max(1, (batch_size-1) // ncols+1)
#     for idx, pc in enumerate(pointclouds):
#         if vis_label:
#             label = categories[labels[idx].item()]
#             pred = categories[pred_labels[idx]]
#             colour = 'g' if label == pred else 'r'
#         elif target is None:
#
#             colour = 'g'
#         else:
#             colour = target[idx]
#         pc = pc.cpu().numpy()
#         ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
#         ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
#         ax.view_init(elev=elev, azim=azim)
#         ax.axis('off')
#         if vis_label:
#             ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))
#
#     plt.savefig(path)
#     plt.close(fig)
#
#
# '''
# Plot stats
# '''
#
# def plot_stats(output_dir, stats, interval):
#     content = stats.keys()
#     # f = plt.figure(figsize=(20, len(content) * 5))
#     f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
#     for j, (k, v) in enumerate(stats.items()):
#         axs[j].plot(interval, v)
#         axs[j].set_ylabel(k)
#
#     f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
#     plt.close(f)
