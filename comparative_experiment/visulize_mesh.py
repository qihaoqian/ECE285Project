import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# 设置使用EGL后端
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def get_camera_pose_euler(euler_angles_deg, distance=2.5):
    """
    euler_angles_deg: (x, y, z) in degrees
    """
    rot = R.from_euler('xyz', euler_angles_deg, degrees=True)
    forward = np.array([0, 0, -1])  # Camera looks along -Z
    up = np.array([0, 1, 0])        # Up is +Y
    eye = np.array([0, 0, distance])
    
    rot_mat = rot.as_matrix()
    eye = rot_mat @ eye
    forward = rot_mat @ forward
    up = rot_mat @ up

    target = np.array([0, 0, 0])
    z = (target - eye)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = -z
    pose[:3, 3] = eye
    return pose

def render_and_save(mesh_path, camera_pose, save_path, resolution=512):
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    
    # 确保网格不是平滑的
    mesh.visual.face_colors = None
    mesh.visual.vertex_colors = None

    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_node)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    try:
        r = pyrender.OffscreenRenderer(viewport_width=resolution, viewport_height=resolution)
        color, _ = r.render(scene)
        r.delete()
        plt.imsave(save_path, color)
        print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error during rendering: {str(e)}")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh1', required=True)
    parser.add_argument('--mesh2', required=True)
    parser.add_argument('--out1', required=True)
    parser.add_argument('--out2', required=True)
    parser.add_argument('--euler', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                        default=[45, 45, 0], help='Euler angles (degrees): x y z')
    args = parser.parse_args()

    pose = get_camera_pose_euler(args.euler)
    render_and_save(args.mesh1, pose, args.out1)
    render_and_save(args.mesh2, pose, args.out2)
