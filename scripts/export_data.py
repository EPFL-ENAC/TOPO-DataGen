import os
import cv2
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
from skimage import io


def raw_data_sanity_check(data_dir : str) -> tuple :
    """
    Sanity check for the raw data
    :param data_dir: Path of the poses folder
    :return: (camera pose, image, scene coordinate, semantics label)
    """
    img_ls = sorted(glob(os.path.join(data_dir, '*_img.png')))
    pc_ls = sorted(glob(os.path.join(data_dir, '*_pc.npy')))
    sm_ls = sorted(glob(os.path.join(data_dir, '*_semantics.npy')))
    cam_pose = glob(os.path.join(data_dir, '*_poses.npy'))
    assert len(cam_pose) == 1, "Camera pose .npy is not found or is not unique!"
    cam_pose = np.load(cam_pose[0])

    # data size check
    assert len(pc_ls) == len(img_ls), "Camera pose list length is {:d}, but {:d} images data are found.".format(
        len(cam_pose), len(img_ls))
    assert len(pc_ls) == len(cam_pose), "Camera pose list length is {:d}, but {:d} point clouds data are found.".format(
        len(cam_pose), len(pc_ls))
    print("{:d} data points are found...".format(len(cam_pose)))
    if len(sm_ls) != len(cam_pose):
        print("Warning: {:d} semantics maps are found but the camera pose list length is {:d}!".format(len(sm_ls),
                                                                                                       len(cam_pose)))
        sm_ls = sm_ls + [None] * (len(cam_pose) - len(sm_ls))

    return cam_pose, img_ls, pc_ls, sm_ls




def plot_data(pose, img, pc, sm,file_root_name,path_folder_out):
    """
    Create raster for Scene coordiantes, Semantics map, Euclidean depth, Surface normals, ORB keypoints.
    :param pose: 6D camera pose, XYZ + yaw-pitch-roll
    :param img: image array [H, W, 3]
    :param pc: scene coordinate [H, W, 3]
    :param sm: semantics label [H, W]
    :param file_root_name: path of the root folder
    :param path_folder_out: path of the output folder
    :return: None
    """

    if img.shape[-1] == 4:
        img = img[:, :, :3]


    # init
    pc_raw = pc.copy()
    pc_flat = pc.reshape(-1, 3)
    mask_valid = pc_flat[:, 0] != -1
    pc_shape = pc.shape

    # depth
    depth = np.linalg.norm(pc - pose[None, None, :3], 2, axis=-1)
    depth[pc[:, :, 0] == -1] = -1

    # surafce normal
    coords = pc_flat.copy()
    coords = coords[mask_valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
    pcd.normalize_normals()
    o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, pose[:3])

    normals = pc.reshape(-1, 3).clip(min=0).astype(np.uint8)  # [N, 3]
    normals[mask_valid] = ((np.asarray(pcd.normals) + 1.0) / 2.0 * 255).astype(np.uint8)  # [X, 3]
    normals = normals.reshape(pc_shape)  # [H, W, 3]

    # orb keypoints
    orb = cv2.ORB_create(512)
    keypoint, des = orb.detectAndCompute(img, None)
    img_orb = cv2.drawKeypoints(img, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # normalize point cloud value
    pc = pc_flat[mask_valid]
    pc -= np.mean(pc, axis=0)
    pc = (pc - np.min(pc, axis=0)) / (np.max(pc, axis=0) - np.min(pc, axis=0))

    pc_all = pc_flat.copy()
    pc_all[mask_valid] = pc
    pc_all[np.logical_not(mask_valid)] *= 0
    pc = pc_all.reshape(pc_shape)
    pc = (pc * 255).astype(np.uint8)


    # Create individual images
    list_array_name = [
        [sm,'_semantics.png'],[pc,'_scene_coordiantes.png'],[normals,'_surface_normals.png'],
        [depth,'_euclidean_depth.png'],[img_orb,'_ORB_keypoints.png']
                       ]

    for item in list_array_name :
        file_name = file_root_name.replace('.png', item[1])
        path_file = os.path.join(path_folder_out, file_name)
        matplotlib.image.imsave(path_file, item[0])
        print(f"Save {file_name}")

    
    # Create merged images
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plt.subplots_adjust(hspace=0.5)
    for row in range(2):
        for col in range(3):
            axes[row, col].axis('off')

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Synthetic RGB image", fontsize=20)

    axes[0, 1].imshow(pc)
    axes[0, 1].set_title("Scene coordiantes (XYZ per pixel)\n *Visualization may be vague.", fontsize=20)

    axes[0, 2].imshow(sm)
    axes[0, 2].set_title("Semantics map \n *Visualization may be vague.", fontsize=20)

    axes[1, 0].imshow(depth)
    axes[1, 0].set_title("Euclidean depth \n *Visualization may be vague.", fontsize=20)

    axes[1, 1].imshow(normals)
    axes[1, 1].set_title("Surface normals \n *Visualization may be vague.", fontsize=20)

    axes[1, 2].imshow(img_orb)
    axes[1, 2].set_title("ORB keypoints \n *Visualization may be vague.", fontsize=20)

    axes[0, 0].text(0.1, 0.97,
                    'Raw data generated by our workflow, e.g., RGB, scene coord, semantics and pose (omitted).',
                    color='blue', fontsize=24, transform=plt.gcf().transFigure)

    axes[1, 0].text(0.1, 0.51,
                    'Some additional data created using raw data, e.g., depth, surface normals and 2D keypoints.',
                    color='orange', fontsize=24, transform=plt.gcf().transFigure)


    file_name_merged_picture = file_root_name.replace('.png','_preview.png')
    plt.savefig(os.path.join(path_folder_out, file_name_merged_picture), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Save {file_name_merged_picture}")





def export_data(path_folder_in : str, path_folder_out : str) -> None :
    """
    Wrapper function to create raster for each poses.
    :param path_folder_in: folder that contains the synthetic images
    :param path_folder_out: folder that contains the output images
    :return: none
    """

    cam_pose, img_ls, pc_ls, sm_ls = raw_data_sanity_check(path_folder_in)
    list_png = [ i for i in os.listdir(path_folder_in) if i.endswith('.png')]
    list_png = list(sorted(list_png))

    if not os.path.exists(path_folder_out) :
        os.makedirs(path_folder_out)

    for i, (pose, img, pc, sm) in enumerate(zip(cam_pose, img_ls, pc_ls, sm_ls)):
        img = io.imread(img)
        pc = np.load(pc)
        sm = np.load(sm)
        file_root_name = list_png[i]
        plot_data(pose, img, pc, sm, file_root_name,path_folder_out)



def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_dir", type=str, nargs='+',help="Path of the synthetic folder")
    parser.add_argument("--out_dir", type=str, nargs='+',help="Path of the output rasters")
    opt = parser.parse_args()
    return opt

def main():
    if not args.pose_dir or not args.out_dir :
        raise Exception("Please, provide input and output folder.")
    export_data(args.pose_dir[0], args.out_dir[0])


if __name__ == "__main__":
    args = config_parser()
    main()

