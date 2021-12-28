import numpy as np
import imageio
import argparse
import glob
import os

np.seterr(invalid='ignore')


def make_grayscale(depth: np.ndarray) -> np.ndarray:
    """Get depth for grayscale images."""
    depth_not_nan = depth[np.logical_not(np.isnan(depth))]
    min_v = np.min(depth_not_nan)
    max_v = np.max(depth_not_nan)
    r = (255 * (depth - min_v) / (max_v - min_v))
    # normalize r
    r[np.isnan(r)] = 255
    return r.astype(np.uint8)


def gaussian(x, mu, sig):
    """Simple gaussian function."""
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / np.sqrt(2 * np.pi)


def normalize(in_mat: np.ndarray) -> np.ndarray:
    """Simple normalize function."""
    in_mat = in_mat - np.min(in_mat[np.logical_not(np.isnan(in_mat))])
    return in_mat / np.max(in_mat[np.logical_not(np.isnan(in_mat))])


def colorize(depth: np.ndarray) -> np.ndarray:
    """Colorize the picture."""
    depth = normalize(depth)
    r = gaussian(depth, 0, 0.2) * 255
    g = gaussian(depth, 0.4, 0.2) * 255
    b = gaussian(depth, 0.6, 0.2) * 255
    rgb = np.stack([r, g, b], 2)
    rgb[np.isnan(rgb)] = 255
    return rgb.astype(np.uint8)


def compute_depth(cloud: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Compute Euclidean depth."""
    xyz = pose[0:3]
    depth = np.sqrt(np.sum(np.square(cloud - xyz), 2))
    depth[cloud[:, :, 0] < 0] = np.nan
    return depth


def compute_depth_images(pc_names: list, poses: np.ndarray, path: str, colorized=False):
    """Get depth images and save to disk."""
    for j, pc_name in enumerate(pc_names):
        cloud = np.load(os.path.join(path, pc_name))
        depth = compute_depth(cloud, poses[j])
        if colorized == 'c':
            depth = colorize(depth)
            imageio.imwrite(os.path.join(path, pc_name[:-4] + '_depth_color.png'), depth)
        elif colorized == 'g':
            depth = make_grayscale(depth)
            imageio.imwrite(os.path.join(path, pc_name[:-4] + '_depth_grayscale.png'), depth)
        else:
            np.save(os.path.join(path, pc_name[:-4] + '_depth.npy'), depth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path of the dataset for which to generate depth images")
    parser.add_argument("-min", type=int, help="Min index")
    parser.add_argument("-max", type=int, help="Max index")
    parser.add_argument("-i", type=int, nargs='+', help="List indexes (separated by ' ')")
    parser.add_argument("-c", action="store_true", default=False,
                        help="Use this to get a colorized png depth visualisation instead of depth tif")
    parser.add_argument("-g", action="store_true", default=False,
                        help="Use this to get a grayscale png depth visualisation instead of depth tif")

    args = parser.parse_args()
    color = 'c' if args.c else 'g' if args.g else False

    path = os.path.abspath(args.path)
    name = os.path.basename(os.path.abspath(path))
    if '_saved' in name:
        name = name[:name.find('_saved')]
    poses = np.load(glob.glob(path + '/*poses.npy')[0])
    min_ = args.min if args.min else 0
    max_ = args.max if args.max else len(poses) - 1
    indexes = np.array(args.i) if args.i else np.arange(min_, max_ + 1)
    poses = poses[indexes, :]
    pc = []
    for j, ind in enumerate(indexes):
        try:
            pc.append(os.path.basename(glob.glob('{0}/{1}_{2:05d}_*pc.npy'.format(path, name, ind))[0]))
        except Exception:
            poses[j, 0] = np.nan
    poses = poses[np.logical_not(np.isnan(poses[:, 0])), :]
    compute_depth_images(pc, poses, path, color)


if __name__ == "__main__":
    main()
