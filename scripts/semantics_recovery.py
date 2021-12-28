import os
import pdb
import time
import torch
import laspy
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, Union
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


def apply_offset(pc: np.ndarray, offset: np.ndarray, scale: float = 1.0, nodata_value: float = None) -> np.ndarray:
    """
    Apply offset and scaling to the point cloud data.
    @param pc:              [N, X] point cloud. (X >= 3, the fourth and later columns are non-coordinates).
    @param offset:          [3] offset vector.
    @param scale:           Float number for scaling.
    @param nodata_value:    Float number for nodata value.
    return                  Point cloud w/ offset and scaling.
    """
    pc = pc.copy()
    if nodata_value is None:
        pc[:, :3] -= offset.reshape(1, 3)
        pc[:, :3] /= scale
    else:
        nodata_value = float(nodata_value)
        pc[pc[:, 0] != nodata_value, :3] -= offset.reshape(1, 3)
        pc[pc[:, 0] != nodata_value, :3] /= scale
    return pc


def pc_local_search(big_pc: torch.Tensor, ref_patch: torch.Tensor, nodata_value: float = -1.0) -> torch.Tensor:
    """
    Point cloud local search based on XYZ boundaries.
    @param big_pc:          [N, D] point cloud. (D >= 3, the fourth and later columns are non-coordinates).
    @param ref_patch:       [K, 3] point cloud as reference.
    @param nodata_value:    Float number for nodata value.
    return                  A subset of pertinent point clouds.
    """
    if ref_patch.numel() == torch.sum(ref_patch == nodata_value).item():
        selected_pc = torch.empty(0)
    else:
        xyz_max, _ = ref_patch[ref_patch[:, 1] != nodata_value].max(dim=0)  # [3]
        xyz_min, _ = ref_patch[ref_patch[:, 1] != nodata_value].min(dim=0)  # [3]

        flag = torch.logical_and(big_pc[:, :3] <= xyz_max, big_pc[:, :3] >= xyz_min).sum(dim=1) == 3
        if torch.sum(flag).item() == 0:
            margin = max(20.0, np.sqrt(len(ref_patch) / 5.0))
            flag = torch.logical_and(big_pc[:, :3] <= xyz_max + margin,
                                     big_pc[:, :3] >= xyz_min - margin).sum(dim=1) == 3
        selected_pc = big_pc[flag]

        if len(selected_pc) == 0:
            selected_pc = torch.empty(0)

    return selected_pc


def split_scene_coord(sc: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    """
    Split the scene coordinate associated with image pixels.
    @param sc:          [H, W, 3] scene coordinates.
    @param block_h:     Block size in height direction.
    @param block_w:     Block size in width direction.
    return              an array of block-wise coordinates, [rows, cols, block_h, block_w, 3].
    """
    h, w = sc.shape[:2]

    assert h // block_h == h / block_h
    assert w // block_w == w / block_w

    sc_split_h_ls = np.vsplit(sc, np.arange(h)[::block_h][1:])  # vertical split in height direction

    sc_split = [[] for _ in range(len(sc_split_h_ls))]
    for row, sc_split_h in enumerate(sc_split_h_ls):
        sc_split_w = np.hsplit(sc_split_h, np.arange(w)[::block_w][1:])  # horizontal split in width direction
        sc_split[row] = sc_split_w

    return np.array(sc_split)


def convert_to_tensor(data: np.ndarray, cuda=False, retain_tensor=False, float16=False) \
        -> Tuple[bool, Union[None, torch.Tensor]]:
    """
    Try making tensor from numpy array.
    """
    if float16:
        data_tensor = torch.tensor(data).bfloat16()
    else:
        data_tensor = torch.tensor(data).float()
    flag_ok = torch.isnan(data_tensor).sum() == 0 and torch.isinf(data_tensor).sum() == 0
    data_tensor = data_tensor if retain_tensor else torch.zeros(1)

    if flag_ok:
        data_tensor = data_tensor.cuda() if cuda else data_tensor
        return True, data_tensor
    else:
        del data_tensor
        return False, None


def sc_query(sc: torch.Tensor, pc: torch.Tensor, nodata_value: float = -1.0) -> np.ndarray:
    """
    Query the scene coords' semantic labels in the given point cloud.
    @param sc:              [H, W, 3] scene coordinates.
    @param pc:              [N, 4] point cloud w/ semantic labels.
    @param nodata_value:    Float number for nodata value.
    @return                 [H, W] semantic label.
    """
    h, w = sc.shape[:2]

    pc = pc.clone()

    sc = sc.reshape(-1, 3)  # [K, 3]
    mask_nodata = sc[:, 0] == nodata_value

    sc_cdist = sc[torch.logical_not(mask_nodata)]  # [K', 3]
    pc_cdist = pc[:, :3]  # [N, 3]

    # torch cdist for distance computation, only p=2 is supported as of pytorch 1.9!
    # See issue: https://github.com/pytorch/pytorch/issues/49928
    qeury2pc_dist = torch.cdist(sc_cdist, pc_cdist, p=2.0)  # [K', N]

    # matrix multiplication, too much GPU memory, don't use.
    # qeury2pc_dist = torch.mm(sc_cdist, pc_cdist.transpose(1, 0))  # [K', N]

    # l1 distance, too much GPU memory, don't use.
    # qeury2pc_dist = (sc_cdist[:, None, :] - pc_cdist[None, :, :]).abs().sum(dim=-1)  # [K', N]

    cloest_dist, idx_cloest_pt = qeury2pc_dist.min(dim=1)  # [K'] + [K']

    semantics_label = -torch.ones(h * w, 2).to(sc.device)  # [H * W]
    semantics_label[torch.logical_not(mask_nodata), 0] = pc[idx_cloest_pt, -1].float()
    semantics_label[torch.logical_not(mask_nodata), 1] = cloest_dist.float()

    semantics_label = semantics_label.reshape(h, w, 2).cpu().numpy()

    return semantics_label


def check_mem(sc_cdist_len: int, pc_cdist_len: int, secure_mem: bool) -> bool:
    """
    check whether the cdist operation will out of memory
    :param sc_cdist_len: number of pixels in the split image patch
    :param pc_cdist_len: number of point in the query scene
    :param secure_mem:   flag to use a more conservative and safer GPU memory checking policy
    :return: bool
    """

    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    max_mem_gb = info.free / 1024 ** 3
    max_mem_gb = max_mem_gb - 1.5 if secure_mem else max_mem_gb - 0.75
    if secure_mem:
        flag_memory_okay = ((sc_cdist_len * pc_cdist_len) / 1e9) <= (max_mem_gb * 2.5 / 15)
    else:
        flag_memory_okay = ((sc_cdist_len * pc_cdist_len) / 1e9) <= (max_mem_gb * 3.25 / 15)
    return flag_memory_okay


def find_opt_split(_sc: np.ndarray, _big_pc_label_tensor: torch.Tensor,
                   block_h: int, block_w: int, secure_mem: bool = False, float16: bool = False) -> (int, int):
    """
    find the optimal strategy to split the input image while fully utilise the GPU
    :param block_w:               default split block width
    :param block_h:               default split block height
    :param _sc:                   input image [480, 720, 3]
    :param _big_pc_label_tensor:  entire point cloud w/ label
    :param secure_mem:            flag to use a more conservative and safer GPU memory checking policy
    :param float16:               flag to use float16 accuracy
    :return block_h, block_w:     optimal split of the image in [block_h, block_w]
    """

    sc_cdist_len, pc_cdist_len = block_h * block_w, _big_pc_label_tensor.shape[0]
    pattern_idx = 0
    optional_list = [(240, 180), (120, 180), (120, 90), (60, 90), (60, 72), (60, 45), (48, 45),
                     (48, 36), (24, 24), (24, 12), (12, 12), (6, 6), (1, 1)]
    selected_pc_ls = []
    _sc_split = None
    while not check_mem(sc_cdist_len, pc_cdist_len, secure_mem):
        block_h, block_w = optional_list[pattern_idx]
        sc_cdist_len = block_h * block_w

        _sc_split = split_scene_coord(_sc, block_h, block_w)  # [rows, cols, b_h, b_w, 3]

        flag_tensor, _sc_split = convert_to_tensor(_sc_split, cuda=True, retain_tensor=True, float16=float16)
        assert flag_tensor
        # selected_pc_len_max
        pc_cdist_len = 0
        selected_pc_ls = []
        for row in range(_sc_split.shape[0]):
            selected_pc_row_ls = []
            for col in range(_sc_split.shape[1]):
                selected_pc = pc_local_search(_big_pc_label_tensor, _sc_split[row, col].reshape(-1, 3),
                                              nodata_value=-1)  # [X, 4]
                pc_cdist_len = max(pc_cdist_len, selected_pc.shape[0])
                selected_pc_row_ls.append(selected_pc)
            selected_pc_ls.append(selected_pc_row_ls)
        pattern_idx += 1

    # release the GPU memory
    torch.cuda.empty_cache()

    return block_h, block_w, _sc_split, selected_pc_ls


def all_path(dirname: str, filter_list: list) -> [list, list]:
    """
    extract all the path of .npy file from the main dir
    :param filter_list:          file format to extract
    :param dirname:         main dir name
    :return:                file name list with detailed path
    """

    file_path_list = []
    folder_path_list = []

    for main_dir, subdir, file_name_list in os.walk(dirname):

        # # current main dir
        # print('main dir:', maindir)
        # # current sub dir
        # print('sub dir:', subdir)
        # # all file under current main dir
        # print('file name list:', file_name_list)

        for this_dir in subdir:
            folder_path_list.append(main_dir + '/' + this_dir)

        for filename in file_name_list:
            if 'poses' in os.path.splitext(filename)[0].split('_'):
                continue

            if os.path.splitext(filename)[1] in filter_list:
                path_detail = os.path.join(main_dir, '_'.join(filename.split('_')[:-1]))
                file_path_list.append(path_detail)

    file_path_list = np.unique(file_path_list).tolist()
    file_path_list.sort()
    folder_path_list = np.unique(folder_path_list).tolist()

    return file_path_list, folder_path_list


def mkdir(output_path: str, folder_ls: list) -> None:
    """
    create folder as the structure of input path
    :param output_path:
    :param folder_ls:
    :return:
    """
    os.makedirs(output_path, exist_ok=True)
    for folder in folder_ls:
        output_folder = os.path.exists(os.path.join(output_path, folder))
        if not output_folder:
            os.makedirs(os.path.join(output_path, folder))


def remove_extreme_points(sc: np.ndarray, threshold: float, nodata_value: float = -1.0):
    """
    Pick the extremely remote points w.r.t. the median center.
    @param sc:              [H, W, 3] scene coordinate.
    @param threshold:       Threshold for the extremely remote points.
    @param nodata_value:    Float number for nodata value.
    return: masked sc, number of outlier points
    """
    sc_shape = sc.shape
    sc = sc.reshape(-1, 3)  # [H*W, 3]
    mask_has_data = sc[:, 0] != nodata_value  # [X]
    sc_valid = sc[mask_has_data]  # [X, 3]
    center_median = np.median(sc_valid, axis=0)  # [3]

    # make sure the thresholding is robust, we purge at most 1                          % of the scene coordinates
    dist_to_center = np.linalg.norm(sc_valid - center_median, axis=1)  # [X]
    dist_max_quantile = np.quantile(dist_to_center, 0.99, axis=0, interpolation='nearest')  # scalar
    threshold_robust = np.max([dist_max_quantile, threshold])

    # reset the possible outliers
    mask_outlier = dist_to_center > threshold_robust  # [X]
    sc_valid[mask_outlier] = nodata_value
    sc[mask_has_data] = sc_valid  # [H*W, 3]

    print("Actual threshold: {:.1f} m, number of points to remove: {:d} ({:.2f}%)".format(
        threshold_robust, np.sum(mask_outlier), np.sum(mask_outlier) / len(sc) * 100.0))

    return sc.reshape(sc_shape)


def main():
    args = config_parser()
    print(args)
    downsample_rate = args.downsample_rate
    start_idx = args.start_idx
    end_idx = args.end_idx

    # set the input and output path
    las_file_path = os.path.abspath(args.las_path)
    input_path = os.path.abspath(args.input_path)
    output_path_semantics = os.path.abspath(args.output_path_semantics)
    output_path_distance = os.path.abspath(args.output_path_distance)
    file_ls, folder_ls = all_path(input_path, filter_list=['.npy'])

    # load raw las and turn into 3D point array
    _big_pc_label = []

    las_ls = [las for las in os.listdir(las_file_path) if las.endswith('.las')]
    for idx, las_name in enumerate(las_ls):
        las = laspy.read(os.path.join(las_file_path, las_name))
        las = np.stack([las.x, las.y, las.z, np.array(las.classification)], axis=1)
        _big_pc_label.extend(las)

    _big_pc_label = np.array(_big_pc_label) # [N, 4]
    _big_pc_label = np.ascontiguousarray(_big_pc_label)  # [N, 4]

    # read point cloud with semantic label data from .npy file
    bound_xyz_min = _big_pc_label[:, :3].min(axis=0)  # [3]
    bound_xyz_max = _big_pc_label[:, :3].max(axis=0)  # [3]
    offset_center = (bound_xyz_max + bound_xyz_min) / 2  # [3]
    interval_xyz = bound_xyz_max - bound_xyz_min  # [3]
    if args.float16:
        scale = np.array(interval_xyz / 1.e5, np.float64)  # [3]
    else:
        scale = 1.0
    # print('Offset origin XYZ: {}, {}, {}, scale: {}'.format(*offset_center, scale))
    _big_pc_label = apply_offset(_big_pc_label, offset_center, scale, nodata_value=None)  # [X, 4]

    flag_tensor, _big_pc_label_tensor = convert_to_tensor(_big_pc_label, cuda=True, retain_tensor=True,
                                                          float16=args.float16)
    assert flag_tensor, "Cannot build tensor for the original data (w/ offset)!"

    # create output folder structure
    input_path_len = len(input_path.split('/'))
    folder_ls = ['/'.join(folder.split('/')[input_path_len:]) for folder in folder_ls]
    folder_ls = np.unique(folder_ls).tolist()
    mkdir(output_path_semantics, folder_ls)
    mkdir(output_path_distance, folder_ls)

    # print(file_ls)

    for idx_dp, file_name in tqdm(enumerate(file_ls[start_idx:end_idx])):

        time_start = time.time()
        block_h, block_w = args.block_h, args.block_w
        """Load ray-traced point cloud"""
        _sc = np.load('{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
        _sc = _sc[::downsample_rate, ::downsample_rate, :]  # [H, W, 3]
        _sc_raw_size = _sc.shape  # [H, W, 3]
        _sc = apply_offset(_sc.reshape(-1, 3), offset_center, scale, nodata_value=-1).reshape(
            _sc_raw_size)  # [H, W, 3], numpy array

        if not args.force_all_pixel:
            _sc = remove_extreme_points(_sc, args.outlier_dist, nodata_value=-1.0)

        block_h, block_w, _sc_split, selected_pc_ls = find_opt_split(_sc, _big_pc_label_tensor, block_h, block_w,
                                                                     args.secure_mem, args.float16)

        """Recover the semantic labels (divide and conquer)"""
        semantics_label_ls = [[[] for _ in range(_sc_split.shape[1])] for _ in range(_sc_split.shape[0])]
        semantics_distance_ls = [[[] for _ in range(_sc_split.shape[1])] for _ in range(_sc_split.shape[0])]
        ttl_time_search, ttl_time_query = 0.0, 0.0
        for row in range(_sc_split.shape[0]):
            for col in range(_sc_split.shape[1]):
                time_search = time.time()
                selected_pc = selected_pc_ls[row][col]
                ttl_time_search += time.time() - time_search

                time_query = time.time()
                if args.secure_mem:
                    torch.cuda.empty_cache()
                if len(selected_pc):
                    semantic_label = sc_query(_sc_split[row, col], selected_pc, nodata_value=-1.0)
                else:
                    semantic_label = -np.ones_like(_sc_split[row, col].cpu().float().numpy())[:, :, :2]

                ttl_time_query += time.time() - time_query

                semantics_label_ls[row][col] = semantic_label[:, :, 0]
                semantics_distance_ls[row][col] = semantic_label[:, :, 1] * scale
                del selected_pc

        semantics_label = np.block(semantics_label_ls)  # [H, W]
        semantics_label[semantics_label == -1] = 0
        semantics_label = np.array(semantics_label, np.uint8)
        semantics_distance = np.block(semantics_distance_ls).astype(np.float32)  # [H, W]
        tiem_elapsed = time.time() - time_start
        print("Semantics label recovery time: {:.1f}s, unique labels: {}".format(tiem_elapsed,
                                                                                 np.unique(semantics_label)))
        torch.cuda.empty_cache()

        """Results saving"""
        np.save(os.path.join(output_path_semantics, '{:s}_semantics'
                             .format('/'.join(file_name.split('/')[input_path_len:]))), semantics_label)
        np.save(os.path.join(output_path_distance, '{:s}_distance'
                             .format('/'.join(file_name.split('/')[input_path_len:]))), semantics_distance)


def config_parser():
    parser = argparse.ArgumentParser(
        description='Semantic label recovery script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', type=str, default=None, required=True,
                        help='Directory where the raw data is.')

    parser.add_argument('--las_path', type=str, default=None, required=True,
                        help='Directory where the las files are.')

    parser.add_argument('--output_path_semantics', type=str, default=None,
                        help='Directory to store semantic label output. Default is the raw data path.')

    parser.add_argument('--output_path_distance', type=str, default=None, required=True,
                        help='Directory to store distance output.')

    parser.add_argument('--start_idx', type=int, default=0,
                        help='start index of the file list')

    parser.add_argument('--end_idx', type=int, default=None,
                        help='start index of the file list')

    parser.add_argument('--float16', action='store_true',
                        help='Use float16 number to accelerate computation (unstable!)')

    parser.add_argument('--downsample_rate', type=int, default=1,
                        help='Downsampling rate.')

    parser.add_argument('--block_h', type=int, default=240,
                        help='Cell block height.')

    parser.add_argument('--block_w', type=int, default=360,
                        help='Cell block width.')

    parser.add_argument('--secure_mem', action='store_true',
                        help='Use a more conservative strategy when estimating the GPU memory need.')

    parser.add_argument('--force_all_pixel', action='store_true',
                        help='Apply label recovery to all pixels forcefully, otherwise we use the median'
                             'center to robustly skip some hard outliers.')

    parser.add_argument('--outlier_dist', default=5000, type=float,
                        help='Distance threshold for outlier point w.r.t. median center, unit in meter.')

    opt = parser.parse_args()

    if opt.float16:
        print("Warning: float16 mode is highly unstable!")
    if opt.output_path_semantics is None:
        opt.output_path_semantics = opt.input_path
        print("Semantics label output path is not specified! The raw data path is used, i.e., the semantic labels are "
              "generated in place.")
    return opt


if __name__ == '__main__':
    main()
