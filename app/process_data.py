import pdb
import subprocess
import argparse
import os
import json
import shutil
import pyDOE
import numpy as np
import time
import pyproj
import glob
import sys
import signal
import rasterio
from tqdm import tqdm
from typing import Union, Tuple
import multiprocessing

DATA_FOLDER_PATH = "/home/regislongchamp/Desktop/temp_vnav"
USER_HOME = os.path.expanduser("~")
PROJ_DIR = os.path.abspath(os.path.join(__file__, '../..'))
CESIUM_HOME = "$HOME/Documents/Cesium"
sys.path.insert(0, PROJ_DIR)
from scripts.reframeTransform import ReframeTransform


number_of_cpu = multiprocessing.cpu_count()


def read_preset(preset_path: str) -> tuple:
    """
    Read preset configuration json file.
    """
    assert preset_path.split('.')[-1] == 'json', "Preset at {:s} is not a json file!".format(preset_path)
    with open(preset_path, 'r') as f:
        data = json.load(f)
        height_ref = None if 'heightRef' not in data else data['heightRef']
        if height_ref is not None:
            height_ref = os.path.abspath(height_ref.replace('PROJ_DIR', PROJ_DIR))
        return (
            data['nbImages'], data['lonRange'], data['latRange'], data['hRange'], data['yawRange'],
            data['pitchRange'],
            data['rollRange'], height_ref)


def geographic_to_ecef(lon: Union[float, np.ndarray], lat: Union[float, np.ndarray], alt: Union[float, np.ndarray]) \
        -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Transform the wgs84 geographic to wgs84 ECEF coordinate system.
    Args:
        lon: Longitude
        lat: Latitude
        alt: Altitude
    Returns:
        (X, Y, Z) in ECEF coordinate system
    """
    # Careful: here we need to use lat, lon
    return pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(lat, lon, alt)


def prepare_lhs(num_dp: int, lon_range: list, lat_range: list, h_range: list,
                yaw_range: list, pitch_range: list, roll_range: list, height_ref: Union[bool, str],
                path_file_numpy_poses: str, data_type: str = "geocentric") -> None:
    """
    Generate npy file for LHS sampling. The output includes the information of camera poses.
    Args:
        num_dp:         The number of samples.
        lon_range:      Longitude range.
        lat_range:      Latitude range.
        h_range:        Height range.
        yaw_range:      Yaw angle range.
        pitch_range:    Pitch angle range.
        roll_range:     Roll angle range.
        height_ref:     Height_ref for relative height.
        path_file_numpy_poses:      The file path of output .npy
        data_type:      Type of range with defaults being "geocentric".
    """

    lhd = pyDOE.lhs(6, samples=num_dp)
    # Careful: in _poses.npy we use lat, lon. In presets txt lon comes first
    ranges = np.array(
        [lon_range[1] - lon_range[0], lat_range[1] - lat_range[0], h_range[1] - h_range[0], yaw_range[1] - yaw_range[0],
         pitch_range[1] - pitch_range[0], roll_range[1] - roll_range[0]])
    min_value = np.array([lon_range[0], lat_range[0], h_range[0], yaw_range[0], pitch_range[0], roll_range[0]])
    lhd = lhd * ranges + min_value
    if height_ref:
        print("Translating heights using provided DTM. The DTM has to be in LV95 with WGS84 heights")
        r = ReframeTransform()  # Transform wgs84 to lv95 poses
        with rasterio.open(height_ref) as src:
            height = src.read()[0]
            for j, pose in enumerate(tqdm(lhd)):
                lv95pose = r.transform([pose[0], pose[1], 0], 'wgs84', 'wgs84', 'lv95', 'bessel')
                row, col = src.index(lv95pose[0], lv95pose[1])
                lhd[j, 2] += height[row, col]
    if data_type == "geocentric":
        lhd[:, 0], lhd[:, 1], lhd[:, 2] = geographic_to_ecef(lhd[:, 0], lhd[:, 1], lhd[:, 2])
    else:
        raise NotImplementedError
    # Find surface and volume:
    origin = np.array(geographic_to_ecef(lon_range[0], lat_range[0], h_range[0]))
    lx = np.linalg.norm(np.array(geographic_to_ecef(lon_range[1], lat_range[0], h_range[0])) - origin)
    ly = np.linalg.norm(np.array(geographic_to_ecef(lon_range[0], lat_range[1], h_range[0])) - origin)
    lz = np.linalg.norm(np.array(geographic_to_ecef(lon_range[0], lat_range[0], h_range[1])) - origin)
    print("Approximate scene surface: " + str(lx * ly) + " m2")
    print("Approximate scene volume: " + str(lx * ly * lz) + " m3")
    print(f"Save latin hypercube sampling poses here :  {path_file_numpy_poses}")
    np.save(path_file_numpy_poses, lhd)


def read_gps_pos(exif_line: str) -> float:
    """Read GPS position: longitude or latitude."""
    if exif_line[0] == '"':
        exif_line = exif_line[1:]
    if exif_line[-1] == '"':
        exif_line = exif_line[:-1]
    exif_line = exif_line.split(' ')
    if exif_line[0] == '':
        exif_line = exif_line[1:]
    d = int(exif_line[0])
    m = int(exif_line[2][:-1])
    s = float(exif_line[3][:-2])
    sign = 1 if (exif_line[-1] == 'E' or exif_line[-1] == 'N') else -1
    m += s / 60
    d += m / 60
    d *= sign
    return d


def prepare_phantom_drone_matching(data_dir: str, path_file_numpy_poses: str, data_type: str = "geocentric") -> None:
    """
    Extract camera poses for images collected by DJI Phantom 4 camera. Save the output to a .npy file.
    Args:
        data_dir:   Directory to the raw images.
        path_file_numpy_poses:  The file pyth of output .npy file.
        data_type:  Type of positional info. with defaults being "geocentric", i.e., ECEF coordinates.
    """
    assert os.path.isdir(data_dir), "Path {:s} is not a directory!".format(data_dir)
    data_dir = os.path.abspath(data_dir)
    subprocess.run(
        '''exiftool -gpsposition -absolutealtitude -camerayaw -camerapitch -cameraroll {0}/*.JPG -csv > {0}/img_meta.csv'''.format(
            data_dir, data_dir), shell=True)
    poses = []
    pic_names = []
    with open('{0}/img_meta.csv'.format(data_dir), 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            line = line.split(',')
            if len(line) <= 1:
                raise Exception("The meta-data seems wrong! Please check {:s}/img_meta.csv!".format(data_dir))
            pic_names.append(line[0][2:-4].split('/')[-1])
            latitude = read_gps_pos(line[1])
            longitude = read_gps_pos(line[2])
            height = float(line[3])
            yaw = float(line[4])
            pitch = float(line[5])
            roll = float(line[6])
            poses.append([longitude, latitude, height, yaw, pitch, roll])
            line = f.readline()
        poses = np.array(poses)
        if data_type == "geocentric":
            poses[:, 0], poses[:, 1], poses[:, 2] = geographic_to_ecef(poses[:, 0], poses[:, 1], poses[:, 2])

        print(f"Save poses based on drone footings here :  {path_file_numpy_poses}")
        np.save(path_file_numpy_poses, poses)
        folder_path = os.path.dirname(path_file_numpy_poses)
        folder_name = os.path.basename(folder_path)
        with open(os.path.join(folder_path, f"{folder_name}_pic_names.csv"), 'w') as f_pic:
            f_pic.write(','.join(pic_names))


def remove_duplicates(file_list: list) -> list:
    """
    Remove duplicate files and return the list of unique files.
    E.g., file.ext, file(1).ext, file(2).ext --> keep only file(2).ext as file.ext
    """
    duplicate_ls = []
    unique_ls = []
    for file in file_list:
        if file.split('.')[-2][-1] == ')':
            unique_name = '('.join(file.split('(')[:-1]) + '.' + file.split('.')[-1]
            duplicate_ls.append(unique_name)
            unique_ls.append(unique_name)
        else:
            unique_ls.append(file)
    unique_ls = list(set(unique_ls))
    duplicate_ls = list(set(duplicate_ls))
    for duplicate in duplicate_ls:
        [filename, ext] = os.path.splitext(duplicate)
        elems = glob.glob(filename + '(*)' + ext)  # Return a list of all matching file paths
        latest = 0
        pop_idx = 0
        for i, elem in enumerate(elems):
            nb = int(elem.split('(')[-1].split(')')[-2])  # extract the number inside parenthesis (nb)
            if nb > latest:
                latest = nb
                pop_idx = i
        shutil.move(elems.pop(pop_idx), duplicate)  # Keep the latest file
        for elem in elems:
            os.remove(elem)
    return unique_ls


def move_files(dataset_name: str) -> None:
    """
    Trim the generated files at `~/Downloads` directory and move all output to the Cesium-specific folder,
    The utility function remove_duplicates is called here.
    """
    os.makedirs(CESIUM_HOME, exist_ok=True)
    dir_path = os.path.join(CESIUM_HOME, dataset_name)
    os.makedirs(dir_path, exist_ok=True)
    names_file = os.path.join(DATA_FOLDER_PATH, 'presets', dataset_name, dataset_name + '_pic_names.csv')
    if os.path.exists(names_file):
        with open(names_file, 'r') as f:
            pic_names = f.readline().split(',')
    elif "seq" in dataset_name.lower():
        pic_names = "SEQ"
    else:
        pic_names = None
    unique_img_ls = remove_duplicates(glob.glob('{0}/Downloads/{1}_img_*.png'.format(USER_HOME, dataset_name)))
    for img in unique_img_ls:
        idx = int(img.split('_')[-1].split('.')[0])
        cur_name = pic_names[idx] if isinstance(pic_names, list) else pic_names
        cur_name = cur_name + '_' if cur_name else ''
        try:
            shutil.copyfile(img,
                            '{0}/{1}_{2}_{3}img.png'.format(dir_path, dataset_name, '{:05d}'.format(idx), cur_name))
            os.remove(img)  # Note: using copy + remove instead of movefile makes it work for different file systems
        except Exception:
            print("Error while moving " + img)
            pass
    npy_list = remove_duplicates(glob.glob('{0}/Downloads/{1}_coor_*.npy'.format(USER_HOME, dataset_name)))
    for npy in npy_list:
        idx = int(npy.split('_')[-1].split('.')[0])
        cur_name = pic_names[idx] if isinstance(pic_names, list) else pic_names
        cur_name = cur_name + '_' if cur_name else ''
        try:
            shutil.copyfile(npy, '{0}/{1}_{2}_{3}pc.npy'.format(dir_path, dataset_name, '{:05d}'.format(idx), cur_name))
            os.remove(npy)
        except Exception:
            print("Error while moving " + npy)
            pass


def start_server(dataset_name: str, scene_name: str, path_file_numpy_poses : str,
                 start_idx: int = 0, stop_idx: Union[None, int] = None,
                 restart: bool = False, n_proc: int = 1, gpu_mode: str = '0', force_cuda: bool = False) -> None:
    """
    Start local server for Cesium rendering.
    When it's in multi GPU mode, we use different firefox profiles. These have to be created beforehand and named like
    8080,8081,8082 ... (number of profiles == n_proc)
    For each new profile, automatic download of png,npy and txt will have to be set (will be asked by firefox).

    Args:
        dataset_name:   name of dataset.
        scene_name:     name of the underlying scene.
        start_idx:      index of the first datapoint.
        stop_idx:       index of the last datapoint (inclusive).
        restart:        boolean flag for restarting rendering.
        n_proc:         number of processes, i.e., the number of firefox windows.
        gpu_mode:       GPU configuration keyword.
        force_cuda:     to forcefully enable CUDA feature to speed up rendering.
    """

    for f in glob.glob(os.path.join(USER_HOME, 'Downloads', '*finished.txt')):
        os.remove(f)
    subprocess.run('''rm -f {:s}'''.format(os.path.join(USER_HOME, 'Downloads', dataset_name + '*')), shell=True)
    os.makedirs(CESIUM_HOME, exist_ok=True)
    dataset_dir = os.path.join(CESIUM_HOME, dataset_name)

    # Try continuing the data rendering, just make sure the python and javascript scripts are talking about
    # the same poses, otherwise a restart is needed.
    if os.path.exists(dataset_dir) and not restart:
        print("Dataset directory already exists")
        try:
            js_poses = np.load(os.path.join(DATA_FOLDER_PATH, 'presets', dataset_name, dataset_name + '_poses.npy'))
            poses = np.load(path_file_numpy_poses)
            if not np.array_equal(js_poses, poses):
                restart = True
            else:
                print("Successfully continuing from previous state")
        except Exception:
            restart = True

    # If dataset already exists but we need to restart, move the existing one as dataset_saved to not overwrite it.
    if os.path.exists(dataset_dir) and restart:
        print("Saving the existing dataset and generate a brand new one from scratch")
        i = 1
        while True:
            save_path = '{0}/{1}_saved{2}'.format(CESIUM_HOME, dataset_name, i)
            if not os.path.exists(save_path):
                break
            i += 1
        shutil.move(dataset_dir, save_path)
        assert not os.path.exists(dataset_dir), "Dataset directory should have been moved to {:s}".format(save_path)

    if not os.path.exists(dataset_dir):
        # start the generation from scratch if the dataset directory is non-existing
        os.makedirs(dataset_dir)
        shutil.copyfile(os.path.join(path_file_numpy_poses),
                        os.path.join(dataset_dir, dataset_name + '_poses.npy'))
        poses = np.load(os.path.join(dataset_dir, dataset_name + '_poses.npy'))
        indexes = np.arange(0, len(poses))
    else:
        poses = np.load(os.path.join(dataset_dir, dataset_name + '_poses.npy'))
        assert not restart, "Dataset saving directory is existent. Flag of restart must be False!"
        move_files(dataset_name)  # trim Cesium intermediate output files at the Downloads folder
        img_ls = glob.glob('{0}/{1}*_img.png'.format(dataset_dir, dataset_name))
        npy_ls = glob.glob('{0}/{1}*_pc.npy'.format(dataset_dir, dataset_name))
        already_done_img_ls = []
        already_done_npy_ls = []
        for img in img_ls:
            already_done_img_ls.append(int(img.split('/')[-1][len(dataset_name) + 1:].split('_')[0]))
        for npy in npy_ls:
            already_done_npy_ls.append(int(npy.split('/')[-1][len(dataset_name) + 1:].split('_')[0]))
        all_indexes = np.arange(0, len(poses))
        indexes = np.sort([i for i in all_indexes if (i not in already_done_img_ls or i not in already_done_npy_ls)])
    indexes = indexes[indexes >= start_idx]
    indexes = indexes if stop_idx is None else indexes[indexes <= stop_idx]

    # GPUs task allocation
    proc_index = []
    if gpu_mode == 'multi':
        # at most two GPUs are supported
        # it's designed such that n_pic ~= n_proc_first_gpu * n_pic_first_gpu +
        # n_proc_sec_gpu * n_pic_sec_gpu
        gpu_priority = 1  # when odd number of processes, we give more to gpu1 (more powerful)
        load_factor = 1.38  # assign more work to the prioritized GPU
        # number of processes
        n_proc_first_gpu = round(n_proc / 2)
        n_proc_sec_gpu = n_proc - n_proc_first_gpu
        # number of pictures to render
        n_pic_sec_gpu = round(len(indexes) / (n_proc_first_gpu * load_factor + n_proc_sec_gpu))
        n_pic_first_gpu = round(load_factor * n_pic_sec_gpu)
    else:
        n_pic_first_gpu = round(len(indexes) / n_proc)
        n_pic_sec_gpu = len(indexes) - n_pic_first_gpu

    for i in range(n_proc):
        if i < n_proc - 1:
            if not i % 2:  # even
                proc_index.append(indexes[0:n_pic_first_gpu])
                indexes = indexes[n_pic_first_gpu:]
            else:  # odd
                proc_index.append(indexes[0:n_pic_sec_gpu])
                indexes = indexes[n_pic_sec_gpu:]
        else:
            proc_index.append(indexes)

    global node_process
    node_process = []
    global firefox_process
    firefox_process = []
    global terrain_server_process
    terrain_server_process = None

    if gpu_mode == '0' or gpu_mode == 'multi':
        subprocess.run('firefox -CreateProfile gpu0', shell=True)
    if gpu_mode == '1' or gpu_mode == 'multi':
        subprocess.run('firefox -CreateProfile gpu1', shell=True)


    use_first_gpu = "__NV_PRIME_RENDER_OFFLOAD=1 __VK_LAYER_NV_optimus=NVIDIA_only __GLX_VENDOR_LIBRARY_NAME=nvidia "
    subprocess.run('killall node', shell=True, stderr=subprocess.DEVNULL)


    for i, idx_ls in enumerate(proc_index):
        if gpu_mode == '0' or gpu_mode == '1':
            gpu = int(gpu_mode)
        elif gpu_mode == 'multi':
            gpu = (i + gpu_priority) % 2
        else:
            raise NotImplementedError
        if idx_ls.size == 0:  # If one (or many) processes don't have a single image.
            node_process.append(None)
            firefox_process.append(None)
            continue

        with open(os.path.join(DATA_FOLDER_PATH, 'msgtojs.txt'), 'w') as f:
            # write name, process index and poses index
            f.write(dataset_name + ',' + str(i) + ',' + ','.join(idx_ls.astype(str)))
        with open(os.path.join(DATA_FOLDER_PATH, 'scene_name.txt'), 'w') as f:
            f.write(scene_name)


        # Start terrain server once and only once. The assets are accessible to all rendering processes.
        if terrain_server_process is None:

            path_cesium_terrain_tile = os.path.join(DATA_FOLDER_PATH,'orthophoto_mosaic_processed','terrain-tiles')

            if not os.path.isdir(path_cesium_terrain_tile) :
                print(f"Cesium terrain tiles have not been found here : {path_cesium_terrain_tile}")

            path_cesium_server = os.path.join(USER_HOME,'go','bin','cesium-terrain-server')
            print('path_cesium_terrain_tile ', path_cesium_terrain_tile)


            terrain_server_process = subprocess.Popen(["{:s}/go/bin/cesium-terrain-server".format(USER_HOME),
                                                       "-dir", "{:s}".format(path_cesium_terrain_tile),
                                                       "-port", "{:d}".format(3000),
                                                       "-cache-limit", "4GB", "-no-request-log"]).pid




            print("Cesium terrain server: port number must be consistent in Cesium app.js script.")
            print("visit http://localhost:3000/tilesets/{:s}-serving/layer.json for sanity check!".format(scene_name))
            print("if everything is fine, you should see the meta-data json file")
            time.sleep(2)

        # Let node.js start

        path_server_js = os.path.join(PROJ_DIR,'server.js')

        print("path_server_js ",path_server_js)
        if not os.path.exists(path_server_js) :
            print(f"Server JS hav not been found here {path_server_js}")

        node_process.append(subprocess.Popen(["node", path_server_js, "--port", "{0}".format(8080 + i)]).pid)
        time.sleep(2)

        firefox_open_mode = "-new-instance -width 720 -height 640 -left 0 -top {0}".format((i % 2) * 640) \
            if i < 1 or (i < 2 and gpu_mode == 'multi') else "-new-window"

        profile = 'gpu0' if gpu == 0 else 'gpu1'
        command = "{0}firefox {1} -P {2} http://localhost:{3}".format(use_first_gpu if gpu == 1 or force_cuda else "",
                                                                      firefox_open_mode,
                                                                      profile, 8080 + i)
        print('ooooo')
        firefox_process.append(subprocess.Popen(["/bin/bash", "-c", command]).pid)
        time.sleep(2)  # Let firefox retrieve information in msgtojs.txt
    generating = any(node_process)
    closed_by_user = False
    while generating:
        time.sleep(10)
        finished = glob.glob('{0}/Downloads/{1}_*_finished.txt'.format(USER_HOME, dataset_name))
        for filename in finished:
            with open(filename, 'r') as f:
                closed_by_user = closed_by_user or f.readline() == "Tab closed by user"
            i = int(filename.split('_')[-2])
            if node_process[i]:
                subprocess.run('kill {:d}'.format(node_process[i]), shell=True)
                node_process[i] = None
            if firefox_process[i]:
                subprocess.run('kill {:d}'.format(firefox_process[i]), shell=True)
                firefox_process[i] = None
            if terrain_server_process is not None:
                subprocess.run('kill {:d}'.format(terrain_server_process), shell=True)
                terrain_server_process = None
            os.remove(filename)
        generating = any(node_process)
        move_files(dataset_name)
        if os.path.exists(os.path.join(DATA_FOLDER_PATH, 'msgtojs.txt')):
            os.remove(os.path.join(DATA_FOLDER_PATH, 'msgtojs.txt'))
        if os.path.exists(os.path.join(DATA_FOLDER_PATH, 'scene_name.txt')):
            os.remove(os.path.join(DATA_FOLDER_PATH, 'scene_name.txt'))
    if not closed_by_user:
        print("***** Dataset {:s} was generated with success! ***** ".format(dataset_name))
    else:
        print("***** Dataset generation was finished for dataset {:s} but at least one process "
              "was closed forcefully before the end! *****".format(dataset_name))


def signal_handler(signal, frame):
    """
    Signal handler for initialization.
    See https://manpages.debian.org/bullseye/manpages-dev/signal.2.en.html for further details.
    """
    if 'firefox_process' in globals():
        for pid in firefox_process:
            try:
                subprocess.check_output('kill {0}'.format(pid), shell=True, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    if 'node_process' in globals():
        for pid in node_process:
            try:
                subprocess.check_output('kill {0}'.format(pid), shell=True, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    if 'terrain_server_process' in globals():
        try:
            subprocess.check_output('kill {0}'.format(terrain_server_process), shell=True, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    sys.exit(0)


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, nargs='+',
                        help="Custom name of dataset(s) to generate. If several names are used, "
                             "datasets need to be *prepared* beforehand.")
    parser.add_argument("scene_name", type=str, nargs='+',
                        help="Name of the underlying scene assets to use. "
                             "It must be consistent with the pre-processed data.")
    parser.add_argument("-p", type=str, help="Preset file name (to generate brand new LHS poses)")
    parser.add_argument("-matchPhantom", type=str,
                        help="Provide a directory of real photos from DJI Phantom 4 to be matched")
    parser.add_argument("-prepare", action="store_true", default=False,
                        help="Use this flag to only prepare datasets generation without actually starting")
    parser.add_argument("-movefiles", action="store_true", default=False,
                        help="Move files from Download folder to dataset directory")
    parser.add_argument("-startindex", nargs='+', type=int, help="Start generation at specific index instead of 0")
    parser.add_argument("-stopindex", nargs='+', type=int,
                        help="Stop generation at specific index instead of going to the end")
    parser.add_argument("-nproc", type=int, default=1, help="Define number of processes to launch in parallel")
    parser.add_argument("-gpumode", type=str, default='0', choices=['0', '1', 'multi'],
                        help="Select the gpu to use. If nproc > 1, 'multi' will send half of the processes to "
                             "each GPU and use different firefox profiles. Note that for a new profile, "
                             "webgl.force-enabled might have to be enabled on firefox about:config")
    parser.add_argument("-restart", action="store_true", default=False,
                        help="Use this flag to force restart jobs (if not set, it will try to continue from last time)")
    parser.add_argument('-force_cuda', action='store_true', default=False,
                        help="To forcecully enable cuda feature to speed up rendering.")
    parser.add_argument("-cesiumhome", type=str, default=USER_HOME + '/Documents/Cesium',
                        help="Location of Cesium datasets home directory")
    opt = parser.parse_args()
    return opt





def define_poses(name, process_type : str, input_path : str = None):


    # Define numpy poses paths
    poses_prefix_name = name
    path_file_numpy_poses = os.path.join(DATA_FOLDER_PATH, 'presets', poses_prefix_name,f"{poses_prefix_name}_poses.npy")
    os.makedirs(os.path.dirname(path_file_numpy_poses), exist_ok=True)

    # If poses defined by drone_footages_poses
    if process_type == 'footage' :
        if not os.path.isdir(input_path) :
            msg = f"Input path {input_path} should be a directory for footage process type."
            raise ValueError(msg)
        path_folder_drone_footages = input_path
        prepare_phantom_drone_matching(path_folder_drone_footages, path_file_numpy_poses)

    # If poses defined by latin hypercube sampling
    elif process_type == 'lhs' :
        if not input_path.endswith('.json') :
            msg = "Input path {input_path} should ends with .json for a lhs process type."
            raise ValueError(msg)
        path_file_presets = input_path
        [n, lon, lat, h, yaw, pitch, roll, height_ref] = read_preset(path_file_presets)
        prepare_lhs(n, lon, lat, h, yaw, pitch, roll, height_ref, path_file_numpy_poses)

    elif process_type == 'move_file' :
        move_files(poses_prefix_name)

    else :
        msg = f"'process_type' must be within the list [lhs,footage,move]"
        raise ValueError(msg)

    return path_file_numpy_poses


def run(name,path_file_numpy_poses, start_index: int = None , stop_index : int = None, gpu_mode : str = 0 , force_cuda : bool = False, restart : bool= False):

    for j, (name, scene_name) in enumerate(zip(name, name)):
        start = start_index if start_index else 0
        stop = stop_index if stop_index else None
        start_server(name, name,path_file_numpy_poses, start_idx=start, stop_idx=stop, restart=restart,
                     n_proc=number_of_cpu, gpu_mode=gpu_mode, force_cuda=force_cuda)


if __name__ == "__main__":

    name = 'trial'
    process_type = 'lhs' # footage or lhs or move_file

    # input_path = "/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/drone_footages"
    input_path = "/home/regislongchamp/Desktop/temp_vnav/presets/presets.json"
    # input_path = None

    path_file_numpy_poses = define_poses(name, process_type = 'lhs',input_path=input_path)


    gpu_mode =  '0' # ['0', '1', 'multi']
    run(name,path_file_numpy_poses,start_index = None, stop_index = None, gpu_mode=gpu_mode,force_cuda = False, restart = False)



"""
def main():
    if len(args.name) > 1:
        if args.p is not None or args.matchPhantom is not None or args.prepare or args.movefiles:
            raise Exception("Cannot prepare new dataset(s) when several dataset names are provided. "
                            "Prepare datasets individually using -prepare first!")
    if len(args.name) != len(args.scene_name):
        raise Exception("Cannot have different number of custom dataset names and scene names!")
    if args.startindex is not None and not len(args.startindex) == len(args.name):
        raise Exception("Cannot have different number of datasets and start indexes")
    if args.stopindex is not None and not len(args.stopindex) == len(args.name):
        raise Exception("Cannot have different number of datasets and stop indexes")
    if len(args.name) == 1:
        # prepare new dataset id needed
        if args.movefiles:
            move_files(args.name[0])
            exit()
        elif args.p:
            [n, lon, lat, h, yaw, pitch, roll, height_ref] = read_preset(args.p)
            prepare_lhs(n, lon, lat, h, yaw, pitch, roll, height_ref, args.name[0])
        elif args.matchPhantom:
            prepare_phantom_drone_matching(args.matchPhantom, args.name[0])
    if not args.prepare:
        print("Following dataset(s) will be generated...")
        for dataset_name, scene_name in zip(args.name, args.scene_name):
            print('Dataset {:s} in scene {:s}'.format(dataset_name, scene_name))
        for j, (dataset_name, scene_name) in enumerate(zip(args.name, args.scene_name)):
            start = args.startindex[j] if args.startindex and args.startindex[j] else 0
            stop = args.stopindex[j] if args.stopindex and args.stopindex[j] else None
            start_server(dataset_name, scene_name, start_idx=start, stop_idx=stop, restart=args.restart,
                         n_proc=args.nproc, gpu_mode=args.gpumode, force_cuda=args.force_cuda)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    args = config_parser()
    np.random.seed(2021)
    CESIUM_HOME = args.cesiumhome
    if CESIUM_HOME[-1] == '/':
        CESIUM_HOME = CESIUM_HOME[:-1]
    main()

"""