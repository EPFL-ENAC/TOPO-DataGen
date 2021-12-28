import os
import glob
import numpy as np
from skimage import io, transform
import argparse
import multiprocessing as mp


def _func_downsampling(in_img, out_img, ttl_iter, mp_progress, mp_lock):
    img = io.imread(in_img)
    out = transform.resize(img, [480, 720], preserve_range=True, anti_aliasing=True)
    out = out.astype(np.uint8)
    io.imsave(out_img, out)
    # release memory
    img = None
    out = None
    with mp_lock:
        print("\rDownsampling {:s} ---> {:s}, progress: {:d}/{:d}".format(os.path.basename(in_img),
                                                                          os.path.basename(out_img),
                                                                          mp_progress.value, ttl_iter),
              end="", flush=True)
        mp_progress.value += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs='+', type=str, help="Path of the dataset to correct images.")
    args = parser.parse_args()

    dataset_dir_ls = args.path
    save_dir_ls = [os.path.abspath(os.path.join(data_dir, '../downsampled')) for data_dir in dataset_dir_ls]

    for dataset_dir, save_dir in zip(dataset_dir_ls, save_dir_ls):
        dataset_dir = os.path.abspath(dataset_dir)
        save_dir = os.path.abspath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        files = list(filter(os.path.isfile, glob.glob(dataset_dir + "/*")))
        files.sort(key=lambda x: os.path.getmtime(x))

        # multiprocessing preparation
        mp_args_ls = []
        mp_manager = mp.Manager()
        mp_lock = mp_manager.Lock()
        mp_progress = mp_manager.Value('i', 1)
        for idx, file in enumerate(files):
            old_name = os.path.abspath(file)
            new_name = os.path.join(save_dir, os.path.basename(old_name))
            mp_args_ls.append((old_name, new_name, len(files), mp_progress, mp_lock))

        # mp pools
        with mp.Pool(processes=os.cpu_count()) as p:
            p.starmap(_func_downsampling, mp_args_ls, chunksize=1)

        print("\nTask is finished! {:s} ---> {:s}".format(dataset_dir, save_dir))


if __name__ == '__main__':
    main()
