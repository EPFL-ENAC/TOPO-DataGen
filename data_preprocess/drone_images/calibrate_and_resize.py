import glob
import os
import argparse
import subprocess
import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm


def calibrate_and_resize(img_path: str, save_path: str, calibrate: bool = False, resize: bool = False) -> None:
    """
    Calibrate and resize the images.
    """
    img_name = os.path.basename(img_path)
    out_path = os.path.join(save_path, img_name)

    img_raw = Image.open(img_path)
    img = np.array(img_raw).copy()
    f = 3646.75
    height, width = img.shape[:2]

    if calibrate:
        mtx = np.array([[f, 0, width / 2 - 18.3],
                        [0, f, height / 2 + 17.4],
                        [0, 0, 1]])
        dist = np.array([-0.265, 0.105, 0.00019, 0.00055, -0.028])
        img = cv.undistort(img, mtx, dist, None, None)

    if resize:
        w = 720
        h = 480
        # rescale to 720x480
        resize_factor_w = width / w
        resize_factor_h = height / h
        resize_factor = min(resize_factor_w, resize_factor_h)
        img = cv.resize(img, (round(width / resize_factor), round(height / resize_factor)))
        img = img[round((img.shape[0] - h) / 2):round((img.shape[0] + h) / 2),
                  round((img.shape[1] - w) / 2):round((img.shape[1] + w) / 2)]
        print(img.shape)

    im = Image.fromarray(np.uint8(img))
    im.save(out_path, exif=img_raw.info['exif'])

    # copy all the metadata (exif + xmp) from the original file to the resized one:
    cmd = 'exiftool -TagsFromFile ' + img_path + ' "-all:all>all:all" ' + out_path
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    # exiftool creates a copy of the original file - 'resized.jpg_original' so we delete it
    try:
        os.remove(out_path + "_original")
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='+', help="Path of the dataset to correct images.")
    parser.add_argument("-calibrate", action="store_true", default=False, help="Use this flag to correct distortion.")
    parser.add_argument("-resize", action="store_true", default=False, help="To resize the image.")

    args = parser.parse_args()

    if args.resize:
        print("Warning: the resize functionality in this script is very primitive and does not have good performance!")
    assert any([args.calibrate, args.resize]), "At least one of calibrate or resize options must be enabled!"

    for i, this_path in enumerate(args.path):
        print("Processing folder {:d} / {:d}: {:s} ...".format(i+1, len(args.path), this_path))
        dataset_dir = os.path.abspath(this_path)
        images_ls = sorted(glob.glob(os.path.join(dataset_dir, '*.JPG')))
        save_path = os.path.join(os.path.dirname(dataset_dir), 'calibrated')
        os.makedirs(save_path, exist_ok=True)
        for img in tqdm(images_ls, desc="Processing images..."):
            calibrate_and_resize(img, save_path, args.calibrate, args.resize)


if __name__ == "__main__":
    main()
