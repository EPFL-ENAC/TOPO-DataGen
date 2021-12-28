import os.path

import numpy as np
import imageio
import argparse


def npy_to_point_cloud(input_file, output_file, png=None, geographic=None):
    """
    Transform .npy to point cloud
    Args:
        input_file:		input file path.
        output_file: 	output file path.
        png: 			PNG image used to colorize the point cloud.
        geographic: 	Defaults to None.
    """
    coord = np.load(input_file)
    output_file = os.path.abspath(output_file)
    res_xy = '7' if geographic else '2'
    if png:
        rgb = imageio.imread(png)
        out_string = "{0:." + res_xy + "f},{1:." + res_xy + "f},{2:.2f},{3},{4},{5}\n"
    else:
        out_string = "{0:." + res_xy + "f},{1:." + res_xy + "f},{2:.2f}\n"
    with open(output_file, 'w') as f_out:
        if png:
            f_out.write("x, y, z, Red, Green, Blue\n")
        else:
            f_out.write("x, y, z\n")
        for i, line in enumerate(coord):
            for j, pixel in enumerate(line):
                if float(pixel[0]) < 0:
                    continue
                if png:
                    f_out.write(out_string.format(pixel[0], pixel[1], pixel[2],
                                                  rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]))
                else:
                    f_out.write(out_string.format(pixel[0], pixel[1], pixel[2]))
    print("Task is done! Please find the output at {:s}".format(output_file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str,
                        help="Input npy file name.")
    parser.add_argument("output_file", type=str,
                        help="Output file name.")
    parser.add_argument("-png", type=str, default=None,
                        help="Add png to colorize. Use 'auto' for automatic search.")
    parser.add_argument("-geographic", action="store_true", default=False,
                        help="Use this for geographic coordinates "
                             "(to use sufficient resolution for longitude and latitude).")
    args = parser.parse_args()
    if args.png == 'auto':
        png_path = os.path.abspath(args.input_file.replace('_pc.npy', '_img.png'))
        if os.path.exists(png_path):
            args.png = png_path
        else:
            args.png = None
    npy_to_point_cloud(args.input_file, args.output_file, args.png, args.geographic)


if __name__ == "__main__":
    main()
