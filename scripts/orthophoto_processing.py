##################################################
## This file provides tools to manupulate orthophotos
##################################################
## Copyright (c) 2021-2022 Topo-DataGen developers
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
##################################################
## Author: Regis Longchamp
## Copyright: Copyright 2022, Topo-DataGen ENAC OS Grant
## Credits: EPFL ENAC
## License: GNU General Public License
##################################################




import os
from datetime import datetime
import typing
import pathlib
from osgeo import gdal
from config import settings




def orthophoto_reprojection(path_in_file_orthophoto: typing.Union[str, pathlib.Path],
                            path_out_file_orthophoto : typing.Union[str, pathlib.Path],
                            epsg_in : int, epsg_out : int,logger = None) -> None :
    """
    Reproject raster data from one coordinate system to another one.
    :param path_in_file_orthophoto: file path of the input raster
    :param path_out_file_orthophoto: file path of the output raster
    :param epsg_in: EPSG of the input raster
    :param epsg_out: EPSG of the output raster
    :param logger: Logger object
    :return: None
    """

    if logger:
        message = f"The Orthophoto {path_in_file_orthophoto} reprojection process has started"
        logger.info(message)
        start = datetime.now()


    command = f"gdalwarp -s_srs  epsg:{epsg_in} -t_srs epsg:{epsg_out}  {path_in_file_orthophoto} {path_out_file_orthophoto} "
    os.system(command)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The orthophoto {path_in_file_orthophoto}  reprojection has finished in {downloading_time} seconds"
        logger.info(message)




def orthophotos_extent(path_in_file : typing.Union[str, pathlib.Path],
                       epsg_in :int, logger = None) -> tuple :
    """
    Compute WGS84 extent (EPSG:4326)
    :param path_in_file: Path of the orthophoto to calculate the lat/long extend
    :param epsg_in: Coordinate system of the input coordianate system
    :param logger: Logger object
    :return: tuple (xmin, ymin, xmax, ymax)
    """

    if logger:
        message = f"The Orthophoto {path_in_file} extent analysis process has started"
        logger.info(message)
        start = datetime.now()

    path_out = path_in_file.replace('.','_temp.')

    orthophoto_reprojection(path_in_file,path_out,epsg_in,'4326')

    ds = gdal.Open(path_out)
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    os.remove(path_out)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The orthophoto {path_in_file} extent analysis has finished in {downloading_time} seconds"
        logger.info(message)

    return (xmin, ymin, xmax, ymax)



def tile_orthophotos(path_in_file_orthophoto : typing.Union[str, pathlib.Path],
                        path_out_folder_tiled_orthophoto : typing.Union[str, pathlib.Path],
                     epsg : str = '4326',logger = None
                     ) -> None :
    """
    Generates directory with TMS tiles, KMLs and simple web viewers.
    :param path_in_file_orthophoto: Path of the orthophoto as basis for the tiling
    :param path_out_folder_tiled_orthophoto: Path of the output folder
    :param epsg: Input coordinate system
    :param logger: Logger object
    :return: None
    """

    if logger:
        message = f"The Orthophoto {path_in_file_orthophoto} tiling process has started"
        logger.info(message)
        start = datetime.now()

    command = f"gdal2tiles.py --zoom 0-20 --s_srs epsg:{epsg} --processes 12 {path_in_file_orthophoto} {path_out_folder_tiled_orthophoto}"
    os.system(command)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The orthophoto {path_in_file_orthophoto} tiling process has finished in {downloading_time} seconds"
        logger.info(message)




def merge_orthophotos(path_in_folder_orthophoto : typing.Union[str, pathlib.Path],
                      path_out_file_merged_orthophoto : typing.Union[str, pathlib.Path],
                      logger = None) -> None :

    """
    Merge multiple orthophotos into one tif file
    :param path_in_folder_orthophoto: Path of the folder that contains the input orthophotos
    :param path_out_file_merged_orthophoto: Path of the folder that contains the output merged orthophotos
    :param logger: Logger object
    :return: None
    """

    list_of_orthophoto = [os.path.join(path_in_folder_orthophoto,i) for i in os.listdir(path_in_folder_orthophoto) if i.endswith('.tif')]

    print(list_of_orthophoto)
    string_of_othophoto_paths = ' '.join(list_of_orthophoto)
    no_data_value = settings.orthophoto.no_data_value

    if logger:
        message = f"The Orthophoto {path_out_file_merged_orthophoto} merging process has started"
        logger.info(message)
        start = datetime.now()

    path_out_file_merged_orthophoto_temp = path_out_file_merged_orthophoto.replace('.tif','.vrt')

    if os.path.exists(path_out_file_merged_orthophoto_temp) :
        os.remove(path_out_file_merged_orthophoto_temp)

    if os.path.exists(path_out_file_merged_orthophoto) :
        os.remove(path_out_file_merged_orthophoto)

    list_tif_files_name = [os.path.join(path_in_folder_orthophoto,i) for i in os.listdir(path_in_folder_orthophoto) if i.endswith('.tif')]
    list_tif_files_name_text = " ".join(list_tif_files_name)

    if not os.path.exists(os.path.dirname(path_out_file_merged_orthophoto_temp)) :
        os.makedirs(os.path.dirname(path_out_file_merged_orthophoto_temp))

    # Merge raster into one
    command = f"gdalbuildvrt {path_out_file_merged_orthophoto_temp} {list_tif_files_name_text}"
    os.system(command)


    # Change raster resolution
    command = f"gdalwarp -dstnodata 0 -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co JPEG_QUALITY=100 -multi {path_out_file_merged_orthophoto_temp} {path_out_file_merged_orthophoto}"
    os.system(command)


    # # Create virtual layer in ESPG:4978 for the point cloud colorization
    # path_out_file_merged_orthophoto_temp_4978 = path_out_file_merged_orthophoto_temp.replace('.', '_4978.')
    # command = f"gdalwarp -s_srs  epsg:4326  -t_srs epsg:4978  -dstnodata 0 -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co JPEG_QUALITY=100 -multi {path_out_file_merged_orthophoto}  {path_out_file_merged_orthophoto_temp_4978} "
    # os.system(command)


    # if os.path.exists(path_out_file_merged_orthophoto_temp) :
    #     os.remove(path_out_file_merged_orthophoto_temp)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The orthophoto {path_out_file_merged_orthophoto} merging process has finished in {downloading_time} seconds"
        logger.info(message)



if __name__ == "__main__":
    merge_orthophotos('/media/regislongchamp/Windows/projects/TOPO-DataGen/data_preprocess/demo/demo-swissimage10','/media/regislongchamp/Windows/projects/TOPO-DataGen/data_preprocess/demo/demo-swissimage10/m.tif')