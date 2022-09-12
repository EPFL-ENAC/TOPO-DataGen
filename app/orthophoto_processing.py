import os
from datetime import datetime
import typing
import pathlib
import shutil
from osgeo import gdal,ogr,osr
from config import settings
import rasterio
import numpy as np
import progressbar
from scipy import interpolate



def orthophoto_reprojection(path_in_file_orthophoto: typing.Union[str, pathlib.Path],
                            path_out_file_orthophoto : typing.Union[str, pathlib.Path],
                            epsg_in : int, epsg_out : int) -> None :


    # Create virtual layer in ESPG:4978 for the point cloud colorization
    command = f"gdalwarp -s_srs  epsg:{epsg_in} -t_srs epsg:{epsg_out}  {path_in_file_orthophoto} {path_out_file_orthophoto} "


    # -dstnodata 0 -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co JPEG_QUALITY=100 -multi
    os.system(command)



def orthophotos_extend(path_in_file,epsg_in) :

    path_out = path_in_file.replace('.','_temp.')

    orthophoto_reprojection(path_in_file,path_out,epsg_in,'4326')

    ds = gdal.Open(path_out)
    xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
    width, height = ds.RasterXSize, ds.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel

    os.remove(path_out)

    return (xmin, ymin, xmax, ymax)











def merge_orthophotos(path_in_folder_orthophoto : typing.Union[str, pathlib.Path],
                      path_out_file_merged_orthophoto : typing.Union[str, pathlib.Path],
                      logger = None) -> None :

    """
    This function merge multipe orthophoto into one tif file
    :param path_in_folder_orthophoto: Path of the folder that contains the input orthophotos
    :param path_out_file_merged_orthophoto: Path of the folder that contains the output merged orthophotos
    :return: None
    """

    list_of_orthophoto = [os.path.join(path_in_folder_orthophoto,i) for i in os.listdir(path_in_folder_orthophoto) if i.endswith('.tif')]
    string_of_othophoto_paths = ' '.join(list_of_orthophoto)
    no_data_value = settings.orthophoto.no_data_value

    if logger:
        message = f"The Orthophoto {path_out_file_merged_orthophoto} merging process has started"
        logger.info(message)
        start = datetime.now()

    path_out_file_merged_orthophoto_temp = path_out_file_merged_orthophoto.replace('.tif','.vrt')

    list_tif_files_name = [os.path.join(path_in_folder_orthophoto,i) for i in os.listdir(path_in_folder_orthophoto) if i.endswith('.tif')]
    list_tif_files_name_text = " ".join(list_tif_files_name)


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
    path_in = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/orthophoto_mosaic_processed/orthophoto_merged.tif'
    orthophotos_extend(path_in,3857)


