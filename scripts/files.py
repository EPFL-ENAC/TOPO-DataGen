import os
import sys
from zipfile import ZipFile
import typing
import pathlib
from config import settings


sys.path.insert(0,os.getcwd())


def get_list_files_from_directory(path_root_folder : typing.Union[str, pathlib.Path], file_type : str ) -> list:
    """This funtion returns a list of files that match the file type argument withing a directory and subdirectory.

    Parameters
    ----------
    path_root_folder : typing.Union[str, pathlib.Path]
        Path of the root folder
    file_type : typing.Literal[&quot;csv&quot;]
        Type of file. Currently only CSV

    Returns
    -------
    list
        List of paths
    """

    list_of_csv_file = []

    for path, subdirs, files in os.walk(path_root_folder):
        for name in files:
            if name.endswith(file_type) :
                list_of_csv_file.append(os.path.join(path, name))

    return list_of_csv_file

def zip_files_with_patter(input_folder : typing.Union[str, pathlib.Path], out_file_name : str, pattern,file_type)-> None :
    all_files = get_list_files_from_directory(input_folder,file_type)
    list_file_to_zip = [i for i in all_files if pattern in i]
    print(list_file_to_zip)
    with ZipFile(out_file_name, 'w') as zipObj :
        for file in list_file_to_zip :
            zipObj.write(file, os.path.basename(file))



def has_csv_file(input_folder : typing.Union[str, pathlib.Path]) -> bool :
    """
    Chech that the CSV files to download data exist
    :param input_folder: path of the folder that should contain the CSV files
    :return: True is all CSV have been found.
    """

    if not os.path.exists(input_folder) :
        raise Exception(f"Folder {input_folder} does not exist")


    list_of_csv_in_folder = get_list_files_from_directory(input_folder,'csv')

    if not list_of_csv_in_folder :
        raise Exception(f"Folder  {input_folder} does not contain any CSV")

    # Check point cloud CSV file
    check_point_clouds =  f'{settings.POINTS_CLOUD.INPUT_FOLDER_NAME}.csv'
    if not any(i for i in list_of_csv_in_folder if i.endswith(check_point_clouds)) :
        raise Exception(f"CSV  *{check_point_clouds} has not be found")

    # Check orthophoto CSV file
    check_orthophoto = f'{settings.ORTHOPHOTO.INPUT_FOLDER_NAME}.csv'
    if not any(i for i in list_of_csv_in_folder if i.endswith(check_orthophoto)):
        raise Exception(f"CSV  *{check_orthophoto} has not be found")

    # Check ditgital surface model CSV file
    check_dtm= f'{settings.DIGITAL_SURFACE_MODEL.INPUT_FOLDER_NAME}.csv'
    if not any(i for i in list_of_csv_in_folder if i.endswith(check_dtm)):
        raise Exception(f"CSV  *{check_dtm} has not be found")

    return True


def has_data_file(input_folder : typing.Union[str, pathlib.Path],dataset : str ) -> bool :
    """
    Check that the input file exists
    :param input_folder:
    :param dataset:
    :return:
    """
    if not os.path.exists(input_folder) :
        raise Exception(f"Folder {input_folder} does not exist")

    list_of_folder = [f for f in os.listdir(input_folder) if not os.path.isfile(os.path.join(input_folder, f))]

    # Check point cloud
    check_point_clouds = f'{dataset}-{settings.POINTS_CLOUD.INPUT_FOLDER_NAME}'
    if not any(i for i in list_of_folder if i.endswith(check_point_clouds)):
        raise Exception(f"Folder  {check_point_clouds} has not be found")
    list_file_point_cloud = get_list_files_from_directory(os.path.join(input_folder,check_point_clouds),'las')
    if not list_file_point_cloud :
        raise Exception(f"Folder  {check_point_clouds} should contain las files")

    # Check orthophoto
    check_orthophoto = f'{dataset}-{settings.ORTHOPHOTO.INPUT_FOLDER_NAME}'
    if not any(i for i in list_of_folder if i.endswith(check_orthophoto)):
        raise Exception(f"Folder  {check_orthophoto} has not be found")
    list_file_point_cloud = get_list_files_from_directory(os.path.join(input_folder, check_orthophoto), 'tif')
    if not list_file_point_cloud:
        raise Exception(f"Folder  {check_orthophoto} should contain tif files")

    # Check digital surface model
    check_dsm = f'{dataset}-{settings.DIGITAL_SURFACE_MODEL.INPUT_FOLDER_NAME}'
    if not any(i for i in list_of_folder if i.endswith(check_dsm)):
        raise Exception(f"Folder  {check_dsm} has not be found")
    list_file_point_cloud = get_list_files_from_directory(os.path.join(input_folder, check_dsm), 'tif')
    if not list_file_point_cloud:
        raise Exception(f"Folder  {check_dsm} should contain tif files")




if __name__ == "__main__":
    has_data_file('/home/regislongchamp/Documents/TOPO-DataGen/data_preprocess/demo','demo')

    # zip_files_with_patter('/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/classified_point_cloud_processed'
    #                       ,'/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/classified_point_cloud_processed/ok.zip',
    #                       '_4978.las','las')
