import os
import sys
from zipfile import ZipFile
import typing
import pathlib


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










if __name__ == "__main__":
    zip_files_with_patter('/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/classified_point_cloud_processed'
                          ,'/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/classified_point_cloud_processed/ok.zip',
                          '_4978.las','las')
