import os
import sys
from datetime import datetime
import requests
import typing
import pathlib
import pandas as pd
import shutil

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




def download_referenced_files(list_of_files : list, logger = None) -> None :
    """Download files locally

    Parameters
    ----------
    list_of_files : list
        list of path
    """

    for referenced_file in list_of_files :

        dataset_name = os.path.basename(os.path.dirname(referenced_file))
        dataset_path = os.path.dirname(os.path.dirname(referenced_file))
        folder_name = os.path.basename(referenced_file).replace('.csv','')
        folder_name = f"{dataset_name}-{folder_name}"
        folder_path = os.path.join(dataset_path,dataset_name,folder_name)

        if os.path.exists(folder_path) :
            shutil.rmtree(folder_path)

        os.makedirs(folder_path)


        df = pd.read_csv(referenced_file,header=None)
        for index, row in df.iterrows():
            url = row[0]

            file_name = os.path.basename(url)
            file_path = os.path.join(folder_path,file_name)

            if url.startswith('http') :

                if logger :
                    message = f"The downloading of  {file_path} has started"
                    logger.info(message)
                    start = datetime.now()

                response = requests.get(url)
                with open(file_path, 'wb') as f:
                    f.write(response.content)

                if logger :
                    downloading_time = (datetime.now() - start).seconds
                    message = f"The downloading of  {file_path} has finished in {downloading_time} seconds"
                    logger.info(message)

            else :
                if logger :
                    message = f"The file  {file_path} has not been downloaded"
                    logger.error(message)




if __name__ == "__main__":
    list_of_paths = get_list_files_from_directory(os.path.join(os.getcwd(),'data_sample'),'csv')
    download_referenced_files(list_of_paths[:1])
