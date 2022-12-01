Command line utilities
========================================




Download + preprocess data
----------------------------------

Data downloading + preprocessing (EPSG:3857 - Web Mercator)

Prepare the data for the local and Cesium Ion processing.

.. argparse::
   :filename: ../scripts/preprocess_data.py
   :func: parser
   :prog: python scripts/preprocess_data.py


Data downloading + preprocessing (EPSG:2056 - swiss coordinate system)r

Please refer to `this page <https://github.com/EPFL-ENAC/TOPO-DataGen/blob/main/data_preprocess/notes.md>`__ for data preprocessing steps.

We provide a quick demo dataset for you to explore and reproduce the workflow using an open-sourced geodata database. The provided EPFL and comballaz assets are respectively used to produce the urbanscape and naturescape sets in our CrossLoc Benchmark Datasets.

Please note that there is no strictly standardized data preprocessing steps, and it is out of the scope of this repo.





Process data
-----------------


Tasks to do before running this processing :

* having run the `Data downloading + preprocessing <#data_downloading_+_preprocessing>`_
* having uploaded the points clouds and and digital surface model on Cesium Ion
* having referenced the token and AssetID into the config file



.. argparse::
   :filename: ../scripts/start_generate.py
   :func: parser
   :prog: python scripts/start_generate.py






Create rasters final products
----------------------------------
Create the different products (scene coordiantes, Semantics map, Euclidean depth, Surface normals, ORB keypoints).

.. argparse::
   :filename: ../scripts/export_data.py
   :func: parser
   :prog: python scripts/export_data.py



