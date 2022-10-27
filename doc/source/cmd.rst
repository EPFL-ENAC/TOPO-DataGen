Command line utilities
**********************



Data downloading + preprocessing (EPSG:3857 - web marcator)
--------------------------------------------------------------

Prepare the data for the local and Cesium Ion processing.

.. argparse::
   :filename: ../../scripts/preprocess_data.py
   :func: parser
   :prog: python scripts/preprocess_data.py












Create rasters final products
-------------------------------

Create the different products (scene coordiantes, Semantics map, Euclidean depth, Surface normals, ORB keypoints).

.. argparse::
   :filename: ../../scripts/export_data.py
   :func: parser
   :prog: python scripts/export_data.py





