
# cd into the data directory of terrain .tif files
cd C:\projects\vnav\data_sel_grant\digital_surface_model_2056
DATA_DIR=$(pwd)

### planimetric lv95 + altimetric ln02 ---> planimetric lv95 + altimetric wgs84 ###
# merge all .tif files into a virtual raster, use QGIS for visual sanity check
gdalbuildvrt mergedVRT-lv95-ln02.vrt *.tif

# generate the merged tif, preserve the LZW compression
# note: it's advised to use the same compression as the raw data
gdalwarp -s_srs epsg:2056 -t_srs epsg:2056 -dstnodata -9999 -co COMPRESS=LZW -multi mergedVRT-lv95-ln02.vrt mergedTIF-lv95-ln02.tif

# convert the altimetric ln02 value to wgs84 ellipsoidal height
# note: you must copy and paste the *reframeTransform.py* and *reframeLib.jar* from ${CESIUM_PROJ}/scripts to current folder
python reframeTransform.py mergedTIF-lv95-ln02.tif mergedTIF-lv95-wgs84.tif -s_h_srs lv95 -s_v_srs ln02 -t_h_srs lv95 -t_v_srs wgs84 -transform_res 5

### planimetric lv95 + altimetric wgs84 ---> wgs84 ###
# transform the LV95+wgs84 CRS into wgs84 as required by ctb-docker, preserve the LZW compression
# note: it's advised to use the same compression as the raw data
gdalwarp -s_srs epsg:2056 -t_srs epsg:4979 -dstnodata -9999 -co COMPRESS=LZW -multi mergedTIF-lv95-wgs84.tif mergedTIF-wgs84.tif
