SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

echo "***** Terrain data preprocessing starts. *****"
DATA_DIR=$1
if [ -z "$DATA_DIR" ]
then
  echo "DATA_DIR is empty"
  DATA_DIR='./'
else
  echo "DATA_DIR is set"
fi
echo $DATA_DIR
DATA_NAME=$DATA_DIR

# cd into the data directory of terrain .tif files
cd $DATA_DIR/$DATA_DIR-surface3d-raster
DATA_DIR=$(pwd)

### planimetric lv95 + altimetric ln02 ---> planimetric lv95 + altimetric wgs84 ###
# merge all .tif files into a virtual raster, use QGIS for visual sanity check
gdalbuildvrt mergedVRT-lv95-ln02.vrt *.tif

# generate the merged tif, preserve the LZW compression
# note: it's advised to use the same compression as the raw data
gdalwarp -s_srs epsg:2056 -t_srs epsg:2056 -dstnodata -9999 -co COMPRESS=LZW -multi mergedVRT-lv95-ln02.vrt mergedTIF-lv95-ln02.tif

# convert the altimetric ln02 value to wgs84 ellipsoidal height
# note: you must copy and paste the *reframeTransform.py* and *reframeLib.jar* from ${CESIUM_PROJ}/scripts to current folder
python "$PROJ_DIR"/scripts/reframeTransform.py mergedTIF-lv95-ln02.tif mergedTIF-lv95-wgs84.tif -s_h_srs lv95 -s_v_srs ln02 -t_h_srs lv95 -t_v_srs wgs84 -transform_res 5

### planimetric lv95 + altimetric wgs84 ---> wgs84 ###
# transform the LV95+wgs84 CRS into wgs84 as required by ctb-docker, preserve the LZW compression
# note: it's advised to use the same compression as the raw data
gdalwarp -s_srs epsg:2056 -t_srs epsg:4979 -dstnodata -9999 -co COMPRESS=LZW -multi mergedTIF-lv95-wgs84.tif mergedTIF-wgs84.tif

# clean up intermediate results
rm mergedVRT-lv95-ln02.vrt mergedTIF-lv95-ln02.tif

# wgs84 ---> Cesium terrain files ###
echo "----- Start to enter docker environment for terrain tiling -----"
SH_STR='rm -rf terrain-tiles
mkdir terrain-tiles
ctb-tile -f Mesh -C -N -o terrain-tiles mergedTIF-wgs84.tif
ctb-tile -c 1 -l -f Mesh -C -o terrain-tiles mergedTIF-wgs84.tif
exit'
echo "$SH_STR" > util_terrain_preprocess.sh

# the terrain meshing tool is adopted from https://github.com/tum-gis/cesium-terrain-builder-docker
sudo docker rm ctb-topo-datagen
sudo docker create --rm -v "$DATA_DIR":"/data" --name ctb-topo-datagen tumgis/ctb-quantized-mesh
sudo docker run --rm --volumes-from ctb-topo-datagen tumgis/ctb-quantized-mesh bash util_terrain_preprocess.sh
sudo docker run --rm --volumes-from ctb-topo-datagen tumgis/ctb-quantized-mesh rm util_terrain_preprocess.sh
sudo docker rm ctb-topo-datagen
echo "----- End of docker session -----"

sudo chmod -R 777 terrain-tiles

rm -rf ../terrain-tiles
mv terrain-tiles ../
cd ../terrain-tiles
TER_DIR=$(pwd)

# serve the cesium tiles
#docker run -p 9000:8000 -v $DATA_DIR:/data/tilesets/terrain --env WEB_DIR=/data/tilesets/terrain/cesium/ geodata/cesium-terrain-server

# create pointer to the terrain tiles
TER_DIR=$SCRIPT_DIR/$DATA_NAME/tilesets/terrain
mkdir "$TER_DIR" -p
rm -f "$TER_DIR"/$DATA_NAME-serving
ln -s "$SCRIPT_DIR"/"$DATA_NAME"/terrain-tiles "$TER_DIR"/$DATA_NAME-serving

echo "***** Terrain data preprocessing is done. Please find it at $TER_DIR *****"
