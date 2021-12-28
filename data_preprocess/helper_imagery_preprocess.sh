SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJ_DIR="$(dirname $SCRIPT_DIR)"
cd "$SCRIPT_DIR"

echo "***** Imagery data preprocessing starts. *****"
DATA_DIR=$1
if [ -z "$DATA_DIR" ]
then
  echo "DATA_DIR is empty"
  DATA_DIR='./'
else
  echo "DATA_DIR is set"
fi
echo $DATA_DIR

# cd into the data directory of orthophoto .tif files
cd $DATA_DIR/$DATA_DIR-swissimage10

# merge all .tif files into a virtual raster, use QGIS for visual sanity check
gdalbuildvrt mergedVRT-lv95.vrt *.tif

# transform the LV95 CRS into WGS84, preserve the YCbCr JPEG compression
# note: it's advised to use the same compression as the raw data
gdalwarp -s_srs epsg:2056 -t_srs epsg:4326 -dstnodata 0 -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co JPEG_QUALITY=100 -multi mergedVRT-lv95.vrt mergedTIF-wgs84.tif

# make TMS tiles from the tif
# note: to avoid PROJ package error
echo "If you see PROJ related error, please run 'conda deactivate' before running the script!"

# note: --processes should be adjusted according to the machine CPU
gdal2tiles.py --zoom 0-20 --s_srs epsg:4326 --processes 12 mergedTIF-wgs84.tif ../imagery-tiles

cd ../imagery-tiles
IMG_DIR=$(pwd)

echo "***** Imagery data preprocessing is done. Please find it at $IMG_DIR *****"
