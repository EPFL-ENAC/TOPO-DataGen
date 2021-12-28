SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJ_DIR="$(dirname $SCRIPT_DIR)"
cd "$SCRIPT_DIR"
THREADS=$(grep -c ^processor /proc/cpuinfo)

echo "***** Point cloud data preprocessing starts. *****"
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

### PDAL to colorize point cloud ###
# prepare pdal json configuration
LAS_IN_DIR=$DATA_DIR/$DATA_NAME-surface3d
IMG_PATH=$DATA_DIR/$DATA_NAME-swissimage10/mergedVRT-lv95.vrt
COLORED_LAS_OUT_DIR=$DATA_DIR/$DATA_NAME-surface3d/color
ECEF_LAS_OUT_DIR=$DATA_DIR/$DATA_NAME-surface3d/ecef
ECEF_DS_LAS_OUT_DIR=$DATA_DIR/$DATA_NAME-surface3d/ecef-downsampled

mkdir $COLORED_LAS_OUT_DIR -p
mkdir $ECEF_LAS_OUT_DIR -p
mkdir $ECEF_DS_LAS_OUT_DIR -p

JS_STR='{
    "pipeline":
    [
        "LAS_IN_DIR/*.las",
        {
            "type": "filters.colorization",
            "raster": "IMG_PATH"
        },
        {
            "type": "filters.range",
            "limits": "Red[1:]"
        },
        {
            "type": "writers.las",
            "compression": "false",
            "minor_version": "2",
            "dataformat_id": "2",
            "a_srs": "EPSG:4978",
            "filename":"COLORED_LAS_OUT_DIR/mergedLAS-lv95.las"
        }
    ]
}'

echo "$JS_STR" > util_pointCloud_colorize.json
sed -i "s|IMG_PATH|$IMG_PATH|g" util_pointCloud_colorize.json

# colorize the point cloud via PDAL
echo "PDAL to colorize the point cloud..."
i=0
for item in "$LAS_IN_DIR"/*
do
  item_base="$(basename "$item" .las)"
  if [[ ${item_base:0:1} =~ ^[0-9] && ${item_base: -1} =~ ^[0-9] ]]
    then
      las_out=$COLORED_LAS_OUT_DIR
      las_out+=/${item_base}.las
      echo "Start processing $item ---> $las_out"
      pdal pipeline util_pointCloud_colorize.json --writers.las.filename="$las_out" --readers.las.filename="$item" &
      pid_ls[$i]=$!
      i=$((i+1))
      if [[ "$i" -eq "$THREADS" ]]
        then
          echo "Waiting for PDAL colorization process..."
          for pid in ${pid_ls[*]}; do
            wait $pid
          done
          i=0
      fi
  fi
done

# wait for all pid_ls
echo "Waiting for PDAL colorization process..."
for pid in ${pid_ls[*]}; do
    wait $pid
done

rm util_pointCloud_colorize.json
echo "PDAL session ends! Please find colorized point clouds at $COLORED_LAS_OUT_DIR"
### PDAL session ends ###

### Reframe the point cloud from lv95+ln02 to wgs84 ecef ###
# convert the altimetric ln02 value to bessel ellipsoidal height
# note: *reframeTransform.py* relies on *reframeLib.jar*
# reframe into wgs84 ecef
echo "reframeTransform to convert coordinates into ecef..."
i=0
for item in "$COLORED_LAS_OUT_DIR"/*
do
  item_base="$(basename "$item" .las)"
  if [[ ${item_base:0:1} =~ ^[0-9] && ${item_base: -1} =~ ^[0-9] ]]
    then
      las_out=$ECEF_LAS_OUT_DIR
      las_out+=/${item_base}.las
      echo "Start processing $item ---> $las_out"
      python "$PROJ_DIR"/scripts/reframeTransform.py "$item" "$las_out" \
             -s_h_srs lv95 -s_v_srs ln02 -t_h_srs wgs84 -t_v_srs wgs84 -silent &
      pid_ls[$i]=$!
      i=$((i+1))
      if [[ "$i" -eq "$THREADS" ]]
        then
          echo "Waiting for reframeTransform process..."
          echo "This might take some time depending on the computing hardware..."
          for pid in ${pid_ls[*]}; do
            wait $pid
          done
          i=0
      fi
  fi
done

# wait for all pid_ls
echo "Waiting for reframeTransform process..."
echo "This might take some time depending on the computing hardware..."
for pid in ${pid_ls[*]}; do
    wait $pid
done
echo "reframeTransform session ends! Please find colorized point clouds at $ECEF_LAS_OUT_DIR"
### Reframe session ends ###


### las point cloud downsampling for semantic retrieval ###
echo "To downsample the ECEF point clouds..."
n_tile=$(find $ECEF_LAS_OUT_DIR -maxdepth 1 -type f|wc -l)
echo "$n_tile tiles are found..."
if [[ $n_tile -gt 24 ]]
  then
    echo "Number of tiles is $n_tile, we use lower resolution to downsample the point clouds..."
    python "$SCRIPT_DIR"/util_las_downsample.py $ECEF_LAS_OUT_DIR --las_out_dir $ECEF_DS_LAS_OUT_DIR \
            --downsample_class 2 3 --voxel_size 2.0 3.0 --multi_process
  else
    echo "Number of tiles is $n_tile, we use normal resolution to downsample the point clouds..."
    python "$SCRIPT_DIR"/util_las_downsample.py $ECEF_LAS_OUT_DIR --las_out_dir $ECEF_DS_LAS_OUT_DIR \
            --downsample_class 2 3 --voxel_size 1.0 2.0 --multi_process
fi
echo "ECEF point cloud downsampling is done! Please find the downsampled point clouds at $ECEF_DS_LAS_OUT_DIR"
### downsampling session ends ###


### entwine to covert the ecef las into cesium 3d tiles ###
echo "entwine to do point cloud tiling..."

EPT_OUT_DIR=$LAS_IN_DIR/mergedLAS-ecef-ept
entwine build -i $ECEF_LAS_OUT_DIR/*.las -o $EPT_OUT_DIR \
              --scale 0.001 --deep --srs EPSG:4978 -t $THREADS -f -v

python $SCRIPT_DIR/util_ept_json_fix.py $EPT_OUT_DIR/ept.json

npx ept tile $EPT_OUT_DIR/ept.json -o $LAS_IN_DIR/pointCloud-tiles -t $THREADS -fv

cd $LAS_IN_DIR
mv pointCloud-tiles ../
### entwine session ends ###


echo "***** Point cloud data preprocessing is done. Please find it at $DATA_DIR/pointCloud-tiles *****"
