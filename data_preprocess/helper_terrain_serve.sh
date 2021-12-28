SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJ_DIR="$(dirname $SCRIPT_DIR)"
cd "$SCRIPT_DIR"

echo "***** Terrain data serving starts. *****"
DATA_DIR=$1
if [ -z "$DATA_DIR" ]
then
  echo "DATA_DIR is empty"
  DATA_DIR='./'
else
  echo "DATA_DIR is set"
fi
echo $DATA_DIR

PORT=$2
if [ -z "$PORT" ]
then
  echo "PORT is empty"
  PORT='3000'
else
  echo "PORT is set"
fi
echo $PORT

cd $SCRIPT_DIR
echo "port number $PORT must be consistent in Cesium app.js script."

echo "visit http://localhost:$PORT/tilesets/$DATA_DIR-serving/layer.json for sanity check!"
echo "if everything is fine, you should see the meta-data json file"
# note: -dir path must end with /tilesets/terrain/, e.g., YOUR_PARENT_DIR/tilesets/terrain/
~/go/bin/cesium-terrain-server -dir $DATA_DIR/tilesets/terrain -port $PORT -cache-limit 4GB -no-request-log

echo "***** Terrain data serving is done. *****"
