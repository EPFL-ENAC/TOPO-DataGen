rm -rf terrain-tiles
mkdir terrain-tiles
ctb-tile -f Mesh -C -N -o terrain-tiles mergedTIF-wgs84.tif
ctb-tile -c 1 -l -f Mesh -C -o terrain-tiles mergedTIF-wgs84.tif
exit
