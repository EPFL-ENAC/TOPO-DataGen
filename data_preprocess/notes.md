# swisstopo dataset preprocessing for Cesium

This README introduces the pipeline to preprocess [**swisstopo** free geodata](https://shop.swisstopo.admin.ch/en/products/free_geodata) for use in [**CesiumJS**](https://cesium.com/platform/cesiumjs/). The swisstopo terms of use could be found [here](https://www.swisstopo.admin.ch/en/home/meta/conditions/geodata/ogd.html).


## 0. Download raw data
```bash
# we provide selected download links for an area of interest
export DATASET=demo
python helper_download_from_csv.py $DATASET
```

## 1. Terrain tiles preprocessing

```bash
# we process the downloaded .tif surface models and do tiling in quantized-mesh format
export DATASET=demo
bash helper_terrain_preprocess.sh $DATASET
```

**Sanity check on terrain data serving**

Note: it helps debug but is **not a must** during preprocessing.

```bash
# serve the local tiles for Cesium usage. $PORT number must be consistent with Cesium app.js
export DATASET=demo
export PORT=3000
bash helper_terrain_serve.sh $DATASET $PORT
# please check the 'install cesium terrain server' section at ../setup/install.sh in case of error
```

## 2. Imagery tiles preprocessing

```bash
# we process the downloaded .tif orthophotos and do tiling in TMS format
export DATASET=demo
bash helper_imagery_preprocess.sh $DATASET
```

## 3. Cesium 3D tiles preprocessing

```bash
# we process the downloaded .las point clouds and do tiling in Cesium 3D tiles format
export DATASET=demo
bash helper_pointCloud_colorize.sh $DATASET
```

## 4. Things to know about swisstopo data

* Swiss coordinate system
  * typically we have raw data in **planimetric** `lv95` and **altimetric** ` ln02` systems ([local swiss reference frames](https://www.swisstopo.admin.ch/en/knowledge-facts/surveying-geodesy/reference-frames/local.html)), and we would stick with the standard `wgs84` ellipsoid (`epsg:4326`).
  * the planimetric (planar coordiantes) conversion from `lv95` to `wgs84` is straightforward and reliable using `gdalwarp`.
  * the altimetric (elevation/height) conversion from `ln02` to `wgs84` is obtained by approximate results in the `wgs84` ellipsoid, i.e., `ln02 ---> (reframeTransform.py) ---> wgs84`. The helper function `reframeTransform.py` is at `/scripts` folder. The numerical accuracy loss during the conversion is generally very small, and please refer to [swisstopo](https://www.swisstopo.admin.ch/content/swisstopo-internet/en/online/calculation-services/_jcr_content/contentPar/tabs/items/documents_publicatio/tabPar/downloadlist/downloadItems/20_1467104436749.download/refsys_e.pdf) and [swissREFRAME](https://github.com/hofmann-tobias/swissREFRAME) for further discussion.
* Rules of thumb regarding height system
  * if surface raster `tif` is used, it's straightforward to convert its `ln02` height to `wgs84` height with `reframeTransform.py` and then covert its planimetric coordinates from`ln95` to `wgs84`  with `gdalwarp`. For surface point cloud `laz` , `reframeTransform.py` could map the `lv95` + `ln02` 3D coordinates into `wgs84 ECEF`  coordinates directly.
* This [repo](https://github.com/bertt/awesome-quantized-mesh-tiles) records a nice list of available conversion tools.

## 5. Citation

If you find our code useful for your research, please cite the paper:

````bibtex
@article{yan2021crossloc,
  title={CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data},
  author={Yan, Qi and Zheng, Jianhao and Reding, Simon and Li, Shanci and Doytchinov, Iordan},
  journal={arXiv preprint arXiv:2112.09081},
  year={2021}
}
````

