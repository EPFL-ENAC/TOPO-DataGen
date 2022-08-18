# swisstopo dataset preprocessing for Cesium

This README introduces the pipeline to preprocess [**swisstopo** free geodata](https://shop.swisstopo.admin.ch/en/products/free_geodata) for use in [**CesiumJS**](https://cesium.com/platform/cesiumjs/). The swisstopo terms of use could be found [here](https://www.swisstopo.admin.ch/en/home/meta/conditions/geodata/ogd.html).
  1. Please first go through the swisstopo free geodata website and refer to **General Data Flow** in [`**README.md**`](../README.md):
      + [swissSURFACE3D](https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html):  
          **Lidar Point Clouds:** models all natural and man-made objects of the surface of Switzerland in the form of a classified point cloud.
      + [swissSURFACE3D Raster](https://www.swisstopo.admin.ch/en/geodata/height/surface3d-raster.html):  
          **DTMs:** a digital surface model (DSM) which represents the earth’s surface including visible and permanent landscape elements
      + [SWISSIMAGE 10 cm](https://www.swisstopo.admin.ch/en/geodata/images/ortho/swissimage10.html):  
          **Orthophotos:** a composition of new digital color aerial photographs over the whole of Switzerland with a ground resolution of 10 cm
  2. Build the folder with scene name under the data_preprocess folder, e.g. `./data_preprocess/scene_name`
  3. Locate your target area in the above three links and download the csv file from the webpage to `./data_preprocess/scene_name` folder. 
     We recommand to paddle the target area when you choose by rectangle, e.g. if you are interested in a 3 \* 3 area then choose 5 \* 5 blocks centered on it.  
     Operations: Selection mode: select by rectangles -> click *New rectangle buttun* -> choose the area of interest in the map 
     -> keep default options and click *search* -> click *export all links* -> click *File ready. Click here to download* -> save to `./data_preprocess/scene_name`
  4. Run the scripts below to download the files and generate the local geodata.

## 0. Download raw data
```bash
cd data_preprocess
export DATASET=demo
python helper_download_from_csv.py $DATASET
```

## 1. Terrain tiles preprocessing
we process the downloaded .tif surface models and do tiling in quantized-mesh format

```bash
bash helper_terrain_preprocess.sh $DATASET
```

**Sanity check on terrain data serving**

Note: it helps debug but is **not a must** during preprocessing.

```bash
# serve the local tiles for Cesium usage. $PORT number must be consistent with Cesium app.js
export PORT=3000
bash helper_terrain_serve.sh $DATASET $PORT
# please check the 'install cesium terrain server' section at ../setup/install.sh in case of error
```

## 2. Imagery tiles preprocessing
we process the downloaded .tif orthophotos and do tiling in TMS format

```bash
bash helper_imagery_preprocess.sh $DATASET
```

## 3. Cesium 3D tiles preprocessing
we process the downloaded .las point clouds and do tiling in Cesium 3D tiles format

```bash
bash helper_pointCloud_colorize.sh $DATASET
```

## 4.Cesium Ion streaming configuration
1. Create a Cesium Ion account and enter **My Assets** page
2. Upload the following asset respectively:  
    (1) Compress`./$DATASET/$DATASET-surface3d/ecef` folder in .zip file  
    
    (2) `./$DATASET/$DATASET-surface3d-raster/mergedTIF-wgs84.tif`

    note: When uploading the .tif file, select the kind as **raster terrain** and choose base terrain as **Cesium World Terrain**.
    
3. Adjust the location of the 3D tile asset with '*-ecef.zip' source file.
   1. Click the **Adjust Tileset Location** button on the right top preview window of the 3D tile asset.
   2. Click the **Global Settings** on the top left
   3. Select the Terrain as '*-mergedTIF-wgs84' we uploaded and click 'Back to Assets' to save the changes.
4. Copy the `ID` and `access_token` of the 3D tile asset with '*-ecef.zip' source file. The token can be accessed via **Access Token** besides 'My Assets' tab. 
5. Configure the Cesium connection and rendering settings in `../source/app.js`:  
   + Access Token: `Cesium.Ion.defaultAccessToken`  
   + Asset_id: `url: Cesium.IonResource.fromAssetId()`  
   + Resolution: `container.style.width = '720px'; container.style.height = '480px';`
   + camera parameters: `var fov_deg = 73.74;`

**Warning**: If you use other Ceisum 3D tiles data source instead of the Swisstopo Free Geodata and the model is already calibrated 
in wgs84 coordinate system and finished the Swisstopo Free Geodata preprocess, please comment line 358 to line 363 in `../source/app.js`**!!!** Otherwise, the local file will overlap with the server file you upload!


## 5. Things to know about swisstopo data

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

