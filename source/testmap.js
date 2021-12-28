// SIMPLE CESIUM APP TO TRY TO VISUALIZE AN ENVIRONMENT (WITHOUT TAKING ANY SNAPSHOT)

(function () {
    "use strict";
    // Group token
    Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkODAzZDIxNi03OWE5LTQ4MDEtYjY1MS05NjRjYTUxYWY1NDEiLCJpZCI6NjIyODAsImlhdCI6MTYyNjg3NjU1OH0.HnC4hqaE96z02820ZD3UwwGFuSoTiJn7iUtH6G4cpII'

    //////////////////////////////////////////////////////////////////////////
    // Creating the Viewer
    //////////////////////////////////////////////////////////////////////////
    var container = document.getElementById('cesiumContainer');
    container.style.width = '720px';
    container.style.height = '480px';
    window.alert("Welcome to the interactive test map for the demo dataset!");

    var fov_deg = 73.74;

    var viewer = new Cesium.Viewer('cesiumContainer', {
        requestRenderMode : true,
        maximumRenderTimeChange : Infinity,
        // terrainProvider: Cesium.createWorldTerrain(), // use cesium terrain
        // imageryProvider: Cesium.createWorldImagery(), // use Bing imagery
        terrainProvider: false,
        imageryProvider: false,
        baseLayerPicker: false,
        resolutionScale: 1.0,
        contextOptions : {
        	webgl : {
        		preserveDrawingBuffer: true,
        	},
        },
    });

    var tileset = viewer.scene.primitives.add(
  		new Cesium.Cesium3DTileset({
            url: Cesium.IonResource.fromAssetId(653067),
            maximumScreenSpaceError: 0.5, // the important parameter
            maximumMemoryUsage: 32768, // high value
            immediatelyLoadDesiredLevelOfDetail : true, //we only want to do a screenshot (not have approximate scene before)
            loadSiblings : false,
  		}))

    tileset.style = new Cesium.Cesium3DTileStyle({
            pointSize : '2.0'
        });

    // var position = new Cesium.Cartesian3(4368061.54, 502898.336, 4605607.55);
    var position = new Cesium.Cartesian3(4397285.00, 473703.00, 4581275.00);
    var orientation = new Cesium.HeadingPitchRoll.fromDegrees(90.0, -60, 0);
    viewer.scene.camera.setView({
        destination : position,
        orientation : {
            heading : orientation.heading,
            pitch : orientation.pitch,
            roll : orientation.roll
        }
    });
    viewer.scene.camera.frustum.fov = Cesium.Math.PI / 180 * fov_deg;
    viewer.scene.requestRender()
}());
