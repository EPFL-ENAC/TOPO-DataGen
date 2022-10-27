"use strict";

//////////////////////////////////////////////////////////////////////////
// Download functions
//////////////////////////////////////////////////////////////////////////

function downloadText(text, name) {
    var link = document.createElement("a");
    link.download = name;
    var file = new Blob([text], {type:'text/plain'});
    link.href = URL.createObjectURL(file);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    //delete link;
}

function downloadURI(uri, name) {
    var link = document.createElement("a");
    link.download = name;
    link.href = uri;
    // mimic click on "download button"
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    //delete link;
}

function downloadNpy(array, name) {
    var link = document.createElement("a");
    link.download = name;
    var file = new Blob([nd.io.npy_serialize(array)], {
        type: "binary/octet-stream"
    });
    link.href = URL.createObjectURL(file);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

//////////////////////////////////////////////////////////////////////////
// Tools
//////////////////////////////////////////////////////////////////////////

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

//////////////////////////////////////////////////////////////////////////
// Main loop
//////////////////////////////////////////////////////////////////////////

function prepareNextPic(){
    var img_idx = img_idx_list[ind];
    t1 = performance.now();
    if(t0 && currenttimeisvalid) {
        trun += (t1-t0)/60000;
        tind += 1;
        var trem = (img_idx_list.length - ind)*trun/tind;
        elapsedtimespan.innerHTML = trun.toFixed(2);
        remainingtimespan.innerHTML = trem.toFixed(2);
    }
    currenttimeisvalid = true;
    t0 = t1;

    htmlconsole.innerHTML = 'Rendering picture number ' + (ind+1) + '/' + img_idx_list.length + '...';
	var point = points[img_idx];

    if (srs === "geographic") {
	   var position = new Cesium.Cartesian3.fromDegrees(point[0],point[1],point[2]); // .fromDegrees needs (lon,lat,alt)
    } else {
        var position = new Cesium.Cartesian3(point[0],point[1],point[2])
    }
    var orientation = new Cesium.HeadingPitchRoll.fromDegrees(point[3], point[4], point[5]);
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
    addRenderListeners();
}

var takeScreenshot = function(j){ 
    var canvas = viewer.scene.canvas;
    canvas.toBlob(function(blob){
    var url = URL.createObjectURL(blob);
    downloadURI(url, name + "_img_"+j.toString()+".png");
    });
    console.log('Taking screenshot')
}

function computeCoordinates(i,j) {
    // Warning: experiments show there is a 1px vertical shift.
    let cur_pos = new Cesium.Cartesian2(j, i+1);
    let ppCartesian = viewer.scene.pickPosition(cur_pos);
    if (Cesium.defined(ppCartesian)){
        if (srs === "geographic") {
            let ppCartographic = Cesium.Cartographic.fromCartesian(ppCartesian);
            let lat = Cesium.Math.toDegrees(ppCartographic.latitude);
            let lon = Cesium.Math.toDegrees(ppCartographic.longitude);
            let h = ppCartographic.height;

            if(save_in_txt){
                coor_string += lon.toFixed(6) + "," + lat.toFixed(6) + "," + h.toFixed(2) + "\n";
            }
            else{
                coor_array.set([i,j,0], lon);
                coor_array.set([i,j,1], lat);
                coor_array.set([i,j,2], h);
            }
        } else { // geocentric
            if(save_in_txt){
                coor_string += ppCartesian.x.toFixed(6) + "," + ppCartesian.y.toFixed(6) + "," + ppCartesian.z.toFixed(2) + "\n";
            }
            else {
                coor_array.set([i,j,0], ppCartesian.x);
                coor_array.set([i,j,1], ppCartesian.y);
                coor_array.set([i,j,2], ppCartesian.z);
            }
        }
    }
    else{
        if(save_in_txt){
            coor_string += "-1,-1,-1\n";
        }
    }
}

async function picRendered() {
    var img_idx = img_idx_list[ind];
	console.log('Finished rendering picture number ' + (ind+1) + '/' + img_idx_list.length);
    // save scene coordinates
    if(coor){
        htmlconsole.innerHTML = 'Computing coordinates for picture ' + (ind+1) + '/' + img_idx_list.length + '...';
        await sleep(3);
		coor_array = nd.tabulate([viewer.canvas.height,viewer.canvas.width,3], 'float32', () => -1);
    	coor_string = "";

        var promises = [];
			for(var i=0;i<viewer.canvas.height;i++){
        	for(var j=0;j<viewer.canvas.width;j++){
            	promises.push(new Promise((resolve) => resolve(computeCoordinates(i,j))));
            }
            if (!((i+1)%10)) {
                await Promise.all(promises); // Let's compute 10 lines by 10 lines and let the browser do its stuff so that it does not hate us
                await sleep(5);
                promises = [];
            }               
        }
        if (promises) await Promise.all(promises);
        if(save_in_txt){
        	downloadText(coor_string, name + "_coor_"+img_idx.toString()+".txt");
        }
       	else{
        	downloadNpy(coor_array, name + "_coor_"+img_idx.toString()+".npy");
       	}
    }

    if (pics) takeScreenshot(img_idx);

    if (reload_request) {
        console.log("Reloading entities to free GPU memory");
        reload_request = false;
        await reloadCesiumEntities();
    }

    ind++;
    if (ind < img_idx_list.length & !pauseAsked) {
        prepareNextPic();
    } else if (ind >= img_idx_list.length) {
        // Creating this file tells python we are finished. It is in a timeout because sometimes the download is not finished
        setTimeout(function(){
            downloadText("Everything went fine",name+"_"+procid+"_finished.txt");
            htmlconsole.innerHTML = '<span style="color:green">Generation terminated with success !</br>Exit message sent to python. You can close this tab if it does not close automatically.</span>';
            window.removeEventListener("beforeunload",preventClosing);
            setTimeout(function(){close();},1000);
        },3000);
        remainingtimespan.innerHTML = 'Finished';
        document.getElementById("finishdiv").style.display = "table";
        document.getElementById("pausediv").style.display = "none";
    } else {
        paused = true;
    }
}

function continueIfAllTilesLoaded() {
    if (viewer.scene.globe.tilesLoaded) {
        removeRenderListeners();
        setTimeout(picRendered,waitTimeAfterRender);
    }
}

function addRenderListeners() {
    timeoutMaxWait = setTimeout(function(){
        console.log('Bypassing for picture ' + img_idx_list[ind]);
        removeRenderListeners();
        picRendered();
    },maxRenderTime);
    timeoutAlreadyReady = setTimeout(function(){
        console.log('View was already ready');
        continueIfAllTilesLoaded(); // If there is two identical views, the view is already ready so no end of rendering event occurs
    },waitTimeAfterRender);
    viewer.scene.globe.tileLoadProgressEvent.addEventListener(continueIfAllTilesLoaded);
    //viewer.scene.postRender.addEventListener(continueIfAllTilesLoaded);
}

function removeRenderListeners() {
    clearTimeout(timeoutMaxWait);
    clearTimeout(timeoutAlreadyReady);
    viewer.scene.globe.tileLoadProgressEvent.removeEventListener(continueIfAllTilesLoaded);
    //viewer.scene.postRender.removeEventListener(continueIfAllTilesLoaded);
}

//////////////////////////////////////////////////////////////////////////
// User interactions
//////////////////////////////////////////////////////////////////////////

window.addEventListener("keydown", event => {
    if (event.isComposing || event.keyCode === 32) {
        currenttimeisvalid = false;
        pauseAsked = !pauseAsked;
        if (paused && !pauseAsked) {
            paused = false;
            prepareNextPic();
        }
        var pausediv = document.getElementById("pausediv").style;
        if (pauseAsked) pausediv.display = "table";
        else pausediv.display = "none";
    }
});

function preventClosing(event) {
    event.returnValue = "Please exit with the red cross on top left to quit properly";
}
window.addEventListener("beforeunload",preventClosing);

function userClose() {
    downloadText("Tab closed by user",name+"_"+procid+"_finished.txt");
    window.removeEventListener("beforeunload",preventClosing);
    htmlconsole.innerHTML = '<span style="color:red">Exit message sent to python. You can close this tab if it does not close automatically.</span>';
    setTimeout(function(){close();},1000);
}
document.getElementById("exitimg").addEventListener("click",userClose);

//////////////////////////////////////////////////////////////////////////
// Initialization
//////////////////////////////////////////////////////////////////////////

// gets the current dataset name and then the points + start the rendering loop
function startLoop() {
    var myRequest = new Request('scripts/msgtojs.txt');
    fetch(myRequest).then(function(response) {
        return response.text().then(function(text) {
            text = text.split(',');
            name = text[0];
            procid = text[1];
            img_idx_list = text.slice(2)
            // read scene name
            fetch('scripts/scene_name.txt').then(function (response){
                return response.text().then(function(text) {
                scene_name = text;
                console.log("Detected scene name is: " + scene_name);
                addCesiumEntities();
                fetch("scripts/presets/" + name + "/" + name + "_poses.npy")
                    .then(function(response){
                        return(response.arrayBuffer())
                    })
                    .then(function(response){
                        var pointslist = new npyjs().parse(response);
                        pointslist = [].slice.call(pointslist.data);
                        while(pointslist.length) points.push(pointslist.splice(0,6));
                        setTimeout(prepareNextPic,1000); // Let a bit of time for the tileset to load
                    });
                });
            });
        });
    });
}


//////////////////////////////////////////////////////////////////////////
// Parameters
//////////////////////////////////////////////////////////////////////////

// HTML elements
var container = document.getElementById('cesiumContainer');
container.style.width = '720px';
container.style.height = '480px';
var htmlconsole = document.getElementById("htmlconsole");
var elapsedtimespan = document.getElementById("elapsedtimespan");
var remainingtimespan = document.getElementById("remainingtimespan");

// Generation parameters
var pics = true;
var coor = true;
var srs = "geocentric" // save in "geographic" coordinates or "geocentric" frame (wgs84 for both)
var save_in_txt = false; // save in txt or npy files
var maxRenderTime = 30000; // in case rendering gets really stuck for one picture, we don't want it to be used
var waitTimeAfterRender = 500; // Sometimes, for some reason everything is not actually loaded when it should be. Waiting for a few ms helps.
var fov_deg = 73.74; // Horizontal FOV for DJI Phantom 4
var reload_period = 18000000; // Reload entities every 5 h to free GPU memory
var scene_name = 'NULL'; // Automatically specified according to the communication text file issued by python script

// Global variables initialization
var img_idx_list; // Index list of all poses to generate
var ind = 0; // Global indexing variable
var points = []; // Global array used to store poses
var procid; // ID of nodejs instance
var timeoutMaxWait;
var timeoutAlreadyReady;
var name; // Dataset name
var coor_array;
var coor_string;
var t1,t0=NaN,trun=0.,tind=0,currenttimeisvalid = true; // global variable for remaining time estimate
var reload_request;
var tileset;

//////////////////////////////////////////////////////////////////////////
// Cesium initialization
//////////////////////////////////////////////////////////////////////////
function addCesiumEntities() {
    if (scene_name === "NULL"){
        throw "The scene name is not correct!";
    }
    console.log("Loading Cesium entities for scene " + scene_name);

        // Load 3D tiles
        tileset = viewer.scene.primitives.add(
            new Cesium.Cesium3DTileset({
                url: Cesium.IonResource.fromAssetId(1377422),
                // url: 'http://localhost:8080/data_preprocess/'+ scene_name + '/pointCloud-tiles/tileset.json',
                maximumScreenSpaceError: 0.5, // the important parameter
                maximumMemoryUsage: 32768, // high value
                immediatelyLoadDesiredLevelOfDetail : true, //we only want to do a screenshot (not have approximate scene before)
                loadSiblings : false,
             }));

        tileset.style = new Cesium.Cesium3DTileStyle({
            pointSize : '2.0'
        });

        tileset.tileFailed.addEventListener(function(error) {
            console.log('An error occurred loading tile: ' + error.url);
            console.log('Error: ' + error.message);
        });

    // general performance settings
    viewer.imageryLayers.addImageryProvider(new Cesium.TileMapServiceImageryProvider({
                url: 'http://localhost:8080/data_preprocess/' + scene_name + '/imagery-tiles'
                }));
    viewer.terrainProvider = new Cesium.CesiumTerrainProvider({
                    url: 'http://localhost:3000/tilesets/' + scene_name + '-serving'
                    });
    viewer.scene.globe.depthTestAgainstTerrain=true; // This has to be set to true for the 3D coordinates retrieval to work because of a bug in cesium
    // See https://github.com/CesiumGS/cesium/issues/8179
    viewer.scene.globe.maximumScreenSpaceError = 1.0;
    viewer.scene.globe.tileCacheSize = 10000;
    viewer.scene.globe.preloadSiblings = true;

    // remove sun or moon
    viewer.scene.moon.show = false;
    viewer.scene.sun.show = false;
    setTimeout(function() {reload_request = true;},reload_period);
}

function removeCesiumEntities() {
        tileset.destroy();
        viewer.scene.primitives.remove(tileset);

}

async function reloadCesiumEntities() {
    removeCesiumEntities()
    htmlconsole.innerHTML = 'Reloading cesium entities to free GPU memory';
    await sleep(4000);
    addCesiumEntities();
}

// TS token

Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJlODAzY2ZhZS0wNjViLTQwM2MtYmFmZS0wMzY1ODhjNWVkMmYiLCJpZCI6OTM5NjcsImlhdCI6MTY1MjcwNDA5NX0.U-QwCcJnPvEaDIDXGO1BlPLDMleNYEaGERMyvzChs_o'

var viewer = new Cesium.Viewer('cesiumContainer', {
    requestRenderMode : true,
    maximumRenderTimeChange : Infinity,
    // terrainProvider: Cesium.createWorldTerrain(), // use Cesium's built-in world terrain
    terrainProvider: false,
    // imageryProvider: Cesium.createWorldImagery(), // use Bing aerial map
    imageryProvider: false,
    baseLayerPicker: false,
    resolutionScale: 1.0,
    contextOptions : {
        webgl : {
            preserveDrawingBuffer: true,
        },
    },
});


//////////////////////////////////////////////////////////////////////////
// Let's go
//////////////////////////////////////////////////////////////////////////

var pauseAsked = false;
var paused = false;
startLoop();
