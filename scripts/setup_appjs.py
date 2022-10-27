import os
import re
from config import settings


def setup_app_js(file_path : str) :
    cesium_asset_id = settings.CESIUM_ASSET_ID
    cesium_token = settings.CESIUM_TOKEN

    file = open(file_path, "r")
    list_lines = []

    for line in file:
        if 'Cesium.Ion.defaultAccessToken' in line :
            new_line = re.sub(r"\'[^)]*\'", f"'{cesium_token}'", line)
        elif 'url: Cesium.IonResource.fromAssetId' in line :
            new_line = re.sub(r'\([^)]*\)', f'({cesium_asset_id})', line)
        else :
            new_line = line


        list_lines.append(new_line)

    file_content = ''.join(list_lines)

    with open(file_path, 'w') as fp:
        fp.write(file_content)


if __name__ == "__main__":
    setup_app_js('/media/regislongchamp/Windows/projects/vnav/TOPO-DataGen-current-dev/source/app.js')
