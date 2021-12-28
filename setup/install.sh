#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJ_DIR="$(dirname $SCRIPT_DIR)"
cd $PROJ_DIR

# install basics
sudo apt-get update
sudo apt-get install default-jre curl exiftool -y

# install docker
sudo apt-get install apt-transport-https ca-certificates gnupg lsb-release -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y

# install cesium terrain server
# adopted from https://github.com/geo-data/cesium-terrain-server
sudo apt update
sudo apt install golang-go -y
go get github.com/geo-data/cesium-terrain-server/cmd/cesium-terrain-server
echo "detected go package: $(ls ~/go/bin)"

# install nodejs
curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
sudo apt-get install -y nodejs

# npm init
sudo ldconfig
npm install

# install conda environment
cd $SCRIPT_DIR
conda env create -f environment.yml