#!/bin/bash

if [ $1 = "sudo_apt" ]
then
    sudo apt-get update -qq && sudo apt-get install -y libfluidsynth2 build-essential libasound2-dev libjack-dev
else
    apt-get update -qq && apt-get install -y libfluidsynth2 build-essential libasound2-dev libjack-dev
fi

pip install -r requirements.txt

# install mt3
# mkdir mt3
git clone --branch=main https://github.com/magenta/mt3
# git clone https://github.com/RafikHachana/figaro
# cd mt3
# cd mt3 && 
# mv mt3 mt3_tmp; pwd && mv mt3_tmp/* .; pwd && rm -r mt3_tmp
# exit
# TODO(iansimon): remove pinned numba/llvmlite/ddsp after ddsp updated
python3 -m pip install nest-asyncio numba==0.56.4 llvmlite==0.39.1 pyfluidsynth==1.3.0 -e ./mt3
python3 -m pip install --no-dependencies --upgrade ddsp

pip install gsutil
pip install protobuf==3.20

# copy checkpoints
gsutil -q -m cp -r gs://mt3/checkpoints .

# CLIP STUFF
git clone https://github.com/RafikHachana/CLIP2Video
pip install -r CLIP2Video/requirements.txt

gdown 1PvFterPV7eMBg6RXElqaCmdpI9aHiGp_
curl --output clip.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

# copy soundfont (originally from https://sites.google.com/site/soundfonts4u)
# gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .