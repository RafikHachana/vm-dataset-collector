#!/bin/bash

apt-get update -qq && apt-get install -qq libfluidsynth2 build-essential libasound2-dev libjack-dev

# install mt3
git clone --branch=main https://github.com/magenta/mt3
mv mt3 mt3_tmp; mv mt3_tmp/* .; rm -r mt3_tmp
# TODO(iansimon): remove pinned numba/llvmlite/ddsp after ddsp updated
python3 -m pip install nest-asyncio numba==0.56.4 llvmlite==0.39.1 pyfluidsynth==1.3.0 -e .
python3 -m pip install --no-dependencies --upgrade ddsp

pip install gsutil

# copy checkpoints
gsutil -q -m cp -r gs://mt3/checkpoints .

# copy soundfont (originally from https://sites.google.com/site/soundfonts4u)
gsutil -q -m cp gs://magentadata/soundfonts/SGM-v2.01-Sal-Guit-Bass-V1.3.sf2 .