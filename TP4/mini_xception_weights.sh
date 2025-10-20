#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
curl -L -o models/mini_xception_fer2013.h5 \
https://github.com/oarriaga/face_classification/raw/refs/heads/master/trained_models/emotion_models/fer2013_mini_XCEPTION.99-0.65.hdf5
