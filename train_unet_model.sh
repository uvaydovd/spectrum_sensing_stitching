#!/bin/bash

set -e

python unet.py --dset $1 --normalize $2 --id_gpu $3
python -m tf2onnx.convert --saved-model ./tf_model --output ./unet.onnx 