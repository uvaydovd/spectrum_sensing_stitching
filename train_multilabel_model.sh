#!/bin/bash

set -e

#milin add commands to train network and convert to onnx file
python multilabel.py -ts $1 -vs $2 -d $3
