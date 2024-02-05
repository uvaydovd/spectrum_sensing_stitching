# Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching
# UNDER CONSTRUCTION: IN PROCESS OF FINALIZING AND UPLOADING DATA

This repo contains the code used in the paper: "Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching".
Our model is an altered 1D version of the famous UNet model with added multi-label multi-class functionality and non-local attention mechanism. 
This model generates a label for every single IQ/sub-band (in frequency domain). The network takes as input 1024 IQs in frequency domain and outputs a class for each. This means that 25MHz is broken
into 25MHz/1024 = 0.0244MHz sub-bands.

## Environment:

To preprocess data, generate datasets, train standard unet, or evaluate onnx model import this environment from yml:

    conda env create -n unet --file unet.yml
    conda activate unet

To train milinet import this environment from yml:

    conda env create -n multilabel --file multilabel.yml
    conda activate multilabel


## Code:

bin2h5.py - Takes raw IQ bin files collected with USRPs and prepares them for a signal bank to be used by data generators 
(not provided).

    usage: bin2h5.py

data_gen.py - Dataset generator where signals do not overlap, signal labels are 1D arrays with multi-class labels. 
This code is used to generate the dataset used to train the standard UNet. Highly suggest looking into code and making change
to hyper-parameters as needed.

    usage: data_gen.py

data_gen_overlap.py - Dataset generator where signals do overlap, signal labels are 2D matrices where each row is a
different class for multi-label multi-class fashion. This dataset is used to train MiliNet. Highly suggest looking into code
and making changes to hyper-parameters as needed.

    usage: data_gen_overlap.py

train_unet_model.sh - convenience bash script that trains the model with unet.py then converts the model to onnx for running

    usage: bash train_unet_model.sh DSET_FP NORMALIZE ID_GPU
    example: bash train_unet_model.sh ./dset.h5 False 0

train_milinet_model.sh - convenience bash script that trains the model with MILINPYTHONFILE then converts the model to onnx for running

    usage: bash train_milinet_model.sh
    example: bash train_milinet_model.sh

run_model.sh - convenience bash script that simply runs eval_DL_onnx.py below,
you can run it this way or call the python script directly up to you

    usage: bash run_model.sh ID_GPU SAMP_RATE INPUT_DIR_FP MODEL_FP NORMALIZE
    example: bash run_model.sh 0 25 ./NEU_wifi_1 ./unet.onnx True

unet.py - Use this file to train the UNet model, note: dataset is not provided

    usage: unet.py [-h] [--dset DSET] [--normalize NORMALIZE] [--id_gpu ID_GPU]

    GPU and Model Specifications

    optional arguments:
      -h, --help                show this help message and exit
      --dset DSET               filepath of h5 dataset
      --normalize NORMALIZE     choose whether to l2 normalize input or not
      --id_gpu ID_GPU           Choose GPU to use

    example: python unet.py --dset ./dset.h5 --normalize False --id_gpu 0

multilabel.py - Use this file to train the customized unet model
```bash
usage: multilabel.py [-h] [-ts] [-vs] [-d]

options:
  -h, --help         show this help message and exit
  -ts , --TrainSet   filepath of training set (default:
                     ./overlap_1024_25mhz_3days_train_2sig.h5)
  -vs , --ValSet     filepath of valiation set (default:
                     ./overlap_1024_25mhz_3days_test_2sig.h5)
  -d , --Device      specify the device for running model (default: -1)
```

eval_DL_onnx.py - Use this file to run the model on raw IQs

    usage: eval_DL_onnx.py [-h] [--id_gpu ID_GPU] [--samp_rate SAMP_RATE] [--input INPUT] [--model MODEL] [--normalize NORMALIZE]

    GPU and Model Specifications

    optional arguments:
      -h, --help                show this help message and exit
      --id_gpu ID_GPU           specify which gpu to use.
      --samp_rate SAMP_RATE     specifies the sampling rate in MHz, for now must be multiple of 25 MHz
      --input INPUT             specifies the IQ samples bin file to be fed to the network
      --model MODEL             specifies the model filepath
      --normalize NORMALIZE     specifies whether to normalize data or not

    example: python eval_DL_onnx.py --id_gpu 0 --samp_rate 25 --input ./test_data.bin --model ./unet.onnx --normalize False



### Contact info:

Daniel Uvaydov
(917)-224-8616
uvaydov.d@northeastern.edu

Milin Zhang
zhang.mil@northeastern.edu
