# Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching


This repo contains the code used in the paper: [D. Uvaydov, M. Zhang, C. Robinson, S. D'Oro, T. Melodia and F. Restuccia, "Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching," Proc. of IEEE Intl. Conf. on Computer Communications (INFOCOM), Vancouver, BC, Canada, May 2024]. 
We utilize a semi-augmented over-the-air data generator to create our diverse dataset.
Our model is an altered 1D version of the famous UNet model with added multi-label multi-class functionality and non-local attention mechanism. 
This model generates a label for every single IQ/sub-band (in frequency domain). The network takes as input 1024 IQs in frequency domain and outputs a class for each. This means that 25MHz is brokenvinto 25MHz/1024 = 0.0244MHz sub-bands to be classified. 

The data used in this paper can be found [here]. We provide both the raw data we used to create the signal bank and the signal bank itself. The raw data was collected with a 25MHz bandwidth/sampling frequency for our experimental use case. Our signal bank is only 17GB but can be used with the data generator to create a quasi-infinite sized dataset. 

The data generator below can be repurposed for other wireless applications (classifying things other than what was classified in the paper
e.g. MCS, RF ID, etc.). To do so one would need to create their own signal bank which means collecting their own mini raw dataset, remember the point of this work
is to alleviate the difficulty of this step not fully eradicate it. To do so find a small, ideally empty bandwdith (we used 25 MHz), and collect your desired signals, this does not need to be a large data collection campaign as the dataset generator will take your raw data and turn it into a large diverse wideband dataset, more information on this process can be found in the paper. Once collecting raw data then one would need to create a signal bank with their own version of ```bin2h5.py```. This file takes raw IQ files grabbed from GNU Radio and pre-processes them
in a manner laid out in the paper. It's outputs are one .h5 file per class (e.g. wifi.h5, lte.h5, etc.) containing a 
matrix with the signals of shape ```(number of samples, number of IQs, 2)```. After a user specific signal bank is generated then changes would need to be made
to the project specific parameters in ```data_gen_overlap.py``` (everything defined outside of the main() function) of the data generator.


## Environment:

To generate the signal bank, use the data generator, or train the multilabel model with self-attention used in the paper import this environment from yml:

    conda env create -n multilabel --file multilabel.yml
    conda activate multilabel

To evaluate onnx version of model (version we use for real-time inference) import this environment from yml:

    conda env create -n eval_DL --file eval_DL.yml
    conda activate eval_DL

## Code:

```bin2h5.py``` - Takes raw IQ bin files collected with USRPs and prepares them for a signal bank to be used by data generators. 
This code is very specific to our paper, if you would like to use our data generator for your own project, you would need
to make your own version of this file. The only thing the data generator expects is that the samples for each class is 
contained in a single h5 file (e.g. wifi.h5, lte.h5, etc), and each h5 file contains a matrix of shape ```(number of samples, number of IQs, 2)```, 
where "2" here is for the real and imaginary channels.


```data_gen_overlap.py``` - Dataset generator where signals overlap, signal labels are 2D matrices where each row is a
different class in a multi-label multi-class fashion. The dataset resulting from this code is used to train the 
multilabel model with self-attention. Highly suggest looking into code
and making changes to project specific hyper-parameters as needed for personal use.


```multilabel.py``` - Use this file to train the customized unet multilabel model with self attention
```bash
usage: multilabel.py [-h] [-ts] [-vs] [-d]

options:
  -h, --help         show this help message and exit
  -ts , --TrainSet   filepath of training set (default:
                     ./train.h5)
  -vs , --ValSet     filepath of validation set (default:
                     ./test.h5)
  -d , --Device      specify the device for running model (default: -1)
  
example: python multilabel.py -ts train.h5 -vs test.h5 -d 0
```

```eval_DL_onnx.py``` - Use this file to run the model on raw IQs. The code expects the IQs in the form of a binary file, similiar to the output file that GNU Radio generates from a file sink, specifically the binary file contains a series of two 32-bit floating point numbers, one each for the real and imaginary components.

    usage: eval_DL_onnx.py [-h] [--id_gpu ID_GPU] [--samp_rate SAMP_RATE] [--input INPUT] [--model MODEL] [--normalize NORMALIZE]

    GPU and Model Specifications

    optional arguments:
      -h, --help                show this help message and exit
      --id_gpu ID_GPU           specify which gpu to use.
      --samp_rate SAMP_RATE     specifies the sampling rate in MHz, for now must be multiple of 25 MHz
      --input INPUT             specifies the IQ samples bin file to be fed to the network
      --model MODEL             specifies the model filepath

    example: python eval_DL_onnx.py --id_gpu 0 --samp_rate 25 --input ./test_data.bin --model ./multilabel.onnx

## Citation

D. Uvaydov, M. Zhang, C. Robinson, S. D'Oro, T. Melodia and F. Restuccia, "Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching," Proc. of IEEE Intl. Conf. on Computer Communications (INFOCOM), Vancouver, BC, Canada, May 2024

```sh
@inproceedings{uvaydov2024infocom,
  author = {Uvaydov, Daniel and Zhang, Milin and Robinson, Clifton P and D'Oro, Salvatore and Melodia, Tommaso and Restuccia, Francesco},
  booktitle = {{IEEE INFOCOM 2024 - IEEE Conference on Computer Communications}},
  title = {{Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching}},
  year = {2024},
  month = {May}
}
```

### Contact info:

Daniel Uvaydov
uvaydov.d@northeastern.edu

Milin Zhang
zhang.mil@northeastern.edu



[//]: # 

   [D. Uvaydov, M. Zhang, C. Robinson, S. D'Oro, T. Melodia and F. Restuccia, "Stitching the Spectrum: Semantic Spectrum Segmentation with Wideband Signal Stitching," Proc. of IEEE Intl. Conf. on Computer Communications (INFOCOM), Vancouver, BC, Canada, May 2024]: <https://arxiv.org/abs/2402.03465>
   [here]: <https://nam12.safelinks.protection.outlook.com/?url=http%3A%2F%2Fhdl.handle.net%2F2047%2FD20661303&data=05%7C02%7Cuvaydov.d%40northeastern.edu%7Ca3205ef7f0de454c4cd408dc7a6eb12a%7Ca8eec281aaa34daeac9b9a398b9215e7%7C0%7C0%7C638519863048001040%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=G7pCaHZaYYA5qgBknNE2q%2BFsjeyW%2FHepr4LA36p6roc%3D&reserved=0>

   
