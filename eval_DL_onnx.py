#Author: Daniel Uvaydov
#Use to evaluate multilabel network on time series IQs given in binfile (like collected from GNU Radio)
#Network outputs label for each sub-band or each IQ (google semantic segmentation)


import numpy as np
import onnxruntime as rt
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch import nn

def aggregate(y_pred):

    all_pred = []
    for band in y_pred:
        start_idxs = np.arange(0,inp_dim*scale_fact-inp_dim+1, stride)
        idx_mat = np.array([np.arange(k,k+inp_dim) for k in start_idxs], dtype=int)
        y_pred_final = np.zeros([nclasses,inp_dim*scale_fact])
        for i in range(inp_dim*scale_fact):
            aggregate_idx = np.where(i == idx_mat)
            y_pred_final[:,i] = np.average(band[aggregate_idx[0],:,aggregate_idx[1]], axis=0)

        all_pred.append(y_pred_final)

    return np.array(all_pred)


def ms_estimate(tensor):
    p = tensor[:,0,:]**2+tensor[:,1,:]**2
    batch = p.shape[0]
    p = torch.from_numpy(p).float()
    p_smooth = nn.functional.conv1d(p.view(batch,1,-1),torch.ones(40).view(1,1,-1),stride=10)/40
    min_p,_ = torch.min(p_smooth,dim=2)
    return min_p.numpy()

def main():


    #Open IQ file and grab fist 1024*1024 IQ samples to be evaluated
    with open(iq_fp) as binfile:
        all_samps = np.fromfile(binfile, dtype=np.complex64, count=1024*1024, offset=0)

    assert (len(all_samps)%inp_dim == 0), "Input needs to be multiple of 1024"

    all_samps = np.reshape(all_samps, [-1, inp_dim*scale_fact])
    #Perform fft and fftshift on each sample
    all_samps_frequency = np.fft.fft(all_samps)
    all_samps_frequency = np.fft.fftshift(all_samps_frequency, axes=1)

    if scale_fact > 1:

        #break bandwidth into smaller overlapped 25 MHz bandwidths
        all_samps_frequency_strided = []
        for samp in all_samps_frequency:
            samp = np.array([samp[k:k + inp_dim] for k in range(0, len(samp) - inp_dim + 1, stride)])
            all_samps_frequency_strided.append(samp)
        all_samps_frequency_strided = np.array(all_samps_frequency_strided)
        nparts = all_samps_frequency_strided.shape[1]

        all_samps_frequency_strided = np.stack((np.real(all_samps_frequency_strided), np.imag(all_samps_frequency_strided)), axis=-1)
        all_samps_frequency_strided = np.reshape(all_samps_frequency_strided, (-1, 1024, 2))
        all_samps_frequency_strided = np.swapaxes(all_samps_frequency_strided,1,2)
        min_p_test = ms_estimate(all_samps_frequency_strided).reshape(-1,1,1)
        correction = min_p_test/0.0015 # 0.0015 is the estimated noise in training set
        all_samps_frequency_strided = all_samps_frequency_strided/np.sqrt(correction)

        y_pred = sess.run(None, {input_name: all_samps_frequency_strided.astype(np.float32)})[0]

        y_pred = np.swapaxes(y_pred, 1,2)
        y_pred = np.reshape(y_pred, [-1,nparts,nclasses,inp_dim])
        y_pred = aggregate(y_pred)
        y_pred = np.swapaxes(y_pred, 1,2)

    else:
        all_samps_frequency = np.stack((np.real(all_samps_frequency), np.imag(all_samps_frequency)), axis=-1)
        all_samps_frequency = np.reshape(all_samps_frequency, (-1, 1024, 2))
        all_samps_frequency = np.swapaxes(all_samps_frequency,1,2)
        min_p_test = ms_estimate(all_samps_frequency).reshape(-1,1,1)
        correction = min_p_test/0.0015 # 0.0015 is the estimated noise in training set
        all_samps_frequency = all_samps_frequency/np.sqrt(correction)

        y_pred = sess.run(None, {input_name: all_samps_frequency.astype(np.float32)})[0]

    y_pred[y_pred<=0.5]=0
    spec_occup = np.sum(y_pred,axis=1)
    spec_empty = np.expand_dims(np.logical_not(spec_occup,out=np.zeros_like(spec_occup)),axis=1)
    y_pred = np.concatenate((spec_empty,y_pred),axis=1)
    y_pred = np.argmax(y_pred,axis=1)

    a, b = np.meshgrid(np.linspace(0,1024*1024,1024),np.linspace(0,1024,1024))
    fig, ax = plt.subplots(1,2,sharey=True,figsize=(9,4))
    all_samps = np.reshape(all_samps, -1)
    spectrum,_,_,_ = plt.specgram(all_samps,noverlap=0,NFFT=1024)
    ax[0].imshow(np.abs(spectrum),cmap='viridis',norm=matplotlib.colors.LogNorm(vmin=np.min(np.abs(spectrum)),vmax=np.max(np.abs(spectrum))),aspect='auto',origin='lower')
    ax[0].set_title('Over-The-Air Spectrogram')
    ax[0].set_ylabel('Frequency [MHz]')
    ax[0].set_xlabel('Time [ms]')
    ax[0].set_xticks([],[])
    cs=ax[1].contourf(a,b,y_pred.T,levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] , cmap = 'Dark2')
    ax[1].set_title('Inference')
    ax[1].set_xlabel('Time [ms]')
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in cs.collections]
    plt.xticks([],[])
    plt.yticks([],[])
    plt.figlegend(proxy,["WiFi", "LTE", "Zigbee", "LoRa", "BLE"])
    plt.tight_layout()
    plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='GPU and Model Specifications')
    parser.add_argument('--id_gpu', default='0', type=str,
                        help='specify which gpu to use.')
    parser.add_argument('--samp_rate', default=25, type=int,
                        help='specifies the sampling rate in MHz')
    parser.add_argument('--input', default='./test_data.bin', type=str,
                        help='specifies the IQ samples bin file to be fed to the network')
    parser.add_argument('--model', default='./multilabel.onnx', type=str,
                        help='specifies the model filepath')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id_gpu

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
        }),
        'CPUExecutionProvider',
    ]

    onnx_fp = args.model
    sess = rt.InferenceSession(onnx_fp, providers=providers)
    input_name = sess.get_inputs()[0].name

    #############Constants#############

    #Number of classes (including empty channel)
    nclasses = 6
    #Bandwidth
    bw = args.samp_rate * 1e6
    #Scale factor from original 25MHz model was trained on
    scale_fact = int(bw//25e6)
    #Number of IQs network takes as input
    inp_dim = 1024
    #2 channels one for I and one for Q
    nchannels = 2
    #stride for sliding in mhz if BW is higher than 25MHz,
    stride_mhz = 12.5
    stride = int((inp_dim*scale_fact)*(stride_mhz/args.samp_rate))
    #Folder containing IQ dat file
    iq_fp = args.input
    #Model filepath
    model_fp = args.model


    main()
