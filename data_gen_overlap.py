import numpy as np
import scipy as sp
from scipy.signal import decimate
import os as os
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm



debug = False
json_reports = False

nsamples = 500_000
max_simultaneous_signals = 2
bw = 25_000_000
buf = 1024
resolution = (bw/1000000)/buf
prob_empty = 0.05
prob_centered = 0.5

nclasses = 5
label_dict_str= {
    'wifi'      : 0,
    'lte'       : 1,
    'zigbee'    : 2,
    'lora'      : 3,
    'ble'       : 4
}

label_dict_int = dict([(value, key) for key, value in label_dict_str.items()])


signal_bw_mhz_dict = {
    'wifi'      : 20,
    'lte'       : 10,
    'zigbee'    : 2,
    'lora'      : 0.5,
    'ble'       : 1
}

signal_bw_mhz_niqs = {}
for label in signal_bw_mhz_dict:
    niqs = np.ceil(signal_bw_mhz_dict[label]/resolution)

    if niqs%2 == 1:
        niqs += 1

    signal_bw_mhz_niqs[label] = int(niqs)

freq_arr = np.linspace(-(bw/1000000)/2, (bw/1000000)/2, buf)

h5_folder_fp = '/mnt/wines/iarpa/arena/da6000/processed/'
data_fp = h5_folder_fp + str(buf) + '/'

def get_sample(protocol):
    f_signal = h5py.File(data_fp + protocol + '.h5', 'r')
    samp = f_signal[protocol][np.random.randint(f_signal[protocol].shape[0])]
    f_signal.close()
    return samp


if __name__ == '__main__':

    all_labels = []
    all_inputs = []

    protocols_used = [0,1,2,3,4]
    for i in tqdm(range(nsamples)):
        label = np.zeros([nclasses,buf], dtype=int)
        input_samp = np.zeros([buf, 2])

        if np.random.rand() > prob_empty:
            nsignals = np.random.randint(1,max_simultaneous_signals+1)

            signal_protocols = np.zeros(nsignals)
            signal_freqs = np.zeros(nsignals)
            prev_centered = False

            for j in range(nsignals):
                protocol = np.random.choice(protocols_used)
                signal_protocols[j] = protocol

                signal_bw = signal_bw_mhz_niqs[label_dict_int[protocol]]
                temp_input = np.zeros([buf+signal_bw*2-2,2])

                samp = get_sample(label_dict_int[protocol])
                samp_cut = samp[buf//2-signal_bw//2:buf//2+signal_bw//2,:]

                if np.random.rand() < prob_centered and not prev_centered:
                    signal_location = int(len(temp_input)/2-signal_bw/2)
                    prev_centered = True
                else:
                    signal_location = int(np.random.choice(len(temp_input)-signal_bw+1))



                temp_mask = np.zeros(buf+signal_bw*2-2, dtype=bool)
                temp_mask[signal_location:signal_location + signal_bw] = True

                signal_freqs[j] = signal_location*resolution

                temp_input[signal_location:signal_location+signal_bw,:] += samp_cut
                label[protocol] |= temp_mask[signal_bw-1:signal_bw+buf-1]
                input_samp += temp_input[signal_bw-1:signal_bw+buf-1,:]


        input_samp += get_sample('empty')
        all_inputs.append(input_samp)
        all_labels.append(label)


        if debug:
            plt.plot(np.linspace(-12.5,12.5,buf),abs(input_samp[:,0] + 1j*input_samp[:,1]))
            plt.xlabel('MHz')
            plt.title(signal_protocols)
            plt.show()

    all_inputs = np.array(all_inputs)
    all_labels = np.array(all_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        all_inputs, all_labels, test_size=0.1, random_state=42)

    f_test = h5py.File(
        h5_folder_fp + 'overlap_1024_25mhz_3days_test_2sig.h5', 'w')
    xtest = f_test.create_dataset('X', (X_test.shape[0], X_test.shape[1], X_test.shape[2]), dtype='f')
    ytest = f_test.create_dataset('y', (y_test.shape[0],y_test.shape[1], y_test.shape[2]), dtype='i')

    xtest[()] = X_test
    ytest[()] = y_test

    f_train = h5py.File(
        h5_folder_fp + 'overlap_1024_25mhz_3days_train_2sig.h5', 'w')
    xtrain = f_train.create_dataset('X', (X_train.shape[0], X_train.shape[1], X_train.shape[2]), dtype='f')
    ytrain = f_train.create_dataset('y', (y_train.shape[0],y_train.shape[1], y_train.shape[2]), dtype='i')

    xtrain[()] = X_train
    ytrain[()] = y_train

    f_test.close()
    f_train.close()