import numpy as np
import scipy as sp
from scipy.signal import decimate
import os as os
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from generate_json import generate_json




debug = False
json_reports = False

nsamples = 1_000_000
max_simultaneous_signals = 2
bw = 25_000_000
buf = 256
resolution = (bw/1000000)/buf
prob_empty = 0.05
prob_centered = 0.5  #probablility of signal being centered

nclasses = 6
label_dict_str= {
    'empty'     : 0,
    'wifi'      : 1,
    'lte'       : 2,
    'zigbee'    : 3,
    'lora'      : 4,
    'ble'       : 5
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

    protocols_used = [1,2,3,4,5]
    for i in tqdm(range(nsamples)):
        label = np.zeros(buf, dtype=int)
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
                while np.any(temp_mask[signal_bw-1:signal_bw+buf-1] & (label>0)):
                    signal_location = int(np.random.choice(len(temp_input)-signal_bw+1))
                    temp_mask = np.zeros(buf + signal_bw * 2 - 2, dtype=bool)
                    temp_mask[signal_location:signal_location + signal_bw] = 1

                signal_freqs[j] = signal_location*resolution
                temp_input[signal_location:signal_location+signal_bw,:] += samp_cut
                label += temp_mask[signal_bw-1:signal_bw+buf-1]*protocol
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

    if json_reports:
        signal_name = 'NEU_' + label_dict_int[protocols_used[0]] + '_1'
        signal_dir = h5_folder_fp + 'airanaculus/' + signal_name + '/'

        if not os.path.isdir(signal_dir):
            os.mkdir(signal_dir)

        sig_report = generate_json(signal_name, bw, buf, all_labels.shape[0], label_dict_int, all_labels, freq_arr)
        sig_report_json = json.dumps(sig_report, indent=4)
        with open(signal_dir + 'signal.meta.json', "w") as outfile:
            outfile.write(sig_report_json)


    else:
        X_train, X_test, y_train, y_test = train_test_split(
            all_inputs, all_labels, test_size=0.1, random_state=42)

        f_test = h5py.File(
            h5_folder_fp + 'iarpa_1024_25mhz_3days_test_2.h5', 'w')
        xtest = f_test.create_dataset('X', (X_test.shape[0], X_test.shape[1], X_test.shape[2]), dtype='f')
        ytest = f_test.create_dataset('y', (y_test.shape[0],y_test.shape[1]), dtype='i')

        xtest[()] = X_test
        ytest[()] = y_test

        f_train = h5py.File(
            h5_folder_fp + 'iarpa_1024_25mhz_3days_train_2.h5', 'w')
        xtrain = f_train.create_dataset('X', (X_train.shape[0], X_train.shape[1], X_train.shape[2]), dtype='f')
        ytrain = f_train.create_dataset('y', (y_train.shape[0],y_train.shape[1]), dtype='i')

        xtrain[()] = X_train
        ytrain[()] = y_train

        f_test.close()
        f_train.close()

    print('end')
