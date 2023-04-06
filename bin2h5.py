import numpy as np
import scipy as sp
from scipy.signal import decimate
import os as os
import h5py
import matplotlib.pyplot as plt
from scipy import signal
from time import sleep

label_dict = {
    'empty'     : 0,
    'wifi'      : 1,
    'lte'       : 2,
    'zigbee'    : 3,
    'lora'      : 4,
    'ble'       : 5
}


buf = 1024
resolution = 25/buf

signal_bw_mhz_dict = {
    'wifi'      : 20,
    'lte'       : 10,
    'zigbee'    : 2,
    'lora'      : 0.125,
    'ble'       : 1
}

signal_bw_mhz_niqs = {}

for label in signal_bw_mhz_dict:
    niqs = np.ceil(signal_bw_mhz_dict[label]/resolution)

    if niqs%2 == 1:
        niqs += 1

    signal_bw_mhz_niqs[label] = int(niqs)



folder_fp = '/mnt/wines/iarpa/arena/da6000/raw/'
dir = os.listdir(folder_fp)
h5_folder_fp = '/mnt/wines/iarpa/arena/da6000/processed/'





data_fp = h5_folder_fp + str(buf) + '_unfilt/'

if not os.path.isdir(data_fp):  
    os.mkdir(data_fp)


stride = buf
offset = 0
plot = 0
debug = 0

#lora = 300
#empty = 0.006
#wifi = 0.03
#zigbee = 0.13
#ble = 0.001
#lte = 0.02
threshold = 0.009

for label in label_dict:


    contains_signal = []
    contains_empty = []

    contains_signal_time = []
    contains_empty_time = []

    if label is not 'empty':
        for file in dir:

            if os.path.isfile(os.path.join(folder_fp,file)) and label in file and 'empty' not in file:
                if debug:
                    plt.ion()
                    fig = plt.figure()
                    ax = fig.add_subplot(211)
                    line, = ax.plot(np.zeros(buf))

                with open(os.path.join(folder_fp,file)) as binfile:
                    print(os.path.join(folder_fp,file))
                    all_samps = np.fromfile(binfile, dtype=sp.complex64, count=-1, offset=0)
                    all_samps = np.array([all_samps[k:k + buf] for k in range(0, len(all_samps) - 1 - buf, stride)])

                    mask = np.ones(signal_bw_mhz_niqs[label], dtype=int)
                    mask_empty = np.zeros(buf-signal_bw_mhz_niqs[label], dtype=int)
                    final_mask = np.insert(mask_empty, len(mask_empty)//2, mask)


                    for samp_idx, samps in enumerate(all_samps):
                        sample_f = np.fft.fft(samps)
                        sample_f = np.fft.fftshift(sample_f)
                        # sample_f[np.where(final_mask == 0)[0]] = 0
                        power = np.abs(sample_f)**2
                        energy = np.sum(power)/len(power)

                        if energy > threshold:
                            contains_signal.append(np.transpose(np.stack((np.real(sample_f), np.imag(sample_f)))))
                            if plot:
                                contains_signal_time.append(np.transpose(np.stack((np.real(samps), np.imag(samps)))))
                        else:
                            contains_empty.append(np.transpose(np.stack((np.real(sample_f), np.imag(sample_f)))))
                            if plot:
                                contains_empty_time.append(np.transpose(np.stack((np.real(samps), np.imag(samps)))))


                        # samps = np.transpose(np.stack((np.real(samps), np.imag(samps))))
                        # samps = np.array([samps[k:k + buf] for k in range(0, len(samps) - 1 - buf, stride)])
                        # print("Number of Samples: " + str(samps.shape[0]))
                        #
                        # name = os.path.splitext(file)[0] + '_' + str(downsample_freq//1000) + 'k'

                        # f = h5py.File(data_fp + name + '.h5', 'w')
                        #
                        # dset = f.create_dataset(name, (samps.shape[0], samps.shape[1], samps.shape[2]), dtype='f')
                        #
                        # dset[()] = samps
                        #
                        # f.close()

                        if debug:
                            line.set_ydata(power)
                            plt.title(file + ': ' + str(energy) + ' ' + str(samp_idx) + '/' + str(len(all_samps)))
                            plt.ylim([0, 10])
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            sleep(0.01)

        contains_signal = np.array(contains_signal)
        contains_empty = np.array(contains_empty)

        print("Number of Signal Samples: " + str(contains_signal.shape[0]))
        print("Number of Empty Samples: " + str(contains_empty.shape[0]))

        f_signal = h5py.File(data_fp + label + '.h5', 'w')
        dset = f_signal.create_dataset(label, (contains_signal.shape[0], contains_signal.shape[1], contains_signal.shape[2]), dtype='f')
        dset[()] = contains_signal
        f_signal.close()

        if plot:
            contains_signal_time = np.array(contains_signal_time)
            contains_empty_time = np.array(contains_empty_time)

            contains_signal_iq = contains_signal_time[:,:,0] + 1j* contains_signal_time[:,:,1]

            f_signal, t_signal, Sxx_signal = signal.spectrogram(contains_signal_iq.flatten()[:int(len(contains_signal_iq.flatten()) * 0.1)], 25_000_000, return_onesided=False, nperseg=buf, noverlap=0)
            Sxx_signal = np.fft.fftshift(Sxx_signal, axes=0)
            plt.pcolormesh(t_signal, np.fft.fftshift(f_signal), Sxx_signal, shading='auto', vmax=np.max(Sxx_signal)/100)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Signal ' + label)
            plt.show()

            contains_empty_iq = contains_empty_time[:,:,0] + 1j* contains_empty_time[:,:,1]

            f_empty, t_empty, Sxx_empty = signal.spectrogram(contains_empty_iq.flatten()[:int(len(contains_empty_iq.flatten()) * 0.1)], 25_000_000, return_onesided=False, nperseg=buf, noverlap=0)
            Sxx_empty = np.fft.fftshift(Sxx_empty, axes=0)
            plt.pcolormesh(t_empty, np.fft.fftshift(f_empty), Sxx_empty, shading='auto', vmax=np.max(Sxx_signal)/100)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Empty ' + label)
            plt.show()
        print('end')

    else:

        for file in dir:
            if os.path.isfile(os.path.join(folder_fp, file)) and label in file and 'empty' in file:

                with open(os.path.join(folder_fp, file)) as binfile:
                    print(os.path.join(folder_fp, file))
                    all_samps = np.fromfile(binfile, dtype=sp.complex64, count=-1, offset=0)
                    all_samps = np.array([all_samps[k:k + buf] for k in range(0, len(all_samps) - 1 - buf, stride)])
                    for samp_idx, samps in enumerate(all_samps):
                        sample_f = np.fft.fft(samps)
                        sample_f = np.fft.fftshift(sample_f)

                        contains_empty.append(np.transpose(np.stack((np.real(sample_f), np.imag(sample_f)))))

        contains_empty = np.array(contains_empty)

        f_signal = h5py.File(data_fp + label + '.h5', 'w')
        dset = f_signal.create_dataset(label, (contains_empty.shape[0], contains_empty.shape[1], contains_empty.shape[2]), dtype='f')
        dset[()] = contains_empty
        f_signal.close()