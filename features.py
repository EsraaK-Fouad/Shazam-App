import os
from  HashTable import  HashTable
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import (generate_binary_structure, iterate_structure, binary_erosion)
from operator import itemgetter
import hashlib
from scipy.io import wavfile
from HashTable import HashTable
from pydub.utils import make_chunks
from pydub import AudioSegment
import difflib
# **********************************************************code implemented****************************************#
DEFAULT_FS = 44100
DEFAULT_WINDOW_SIZE = 4096
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_AMP_MIN = 10
PEAK_NEIGHBORHOOD_SIZE = 20

IDX_FREQ_I = 0
IDX_TIME_J = 1
DEFAULT_FAN_VALUE = 15
MIN_HASH_TIME_DELTA = 0
MAX_HASH_TIME_DELTA = 200
PEAK_SORT = True
FINGERPRINT_REDUCTION = 20
PEAK_NEIGHBORHOOD_SIZE = 20
# *********************************************Extract main features Func*****************************************
def get_2D_peaks(arr2D, plot=False, amp_min=DEFAULT_AMP_MIN):
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)
    local_max = maximum_filter(arr2D, footprint=neighborhood) == arr2D
    background = (arr2D == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    amps = arr2D[detected_peaks]
    j, i = np.where(detected_peaks)
    amps = amps.flatten()
    peaks = zip(i, j, amps)
    peaks_filtered = [x for x in peaks if x[2] > 10]
    frequency_idx = [x[1] for x in peaks_filtered]
    time_idx = [x[0] for x in peaks_filtered]
    # scatter of the peaks
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(arr2D)
        ax.scatter(time_idx, frequency_idx)
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.set_title("Spectrogram")
        plt.gca().invert_yaxis()
        plt.show()
    return zip(frequency_idx, time_idx)
# *******************************************************************************
def generate_hashes(peaks, fan_value=DEFAULT_FAN_VALUE):
    if PEAK_SORT:
        peaks = sorted(peaks, key=itemgetter(1))
    store = list()
    for i in range(len(peaks)):
        for j in range(1, fan_value):
            if (i + j) < len(peaks):
                freq1 = peaks[i][IDX_FREQ_I]
                freq2 = peaks[i + j][IDX_FREQ_I]
                t1 = peaks[i][IDX_TIME_J]
                t2 = peaks[i + j][IDX_TIME_J]
                t_delta = t2 - t1
                if t_delta >= MIN_HASH_TIME_DELTA and t_delta <= MAX_HASH_TIME_DELTA:
                    h = hashlib.sha1(("%s|%s|%s" % (str(freq1), str(freq2), str(t_delta))).encode('utf-8'))
                    hashvalue = [h.hexdigest()[0:FINGERPRINT_REDUCTION], t1]
                    store.append(hashvalue)
    return store
#************************************* to find the smaller lenghth of all songs **********************************
lenArr = []
for filename in os.listdir("dsp_4/songs"):
    song = f'dsp_4/songs/{filename}'
    wav = AudioSegment.from_mp3(song)
    wav.export("wavfile.wav", format="wav")
    sample_rate, Data = wavfile.read("wavfile.wav")
    x = len(Data)
    lenArr.append(x)
X = lenArr[0]
for i in range(len(lenArr)):
    y = lenArr[i]
    if (X > y):
        X = y


# ********************************************************************************************************************
def fingerprint_Mixed_Song(songname1, songname2, slidervalue):
    wav_audio1 = AudioSegment.from_mp3(songname1)
    wav_audio1.export("wavfile1.wav", format="wav")
    sr1, y1 = wavfile.read("wavfile1.wav")
    wav_audio = AudioSegment.from_mp3(songname2)
    wav_audio.export("wavfile2.wav", format="wav")
    sr2, y2 = wavfile.read("wavfile2.wav")

    y = (0.1 * slidervalue * y1[:int(X)]) + ((1 - 0.1 * slidervalue) * y2[:int(X)])

    chunks = make_chunks(y[:, 1], 100000)
    arr2D = mlab.specgram(chunks[0], NFFT=DEFAULT_WINDOW_SIZE, Fs=44100, window=mlab.window_hanning,
                          noverlap=int(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO))[0]
    local_maxima = get_2D_peaks(arr2D, plot=True, amp_min=10)
    hash = generate_hashes(local_maxima, fan_value=15)
    hash = np.asarray(hash).T
    return hash


