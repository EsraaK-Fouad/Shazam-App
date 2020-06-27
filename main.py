
import os
from PyQt5 import QtWidgets
import cv2
from PyQt5.QtWidgets import QFileDialog
from PIL import Image
from PyQt5.uic.properties import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction
from PyQt5.QtGui import QPixmap, QIcon, QFont
from design import *
from features import *
import sys
import pyqtgraph as pg
import numpy as np
import csv
from matplotlib.backends.backend_template import FigureCanvas
from pydub import AudioSegment
import librosa
import librosa.display
import imagehash
import matplotlib.pyplot as plt
#***********************************************************************************************************************
class App(Ui_MainWindow):
    def __init__(self,window):
        self.Hash = HashTable()
        self.Hasharr = []
        self.phashratio = list()
        self.FeatureRatio = list()
        self.HashSmilarityArr=[]
        self.TableArr = []
        self.songArr = []
        self.avgArr = []
        self.setupUi(window)
        self.pushButton_2.clicked.connect(self.OpenFile1)
        self.pushButton.clicked.connect(self.OpenFile2)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(10)
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.setSingleStep(1)
        self.pushButton_3.clicked.connect(self.searchfunc)
        self.pushButton_4.clicked.connect(self.clear)
        self.batch()
        self.Phashing()
        self.fingerprint()
        for row in range(10):
            self.tableWidget.insertRow(0) 
        for col in range(4):
            self.tableWidget.insertColumn(0)
#**************************************************OPEN FUNC ***********************************************************
    def OpenFile1(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName()
        try:
            self.path1 = filepath[0]
        except :
            pass
        

    def OpenFile2(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName()
        
        try:
            self.path2 = filepath[0]
        except :
            pass
# ******************************* BATCH FUNCTION (make spectrograms for songs )***********
    def batch(self):
        for filename in os.listdir("dsp_4/songs"):
            songname = f'dsp_4/songs/{filename}'
            y, sr = librosa.load(songname, sr=None, duration=int(X/sample_rate))
            hop_length = 512
            window_size = 1024
            window = np.hanning(window_size)
            out = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=hop_length,
                                             window=window)
            out = 2 * np.abs(out) / np.sum(window)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
            fig.savefig(f'dsp_4/spectro/{filename[:-3].replace(".", "")}.png')
            plt.clf()
            self.songArr.append(filename)
# *************************************** PHash function (return array of the phash of all sectograms)******************
    def Phashing(self):
        for figure in os.listdir("dsp_4/spectro"):
            figurename = f'dsp_4/spectro/{figure}'
            figurename = Image.open(figurename)
            phash = imagehash.phash(figurename)
            self.Hasharr.append(phash)
# ************************************MixedSongPhash func(return the phash of mixed song))***************************************
    def MixedSongPhash (self):
        data1, fs = librosa.load(self.path1, sr=None, duration=int(X/sample_rate))
        data2, fs2 = librosa.load(self.path2, sr=None, duration=int(X/sample_rate))
        data = (0.1*self.horizontalSlider.value()* data1) + ((1-0.1*self.horizontalSlider.value())*data2)
        hop_length = 512
        window_size = 1024
        window = np.hanning(window_size)
        out = librosa.core.spectrum.stft(data, n_fft=window_size, hop_length=hop_length,
                                         window=window)
        out = 2 * np.abs(out) / np.sum(window)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
        fig.savefig('input.png')
        MixedSongPhash= imagehash.phash(Image.open('input.png'))
        return MixedSongPhash
#**************************************FEATURE EXTRACTION ************************************************************
    def fingerprint(self):
        for filename in os.listdir("dsp_4/songs"):
            songname = f'dsp_4/songs/{filename}'
            wav_audio = AudioSegment.from_mp3(songname)
            wav_audio.export("wavfile.wav", format="wav")
            sr, data = wavfile.read("wavfile.wav")
            y = data[:int(X)]
            chunks = make_chunks(y[:, 1], 100000)
            arr2D = mlab.specgram(chunks[0], NFFT=DEFAULT_WINDOW_SIZE, Fs=44100, window=mlab.window_hanning,
                                  noverlap=int(DEFAULT_WINDOW_SIZE * DEFAULT_OVERLAP_RATIO))[0]
            local_maxima = get_2D_peaks(arr2D, plot=False, amp_min=10)
            hash = generate_hashes(local_maxima, fan_value=15)
            hash = np.asarray(hash).T
            self.Hash.insert(filename, hash)
#*********************************** MixedSong_Featurefunc ***********************************************************
    def  MixedSong_Featurefunc(self):
        input_song_feature_hash = fingerprint_Mixed_Song(self.path1,self.path2,self.horizontalSlider.value())
        for k in self.songArr:
            sm = difflib.SequenceMatcher(None, str(input_song_feature_hash), str(self.Hash.find(k)))
            self.FeatureRatio.append(sm.ratio())
#******************************MaxSimilarity FUNC (return max similarity value and name of this song)*****************
    def maxsimilarity (self):
        MixedSongHash = self.MixedSongPhash()
        self.MixedSong_Featurefunc()
        for i in range(len(self.songArr)):
            SM = difflib.SequenceMatcher(None, str(MixedSongHash ), str(self.Hasharr[i]))
            self.phashratio.append(SM.ratio())
        for i in range(len(self.phashratio)):
            avrage = (self.phashratio[i] + self.FeatureRatio[i])*0.5
            self.avgArr.append(avrage)

        self.TableArr.append(['Song_Name', 'Avrage', 'Phash_Similarity','Feature_Similarity'])
        self.maxsimilarity_value = self.avgArr[0]
        self.song_with_maxsimilarity = self.songArr[0]
        for i in range(len(self.Hasharr)):
            self.TableArr.append([self.songArr[i], str(self.avgArr[i]*100),str(self.phashratio[i]*100),str(self.FeatureRatio[i]*100)])
            if (self.maxsimilarity_value < self.avgArr[i]):
                self.maxsimilarity_value = self.avgArr[i]
                self.song_with_maxsimilarity = self.songArr[i]
            else:
                pass
        self.progressBar.setValue(self.maxsimilarity_value * 100)
        self.label.setText(self.song_with_maxsimilarity)
#***************************************SEARCH FUNCTION (display the TableArr throgh th GUI)****************************
    def searchfunc (self):
        print("begin")
        self.maxsimilarity()
        for rowNr, rowValue in enumerate(self.TableArr):
            for itemNr, itemValue in enumerate(rowValue):
                self.tableWidget.setItem(rowNr, itemNr, QtWidgets.QTableWidgetItem(self.TableArr[rowNr][itemNr]))
        print('done')

#***********************************************************************************************************************
    def clear(self):
        for rowNr, rowValue in enumerate(self.TableArr):
            for itemNr, itemValue in enumerate(rowValue):
                self.tableWidget.clear()
        self.TableArr.clear()
        self.avgArr.clear()
        self.phashratio.clear()
        self.FeatureRatio.clear()
        self.progressBar.setValue(0)
        self.label.setText('')
#***********************************************************************************************************************
def main():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = App(MainWindow)
    MainWindow .show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())

