# -*- coding: utf-8 -*-
"""

"""
from scipy import signal
import numpy as np
from PIL import Image
from scipy.io import wavfile

fs, data = wavfile.read('./data/dataset1/wav/sample_0.wav')
f, t, Sxx = signal.spectrogram(data, fs, noverlap=0 ,scaling='spectrum',nperseg=345,nfft=51200)
newim = np.zeros(shape=[512,512])
for x in range(Sxx.shape[1]-1):
    size=7.82
    i=0
    y=0
    while i < Sxx.shape[0]:
        if y%127 is 0:
            size = size*2
        val = Sxx[int(i)][x]
        for c in range(int(size)-1):
            if Sxx[int(i)-c][x] > val:
                val =  Sxx[int(i)-c][x]
        newim[511-y][x] = np.sqrt(val)
        y+=1
        i+=size

im = Image.fromarray(newim*255/np.max(newim)).convert("L")
im.save('2.png')