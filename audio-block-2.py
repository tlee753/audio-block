# imports
import pandas as pd
import os
import librosa
import librosa.display
import struct


# wave file visualizer
    # visualize the data

class waveFileVisualizer:


# wavefilehelper
    # get wave file data

class waveFileHelper():

    def readFileProperties(self, filename):

        waveFile = open(filename, "rb")

        riff = waveFile.read(12)
        fmt = waveFile.read(36)

        numChannels = struct.unpack('<H', fmt[10:12][0])
        sampleRate = struct.unpack('<H', fmt[12:16][0])
        bitDepth = struct.unpack('<H', fmt[22:24][0])

        waveFile.close()

        return (numChannels, sampleRate, bitDepth)

# create the model
    # training
    # weight compilation

# test the model