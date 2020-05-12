from python_speech_features import mfcc
from sklearn.cluster import KMeans
import numpy as np
from pydub import AudioSegment

# Input
# param: path String
# param: audiocodec String
#
# Output:
# Turple of nympy arrays (mfcc for left channel, mfcc for right channel)
def get_mfcc(lesson, duration):
    """
    Input
    param: pydub AudioSegment
    param: audiocodec String

    Output:
    np array -1 x 13
    """
    frame_duration = duration*100
    rate = lesson.frame_rate
    nfft = round(rate / 40 + 1)
    channel = np.asarray(lesson.get_array_of_samples())
    mfccs = mfcc(channel, rate, nfft=nfft)
    # rchannel_mfcc = mfcc(rchannel, rate, nfft=nfft, lowfreq=200, highfreq=8000, winfunc=np.hamming)

    # omit the last partial second
    return mfccs[:frame_duration]
