import sys
import collections
import numpy as np
from librosa import estimate_tuning
from librosa.onset import onset_strength
from librosa.beat import tempo
from librosa.util import buf_to_float
from pydub import AudioSegment
from io import BytesIO
from json import loads
from vosk import Model, KaldiRecognizer, SetLogLevel
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from collections import Counter
from Vocabulary import Vocabulary
from utils import tokenizer, count_vocab

class LessonSegment(Vocabulary):

    """docstring for LessonSegment."""
    def __init__(self, target_vocabulary, bytes):
        super().__init__(target_vocabulary)
        self.bytes = bytes
        self.transcript = []
        self.statistics = {}

    def transcribe(self, recognizer):
        """
        use kaldi asr model to transcribe pcm_data
        for model structure check https://alphacephei.com/vosk/models.html
        input:
            pcm_data,
            instance of KaldiRecognizer,
            instance of nltk PorterStemmer
        output:
            [list of single-word Strings]
        """
        # ASR
        recognizer.AcceptWaveform(self.bytes)
        utterance = loads(recognizer.Result())['text']
        # tokenize into a list of 'words'
        self.transcript.extend(tokenizer(utterance))

    def get_features(self, sr):
        """
        calculates tempo and pitch using librosa
        documentation https://librosa.github.io/librosa/
        """
        timeseries = buf_to_float(self.bytes)
        pitch = estimate_tuning(timeseries, sr)
        # onset_env = onset_strength(timeseries, sr)
        # temp = tempo(onset_env, sr)[0]
        return([pitch])

    def get_staistics(self, dictionary):
        """count occurances of target vocabulary within self.transcript"""
        # if there are no entries in the dictionary, stop
        if len(dictionary)==0:
            return

        self.statistics.update(
            {key: count_vocab(value, self.transcript) for key, value in dictionary.items()})
