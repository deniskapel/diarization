import sys
import torch
import numpy as np
# SCD
from pyannote.audio.utils.signal import Peak

def main(args):
    if len(args) != 1:
        sys.stderr.write(
            'Usage: speaker_change_detection.py <path to wav file>\n')
        sys.exit(1)
    # speaker change detection SCD model trained on AMI training set
    scd = torch.hub.load('pyannote/pyannote-audio', 'scd_ami')
    lesson = {'audio': args[0]}

    # Speaker change detection
    # obtain raw SCD scores (as `pyannote.core.SlidingWindowFeature` instance)
    scd_scores = scd(lesson)
    # detect peaks and return speaker homogeneous segments
    # NOTE: both alpha/min_duration values were tuned on AMI dataset.
    # might need to use different values for better results.
    peak = Peak(alpha=0.10, min_duration=0.1, log_scale=True)
    # speaker change point (as `pyannote.core.Timeline` instance)
    partition = peak.apply(scd_scores, dimension=1)

    # .uem files are easily transformed into pandas dataframe
    with open('stat/partition.uem', 'w') as f:
        partition.write_uem(f)

if __name__ == '__main__':
    main(sys.argv[1:])
