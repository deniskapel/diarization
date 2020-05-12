import sys
import os
import re
import pandas as pd
import numpy as np
import librosa
from pydub import AudioSegment
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def calc_pitch_and_tempo(segment, sr):
    """takes an audio segment and sampling rate
    returns pitch and static tempo
    """
    timeseries = np.asarray(segment.get_array_of_samples()).astype(float)
    pitch = librosa.estimate_tuning(timeseries, sr)
    onset_env = librosa.onset.onset_strength(timeseries, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    return pitch, tempo[0]

def main(args):
    if len(args) != 1:
        sys.stderr.write(
            'Usage: partition.py <num of clusters>\n')
        sys.exit(1)

    path = os.getcwd() + '\\chunks\\'
    segments = []

    for file in os.listdir(path):
        if re.match(r"chunk-\d+\.wav", file):
            segments.append(AudioSegment.from_wav(path + file))

    sr = segments[0].frame_rate

    df = pd.DataFrame(index=range(len(segments)),
                      columns=['duration', 'loudness', 'pitch', 'tempo'])

    for i, segment in enumerate(segments):
        df['duration'][i] = segment.duration_seconds
        df['loudness'][i] = segment.dBFS
        df['pitch'][i], df['tempo'][i] = calc_pitch_and_tempo(segment, sr)

    norm = df[['loudness', 'pitch', 'tempo']].to_numpy()
    df[['loudness', 'pitch', 'tempo']] = MinMaxScaler().fit_transform(norm)


    # df = df[df.duration >= 1]
    #
    # # model = KMeans(n_clusters=int(args[0]),
    # #                     random_state=42).fit(df[['loudness']].to_numpy())
    #
    model = GaussianMixture(n_components=int(args[0]),
                            covariance_type='full')
    clusters = model.fit_predict(df[['loudness', 'pitch', 'tempo']].to_numpy())

    # # VISUALIZATION
    # df = pd.merge(df,
    #               pd.DataFrame({'cluster': clusters}),
    #               how='left',
    #               left_index=True, right_index=True)
    #
    # loud_stat = []
    # pitch_stat = []
    # tempo_stat = []
    # fig=plt.figure()
    # ax=fig.add_axes([0,0,1,1])
    #
    # for i in range(int(args[0])):
    #     loud_stat.append(df[df.cluster == i]['loudness'].to_numpy())
    #     pitch_stat.append(df[df.cluster == i]['pitch'].to_numpy())
    #     ax.scatter(loud_stat[i], pitch_stat[i], alpha=0.5)
    #
    # ax.set_xlabel('Loudness')
    # ax.set_ylabel('Pitch')
    # ax.set_title('PITCH / LOUDNESS')
    # plt.show()
    # # END VISUALIZATION

    # RESEGMENTATION
    chunks = []
    for i in range(int(args[0])):
      chunks.append(AudioSegment.empty())

    for i, cluster in enumerate(clusters):
      chunks[int(cluster)] += segments[i]

    for i, chunk in enumerate(chunks):
      path = 'resegmentation/cluster-%002d.wav' % (i,)
      print(' Writing %s' % (path,))
      chunk.export(path, format='wav')


if __name__ == '__main__':
    main(sys.argv[1:])
