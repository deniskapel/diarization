import pandas as pd
import numpy as np
import sys
import librosa
from pydub import AudioSegment
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main(args):
    if len(args) != 3:
        sys.stderr.write(
            'Usage: partition.py <path to wav file> <path to partition file> <num of clusters>\n')
        sys.exit(1)

    lesson = AudioSegment.from_wav(args[0])
    samplerate = lesson.frame_rate

    df = pd.read_csv(args[1], sep=' ', usecols=[2,3], names=['start', 'end'])
    df = df.assign(duration=lambda df: df.end - df.start)

    def calc_loudness(row, audio = lesson):
        """Reads a dataframe row and an audiofile
        returns loudness in dB.
        """
        return(audio[row['start']*1000:row['end']*1000].dBFS)

    def calc_pitch(row, audio = lesson, sr = samplerate):
        """Reads a dataframe row and an audiofile
        returns pitch
        """
        chunk = audio[row['start']*1000:row['end']*1000]
        timeseries = np.asarray(chunk.get_array_of_samples()).astype(float)
        pitch = librosa.estimate_tuning(
            timeseries,
            sr
            )
        return(pitch)

    def calc_tempo(row, audio = lesson, sr = samplerate):
        """Reads a dataframe row and an audiofile
        returns static tempo
        """
        chunk = audio[row['start']*1000:row['end']*1000]
        timeseries = np.asarray(chunk.get_array_of_samples()).astype(float)
        onset_env = librosa.onset.onset_strength(timeseries, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        return(tempo)

    df['loudness'] = df.apply(calc_loudness, axis=1)
    df['pitch'] = df.apply(calc_pitch, axis=1)
    # df['tempo'] = df.apply(calc_tempo, axis=1)
    # Normalize loudness and pitch
    df[
       ['loudness', 'pitch']
       ] = PowerTransformer().fit_transform(df[
                                               ['loudness', 'pitch']
                                               ].to_numpy())

    model = KMeans(n_clusters=int(args[2]),
                        random_state=42).fit(df[['loudness','pitch']].to_numpy())

    df = pd.merge(df,
                  pd.DataFrame({'cluster': model.labels_}),
                  how='left',
                  left_index=True, right_index=True)
    #
    # # VISUALIZATION
    # loud_stat = []
    # pitch_stat = []
    # fig=plt.figure()
    # ax=fig.add_axes([0,0,1,1])
    #
    # for i in range(int(args[2])):
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
    clusters = df[['start', 'end', 'cluster']].to_numpy()
    chunks = []
    for i in range(int(args[2])):
        chunks.append(AudioSegment.empty())

    for segment in clusters:
        chunks[int(segment[2])] += lesson[segment[0]*1000:segment[1]*1000]

    for i, chunk in enumerate(chunks):
        path = 'resegmentation/cluster-%002d.wav' % (i,)
        print(' Writing %s' % (path,))
        chunk.export(path, format='wav')

if __name__ == '__main__':
    main(sys.argv[1:])
