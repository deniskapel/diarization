import sys
import webrtcvad
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from vosk import Model, KaldiRecognizer, SetLogLevel
from utils import read_audio, write_audio, frame_generator, vad_collector
from LessonSegment import LessonSegment
from Config import Config


def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: analyze.py <path to audio file> <n_clusters>\n')
        sys.exit(1)

    """
    Initialize Config
    input:
    n_clusters: Integer set by a user
    text_processor: by default it is set to nltk.stem.snowball.SnowballStemmer
    sample_rate: by default set to 16 kHz due to ASR model specs
    aggressivness: required for VAD, by default set to maximum=3 as audiofiles are long
    """
    config = Config(n_clusters = int(args[1]))

    print(
        "If you want to check any specific target vocabulary, please type them\n",
        "Ex.: train, dog, work, seventeen, Brazil\n",
        "Otherwise, hit enter to skip"
        )

    try:
        lesson_vocabulary = input().lower()
    except SyntaxError:
        pass

    lesson = LessonSegment(
        lesson_vocabulary, # target_vocabulary
        read_audio(args[0], config.sample_rate) # audio to get pcm_data
        )

    # update lesson dictionary to collect statistics
    lesson.update_dictionary(config.text_processor)

    # VAD
    vad = webrtcvad.Vad(config.aggressivness)
    frames = frame_generator(30, lesson.bytes, config.sample_rate)
    frames = list(frames)
    segments = vad_collector(config.sample_rate, 10, 150, vad, frames)

    # ASR
    asr = KaldiRecognizer(Model("model"), config.sample_rate)

    # store LessonSegment instances
    lesson_segments = []
    # store static tempo and pitch of each LessonSegment
    features = []
    for segment in segments:
        seg = LessonSegment('', segment)
        seg.transcribe(asr)
        features.append(seg.get_features(config.sample_rate))
        lesson_segments.append(seg)

    # Clustering
    features = MinMaxScaler().fit_transform(np.array(features))
    cl = GaussianMixture(n_components=config.n_clusters, covariance_type='full')
    clusters = cl.fit_predict(features)


    # Resegmentation - create empty n*LessonSegments
    segments = [LessonSegment('', b'') for n in range(config.n_clusters)]
    for i, cluster in enumerate(clusters):
        cluster = int(cluster)
        segments[cluster].bytes += lesson_segments[i].bytes
        segments[cluster].transcript.extend(lesson_segments[i].transcript)

    [segment.get_staistics(lesson.dictionary) for segment in segments]

    for i, segment in enumerate(segments):
        path = 'resegmentation/cluster-%002d.mp3' % (i,)
        print('Writing %s' % (path,))
        write_audio(path, segment.bytes, config.sample_rate)
        print("\n", segment.statistics, "\n")

if __name__ == '__main__':
    start = datetime.now()
    main(sys.argv[1:])
    print(datetime.now()-start)
