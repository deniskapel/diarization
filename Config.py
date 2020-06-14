from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

class Config(object):
    """docstring for Config"""

    def __init__(self, n_clusters,
                 text_processor=SnowballStemmer('english'), sample_rate=16000, aggressivness=3):
        self.n_clusters = n_clusters
        self.text_processor = text_processor
        self.sample_rate = sample_rate
        self.aggressivness = aggressivness
