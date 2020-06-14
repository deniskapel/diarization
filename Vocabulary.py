from utils import tokenizer

class Vocabulary(object):
    """Update it later to include wordbanks
    related to specific topics and CEFR levels"""

    def __init__(self, target_vocabulary):
        self.target_vocabulary = tokenizer(target_vocabulary)
        self.dictionary = {}

    def update_dictionary(self, text_processor):

        # If there is no target_vocabulary, stop.
        if len(self.target_vocabulary) == 0:
            return

        # update a dictionary {stem: word}
        self.dictionary.update(
            {word: text_processor.stem(word) for word in self.target_vocabulary}
            )
