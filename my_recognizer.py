import warnings
from asl_data import SinglesData

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_index, test_tuple in test_set.get_all_Xlengths().items():
        X, length = test_tuple

        possible_words_dict = {}

        for model_word, model in models.items():
            try:
                logL = model.score(X, length)
                possible_words_dict[model_word] = logL
            except:
                possible_words_dict[model_word] = float('-inf')

        probabilities.append(possible_words_dict)
        guesses.append(max(possible_words_dict, key = possible_words_dict.get))

    return probabilities, guesses

