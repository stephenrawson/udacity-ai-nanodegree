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
    for index in range(0, len(test_set.get_all_Xlengths())):
        X, lengths = test_set.get_all_Xlengths()[index]
        best_score = float("-inf") # Top score
        best_word = None # Top word
        probabilities_dict = {} # Save word scores
        for word, model in models.items():
            try:
                # Get model score
                score = model.score(X, lengths)
            except Exception as e:
                # Error processing word
                score = float("-inf")
            if score > best_score:
                best_score, best_word = score, word

            probabilities_dict[word] = score
        guesses.append(best_word)
        probabilities.append(probabilities_dict)
    return probabilities, guesses
