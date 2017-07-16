import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.n_components = range(self.min_n_components, self.max_n_components + 1)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("inf") # KEEP LOWEST
        best_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                # From Forum -> parameters = n * n + 2 * n * d
                parameters = num_states * (num_states - 1) + 2 * self.X.shape[1] * num_states
                score = -2 * logL + parameters * math.log(self.X.shape[0])
                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                pass
        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("-inf") # KEEP HIGHEST
        best_model = None

        vocab = list(self.words)
        vocab.remove(self.this_word)
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                total_score = 0
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                for word in vocab:
                    X, lengths = self.hwords[word]
                    total_score += model.score(X, lengths)
                score = logL - total_score / (len(self.words) - 1)
                if best_score < score:
                    best_score = score
                    best_model = model
            except:
                    pass
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score = float("-inf") #KEEP HIGHEST
        best_model = None
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            n_splits = 3
            model, logL = None, None
            # Check Data Amount
            if len(self.sequences) < n_splits:
                break
            split_method = KFold(random_state=self.random_state, n_splits=n_splits)
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # Help from the Forums.
                x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                # Split Training sequences
                x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = GaussianHMM(n_components=num_states, 
                                        n_iter=1000,).fit(x_train, lengths_train)
                    logL = model.score(x_test, lengths_test)
                    scores.append(logL)
                except Exception as e:
                    break
            avg = None
            if len(scores) > 0:
                avg = np.average(scores)
            else:
                avg = float("-inf")
            if avg > best_score:
                best_score, best_model = avg, model
        if best_model is None:
            return self.base_model(self.n_constant)
        else:
            return best_model
