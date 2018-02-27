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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            logN = math.log(len(self.X))
            num_features =  len(self.X[1])
            best_bic_score = float("inf")

            for num_states in range(self.min_n_components, self.max_n_components):

                model = self.base_model(num_states)

                logL = model.score(self.X, self.lengths)
                num_parameters = math.pow(num_states, 2) + 2 * num_features * num_states - 1

                bic_score = -2 * logL + num_parameters * logN

                if bic_score < best_bic_score:
                    best_bic_model = model
                    best_bic_score = bic_score

            return best_bic_model
        except:
            return self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            best_dic_score = float("-inf")

            for num_states in range(self.min_n_components, self.max_n_components):
                model = self.base_model(num_states)
                logL = model.score(self.X, self.lengths)
                anti_logL_list = []

                for word, value in self.hwords.items():
                    if word != self.this_word:
                        anti_logL_list.append(model.score(value[0], value[1]))

                        dic_score = logL - np.mean(anti_logL_list)

                if dic_score > best_dic_score:
                    best_dic_model = model
                    best_dic_score = dic_score

            return best_dic_model
        except:
            return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def select(self):
        try:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            best_logL_score = float("inf")

            for num_states in range(self.min_n_components, self.max_n_components):

                logL_list = []

                split_method = KFold(n_splits = 2)
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                    model = self.base_model(num_states)
                    logL = model.score(test_X, test_lengths)

                    logL_list.append(logL)

                    logL_score = np.mean(logL_list)

                    if logL_score < best_logL_score:
                        best_model = model
                        best_logL_score = logL_score

            return best_model
        except:
            return self.base_model(self.n_constant)

