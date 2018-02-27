# What do we want to do here. We want to recognize sequences of words. Before
# we were training models on specific words and trying to figure out the best
# model to use based on various criteria. Now we want to use these trained models
# in order to predict sequences of words. So first we train a model for every
# word using some specific feature set. This is fed into the recognize function
# as the models parameter. This is a dictionary where the keys are words and
# the values are GaussianHMM model objects. The second parameter is the
# test_set object which is of SinglesData type. This is basically the same
# type of object as training except that instead of containing individual words
# it contains sequences of words.

# We are returning a tuple of two lists. The first element of the tuple is
# a list of dictionaries where each key is word and value is a Log Likelihood.
# So we want to recognize a sequence, so that is the unit of interest.

# So lets say we have a sentence with three words. Each word
# is going to be a list of lists of features with four features
# per frame. If we feed in this training example will our model output
# word. It will simply output a log likelihood. In which case do we
# have to pass each word to all of the models and calculate the log likelihod
# for each model? Yes because we don't have a single model. We have a collection
# of models one for each word.

# Logic is the following. For each example in SinglesData calculate the
# likelihood of that word for each model. Then take highest likelihood
# and identify the word that model is associated with. That is our best
# guess for the word.

# We just need to use the model.score method on specific word
# and length. For each word_data object we return the output

# WORDGUESS0 corresponds to a single dictionary in the probabilites list
# which contains a word and the correspodning words likelihood. We choose
# the word guess by taking the max of this dictionary.


# The idea here is to to train the models based on the training
# data as defined by the k-fold function and then calculate the
# model score of the data based on the likelihood of the data
# given the the test fold data. No, no, no for each model
# corresponding to some topology of the HMM we train the model
# on the subsets of the data and calculate likelihoods on the
# remaining test parts. If we do a 3-fold that means for each
# topology we will calculate three likelihood values and take
# an average of them and return that as our "best" score.

# Immediate problem. How can I use the base_model function because
# that automatically trains the model on the entire dataset. Well
# you just redefine the self.X and self.

# :param split_index_list: a list of indices as created by KFold splitting
# :param sequences: list of feature sequences
# :return: tuple of list, list in format of X,lengths use in hmmlearn

# What is X again? self.X is a list of training sequences. The most
# elemental item is a list our four features correspoding to a single
# frame. We then of lists of these lists which correspond to single
# motions. We then have lists of these lists so X is a list of
# sign language motions. Each element of X is considered a distinct
# training set? Or it each video frame.

# SO we basically need to just get a list of train indices for example.
# We then need to get the sequence of words which is just self.sequences
# This function then return self.X and self.Length for us.

# Formula we need to implement is DIC = likelihood this word - average(likelihood other words)
# How do we do this. The BIC critiera only considers the likelihood of this specific
# word not all the other words in our training set. We still are looping through different
# numbers of components and trying to find the best fit given the different components. Just
# in this case we are considering the likelihood of all other words.

# The question is do we train a model for each num_components for each word
# on all the other words? This seems enormously computationally expensive.

# We basically need to loop through the dictionary



X, lengths = training.get_word_Xlengths(word)
        logL = model.score(X, lengths)

        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]

        Okay how are we going to do this. We need to loop through all ofthe
        possible values for hidden layers between min and max and select the model
        with the highest BIC score, which is calculated as -2 * logL + p * logN
        where L is the likelihood of the model p is the number of parameters
        and N is the number of data ponts.

        # Number of features can be calculated from

        # p = m^2 + km - 1 # This is worng.
        # n^2 + 2 * d * n - 1
        # where d = # of featuers
        # n - # of hmm states.

        N = length size of observation sequence.




asl.df['left-x-mean'] = asl.df['speaker'].map(std_df['left-x'])
asl.df['left-y-mean'] = asl.df['speaker'].map(std_df['left-y'])
asl.df['right-x-mean'] = asl.df['speaker'].map(std_df['right-x'])
asl.df['right-y-mean'] = asl.df['speaker'].map(std_df['right-x'])

asl.df['left-x-std'] = asl.df['speaker'].map(std_df['left-x'])
asl.df['left-y-std'] = asl.df['speaker'].map(std_df['left-y'])
asl.df['right-x-std'] = asl.df['speaker'].map(std_df['right-x'])
asl.df['right-y-std'] = asl.df['speaker'].map(std_df['right-x'])




