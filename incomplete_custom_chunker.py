"""
FILE: incomplete_custom_chunker.py AS GIVEN IN ASSIGNMENT

Based on code from http://www.nltk.org/book/ch07.html#code-classifier-chunker

Authors: Alexis Dimitriadis, Meaghan Fowlie, and #TODO you!

Use ConsecutiveNPChunker to train and use a classifier

Treat _ConsecutiveNPChunkTagger as private: do not use it directly; it is called by ConsecutiveNPChunker

"""
from abc import ABC

import nltk
from nltk.chunk.util import conlltags2tree, tree2conlltags

# If numpy is absent, nltk fails with a very confusing error.
# We avoid problems by checking directly
try:
    import numpy
except ImportError:
    print("You need to download and install numpy!!!")
    raise


class ConsecutiveNPChunker(nltk.ChunkParserI, ABC):
    """
    Trained classifier for NER
    Classifier Input: a POS-tagged sentence -- (word, POS) list
    Classifier Output: an IOB-tagged sentence -- ((word, POS), IOB) list
    Attributes:
        tagger: a _ConsecutiveNPChunkTagger object, trained on the feature map
                and training set given to __init__
    """

    def __init__(self, feature_function, training_sentences, algorithm="NaiveBayes", verbose=0):
        """
        Train a classifier on chunked data in Tree format.
        :param feature_function: The function that will compute features for each
         word in a sentence.
        :param training_sentences: A list of sentences in chunked (Tree) format.
        :param algorithm: str: which classifier to use
            (default NaiveBayes; other possibilities IIS, GIS, and DecisionTree)
        :param verbose: int: how much to print during training (default 0, meaning nothing)
        """

        # train the tagger
        self.tagger = _ConsecutiveNPChunkTagger(feature_function,
                                                training_sentences,
                                                algorithm=algorithm,
                                                verbose=verbose)

    def parse(self, sentence):
        """
        tag a sentence with IOB tags and return a tree
        :param sentence: list of (word, POS) pairs
        :return: Conll tree
        """
        tagged_sent = self.tagger.tag(sentence)
        # return to conll format
        conll_tags = [(word, pos, iob) for ((word, pos), iob) in tagged_sent]
        return conlltags2tree(conll_tags)

    def explain(self):
        """Print the docstring of our feature extraction function"""
        print("Algorithm:", self.tagger.algorithm)
        # Print the feature map's doc string:
        print(self.tagger.feature_function.__doc__)

    def show_most_informative_features(self, n=10):
        """
        Call our classifier's `show_most_informative_features()` function.
        :param n : int: the number of features to print (default 10)
        """
        self.tagger.classifier.show_most_informative_features(n)

    def tag_corpus_sentence(self, sentence):
        """
        tags a sentence in nltk.Tree form
        :param sentence: nltk.Tree formated sentence, as in the corpora
        :return tagged sentence as ((word, POS), IOB) pairs,
                where IOB are the tags predicted by the model
        """
        # turn the sentence into a unary list,
        # use reformat_corpus_for_tagger,
        # and untag the sentence
        s = nltk.tag.untag(self.tagger.reformat_corpus_for_tagger([sentence])[0])

        # use the trained tagger to re-tag the sentence
        return list(self.tagger.tag(s))

    def compare_output_to_gold(self, sentence):
        """
        tags a sentence from the corpus and prints out a word-by-word comparison with the gold data
        :param sentence: a sentence in nltk.Tree form, as in the corpora
        """
        gold = self.tagger.reformat_corpus_for_tagger([sentence])[0]
        tagged = self.tag_corpus_sentence(sentence)
        print("gold")
        print("tagged\n")
        for i in range(len(gold)):
            print(gold[i])
            print(tagged[i], "\n")

#%%
class _ConsecutiveNPChunkTagger(nltk.TaggerI):
    """This class is not meant to be
    used directly: Use ConsecutiveNPChunker instead.
    Attributes:
        feature_function: map from
                    (sentence, word index, history of features assigned so far)
                    to dict of feature name: feature value.
                    Imported from features.py.
        train_set: list of (feature dict, IOB tag) pairs
        classifier: nltk.NaiveBayesClassifier trained on training_sentences (default)
        algorithm: str: name of the algorithm for reporting
    """

    def __init__(self, feature_function, training_sentences, algorithm="NaiveBayes", verbose=0):
        """
        Initialises and trains a tagger using the given features
         and training sentences
        :param feature_function: function that maps (untagged sentence, word index,
         history) to a dict of features (from features.py)
        :param training_sentences  : training sentences as list of
                            ((word, pos_tag), iob_tag) pairs
        :param algorithm: str:  which training algorithm to use. Default NaiveBayes.
                                Other options are IIS, GIS, and DecisionTree.
        :param verbose: int: IIS and GIS only: how much to print during training (0 = nothing)
        """

        self.train_set = []  # initialise self.train_set

        # TODO: store the feature_function parameter as self.feature_function
        # TODO: call self.create_training_data on training_sentences
        # TODO: check that algorithm is one of "NaiveBayes", "DecisionTree", "IIS", and "GIS"
        # and raise an error if it's not


        # set and train the classifier
        if algorithm == "NaiveBayes":
            self.classifier = nltk.NaiveBayesClassifier.train(self.train_set)
            self.algorithm = "Naive Bayes"
        elif algorithm == "DecisionTree":
            self.classifier = nltk.DecisionTreeClassifier.train(self.train_set)
            self.algorithm = "Decision Tree"
        else:
            self.classifier = nltk.MaxentClassifier.train(
                self.train_set, algorithm=algorithm, trace=verbose)
            self.algorithm = f"Maximum Entropy with {algorithm}"

    @staticmethod
    def reformat_corpus_for_tagger(training_sentences):
        """
        Given a corpus in nltk.Tree list format, returns the corpus as a list of lists of tuples,
        where each tuple ((word, POS), IOB) includes the word, its POS tag, and the IOB tag to be predicted.
        :param training_sentences nltk.Tree list of IOB-tagged sentences
        """
        return [[((word, pos), iob) for (word, pos, iob) in tree2conlltags(sent)] for sent in training_sentences]

    def create_training_data(self, training_sentences):
        """
        Creates training data from the corpus of training_sentences and self.feature_function
        stores a list of (dict, IOB tag) pairs as self.train_set

        :param training_sentences: list of nltk.Trees with IOB tags

        TODO make your function into a method that
            uses the stored self.feature_function,
            calls self.reformat_corpus_for_tagger on training_sentences,
            and stores the training data as self.train_set
            (and update this comment!)
        """
        # TODO reformat sentences to ((word, pos_tag), iob_tag) pairs
        # TODO turn the sentences into appropriate training data by finding their features
        # TODO store them in self.train_set
        ...


    def tag(self, sentence):
        """
        uses the trained classifier to tag a sentence
        :param sentence: list of (word, pos_tag) pairs
        :return: list of ((word, pos_tag), IOB_tag) pairs
        """
        history = []
        for i in range(len(sentence)):
            # extract the features
            feature_dict = self.feature_function(sentence, i, history)
            # tag the sentence
            tag = self.classifier.classify(feature_dict)
            history.append(tag)
        return zip(sentence, history)
