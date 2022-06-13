import json
from collections import Counter, OrderedDict

import jieba
import nltk
from scipy.stats import entropy

from discover_utils import *

class Discoverer(object):
    def __init__(self, verbose=0.01, cache_path='./preprocessed', save_segmentation=True):
        self.is_fitted = False
        self.verbose = verbose

        self.corpus_name = None

        # The location to save and load the cached segmented documents.
        self.cache_path = cache_path
        '''
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        '''
        # A flag variable determining whether to cache the segmented documents or not.
        # If this flag is set to `False`, the above defined cache_path will be of no use.
        self.save_segmentation = save_segmentation

        # A list containing raw documents, with each element as a document.
        self.documents = None
        self.nr_of_documents = None

        # segmented docs. Identical to `self.documents` except that the documents are converted to a list of words
        # comprising them.
        self.unigram_docs = None

        self.char_counter = self.unigram_counter = self.bigram_counter = self.trigram_counter = None
        self.nr_of_chars = self.nr_of_unigrams = self.nr_of_bigrams = self.nr_of_trigrams = None

        # 8 columns (excluding the index column): `tf`, `agg`, `max_entropy`, `min_entropy`, `left_entropy`
        # `right_entropy`, `left_wc`, 'right_wc`.
        self.unigram_stats = self.bigram_stats = None

        # Same to `self.unigram_stats` except that the words that already in the given dictionary are removed.
        self.new_unigram_stats = None

    def fit(self, documents, corpus_name='default'):
        """
        Everything before generating `.csv` files is done here. For the purpose of each field, refer to `__init()__`.
        Once `fit()`, `get_new_unigrams()` and `purify()` are called, all the remaining works are handed over to 
        `discoverer_utils`.
        """
        self.is_fitted = True
        self.corpus_name = corpus_name
        self.documents = documents
        self.nr_of_documents = len(documents)

        # Segment every document. Each element inside the `unigram_docs` is a list of words compromising the
        # corresponding document.
        self.unigram_docs = self._segment(documents)

        # Count the occurrence of each char.
        self.char_counter = self._get_chars(self.documents)
        self.char_counter[' '] = 1
        
        self.nr_of_chars = sum(self.char_counter.values())
        
        # Count the occurrence of each word.
        self.unigram_counter = self._get_unigrams(self.unigram_docs)
        self.nr_of_unigrams = sum(self.unigram_counter.values())

        # Count the occurrence of each bigram.
        self.bigram_counter = self._get_xgrams(self.unigram_docs)
        self.nr_of_bigrams = sum(self.bigram_counter.values())

        # Count the occurrence of each trigram.
        self.trigram_counter = self._get_xgrams(self.unigram_docs, tokenizer=nltk.trigrams)
        self.nr_of_trigrams = sum(self.trigram_counter.values())

        self.unigram_stats = self._get_stats(self.unigram_counter)
        self.bigram_stats = self._get_stats(self.bigram_counter)
        
    def _load_json(self, obj_name):
        """
        A helper function that facilitates cache loading process. This function will examine the existence of caches 
        corresponding to the given corpus, load, and log.
        """
        # The cache file names are of the form "/{root dir}/{obj_name} @# {corpus_name}.json"
        ## The former `join()` is `os.path.join()` and the latter one is `str.join()`.
        json_path = join(self.cache_path, ' @# '.join([obj_name, self.corpus_name, '.json']))
        if os.path.exists(json_path):
            logger.info('`{}` exists. Loading ...'.format(' @# '.join([obj_name, self.corpus_name])))
            with open(json_path, encoding='utf8') as f:
                obj = json.load(f)
                logger.info('`{}` loaded.'.format(' @# '.join([obj_name, self.corpus_name])))
                return obj
        else:
            logger.info('`{}` preprocessed does not exist. Get ready to train'.format(obj_name))

    def _dump_json(self, save_flag, obj, obj_name):
        """
        A helper function to facilitates the caching process.
        :param save_flag: determine whether to cache or not.
        :param obj: the `object` (counter here) to be cached.
        :param obj_name: the name of the `obj`.
        """
        json_path = join(self.cache_path, ' @# '.join([obj_name, self.corpus_name, '.json']))
        if save_flag and not os.path.exists(json_path):
            with open(json_path, 'w', encoding='utf8') as f:
                json.dump(obj, f)
                logger.info('`{}` dumped'.format(obj_name))
        elif os.path.exists(json_path):
            logger.info('`{}` exists. No need to dump.'.format(obj_name))
            
    def _segment(self, documents):
        """
        Invoke `jieba.lcut()` and return segemented documents
        """
        
        unigram_docs = list()
        for idx, each_doc in enumerate(documents):
            #verbose_logging('cutting ... {} / {}', idx, self.nr_of_documents, self.verbose)
            unigram_docs.append(jieba.lcut(each_doc))
        self._dump_json(self.save_segmentation, unigram_docs, 'unigram_docs')
        return unigram_docs

    def _get_chars(self, documents):
        """
        `collections.Counter()` is a subclass of `dict`, which takes an iterable or a `collections.Counter()` and update
        itself.
        """
        char_counter = Counter()
        for idx, each_doc in enumerate(documents):
            # verbose_logging('counting characters ... {} / {}', idx, self.nr_of_documents, self.verbose)
            char_counter.update(each_doc)
        return OrderedDict(sorted(char_counter.items(), key=lambda x: x[1], reverse=True))

    def _get_unigrams(self, unigram_docs):
        """
        Return a dict counting the occurrence of each unigram. 
        :param unigram_docs: 
        :return: 
        """
        unigram_counter = Counter()
        for idx, each_unigram_doc in enumerate(unigram_docs):
            # verbose_logging('counting unigrams ... {} / {}', idx, self.nr_of_documents, self.verbose)
            unigram_counter.update(each_unigram_doc)
        return OrderedDict(sorted(unigram_counter.items(), key=lambda x: x[1], reverse=True))

    def _get_xgrams(self, unigram_docs, tokenizer=nltk.bigrams):
        """
        Return a dict counting the occurrence of each x_gram. When the process is being logged to the console.
        The "x" in the `x_gram` is inferred from the `tokenizer`.
        """
        x_gram_counter = Counter()
        for idx, each_unigram_doc in enumerate(unigram_docs):
            # verbose_logging('counting {} ... {} / {}', idx, self.nr_of_documents, self.verbose, tokenizer.__name__)
            x_gram_counter.update(tokenizer((each_unigram_doc)))
        return OrderedDict(sorted(x_gram_counter.items(), key=lambda x: x[1], reverse=True))

    def _get_stats(self, counter, by='tf'):
        """
        Compose a `stats` of type `pandas.DataFrame`, with 8 columns as followed:
            `tf`, `aggregation coefficient`, `max_entropy`, `min_entropy`, `left_entropy`, `right_entropy`, 
            `left_wc`, `right_wc`
        """
        counter_aggregation = self._aggregation_coef(counter)
        # Convert `counter_aggregation` to a `Series` to facilitate the following concatenation process.
        counter_aggregation = pd.Series(counter_aggregation, name='agg_coef')

        # Calculate the boundary entropy.
        boundary_entropy = self._get_boundary_stats(counter)

        # Convert the `counter` to a `Series` to facilitate the following concatenation process.
        counts = pd.Series(counter, name='tf')
        return pd.concat([counts, counter_aggregation, boundary_entropy], axis=1).sort_values(by=by, ascending=False)

    def get_new_unigrams(self, dictionary):
        """
        Initialize `new_unigram_counter` and `new_unigram_stats` attributes. 
        The words already in the dictionary will be filtered.
        :param dictionary: An iterable containing words.
        """
        logger.info('Getting new words...')
        new_words = set(self.unigram_counter) - set(dictionary)
        self.new_unigram_counter = OrderedDict(
            [(word, self.unigram_counter[word]) for word in self.unigram_counter if word in new_words])

        self.new_unigram_stats = self.unigram_stats.loc[new_words].sort_values(by='agg_coef', ascending=False)
        logger.info('New unigrams gotten. Please refer to `new_unigram_counter and `new_unigram_stats`.')

    def __parent(self, counter):
        """
        trigram counter > bigram counter > unigram counter > char counter.
        For example, when you are about to calculate the boundary entropy of a unigram, the bigrams containing that 
        unigram will greatly facilitate your calculation. In this case, you need to the access the the parent of the
        `unigram_counter`, and this is where the function comes in.
        :param counter: 
        :return: 
        """
        if counter == self.char_counter:
            return self.unigram_counter
        if counter == self.unigram_counter:
            return self.bigram_counter
        if counter == self.bigram_counter:
            return self.trigram_counter

    def __counter_grams(self, counter):
        """
        A helper function to get the number of grams in the `counter`.
        Only `unigram_counter`, `bigram_counter` and `trigram_counter` are supported.
        """
        if counter == self.unigram_counter:
            return 1
        if counter == self.bigram_counter:
            return 2
        if counter == self.trigram_counter:
            return 3
        else:
            raise Exception('Not supported. Refer to `help(__counter_grams)`')

    def _get_boundary_stats(self, counter):
        """
        get the boundary statistics of each headword. A boundary stats contains the following columns:
        'max_entropy', 'min_entropy', 'left_entropy', 'right_entropy', 'left_wc', 'right_wc'
        """
        if not self.is_fitted:
            raise Exception('This model has not been trained')

        left_word_counter, right_word_counter = self._get_boundary_word_counts(counter)

        columns = ['max_entropy', 'min_entropy', 'left_entropy', 'right_entropy', 'left_wc', 'right_wc']
        stats = []
        words = []

        # Calculate the entropy after the left adjacent words and right adjacent words of each x-gram are gotten.
        for idx, each_word in enumerate(counter):
            # verbose_logging('Calculating boundary entropy ... {} / {}', idx, len(counter), self.verbose)
            left_entropy = entropy([count[1] for count in left_word_counter[each_word]], base=2)
            right_entropy = entropy([count[1] for count in right_word_counter[each_word]], base=2)
            words.append(each_word)
            stats.append((
                max(left_entropy, right_entropy),
                min(left_entropy, right_entropy),
                left_entropy,
                right_entropy,
                left_word_counter[each_word],
                right_word_counter[each_word],
            ))
        # Name the index. This seems to be of no use, however.
        words_index = pd.Index(words, name=('word{}'.format(num + 1) for num in range(self.__counter_grams(counter))))
        return pd.DataFrame(stats, index=words_index, columns=columns).sort_values(by='max_entropy', ascending=False)

    def _get_boundary_word_counts(self, counter):
        """
        Get all the left and right adjacent words of each x-gram in the given `counter` and sort them by the frequency 
        in descending order.
        """
        left_adjacent_word_counter = defaultdict(lambda: defaultdict(int))
        right_adjacent_word_counter = defaultdict(lambda: defaultdict(int))

        # For the behavior and the motivation, refer to `__parent()`.
        parent_counter = self.__parent(counter)

        for idx, each_x_gram in enumerate(parent_counter):
            # verbose_logging('counting neighboring words ... {} / {}', idx, len(parent_counter), self.verbose)

            # The words in a x_gram ranging from 0-position to penultimate-position are considered as left adjacent words.
            # The words in a x_gram ranging from 1-position to last-position are considered as right adjacent words.
            head_left, head_right = each_x_gram[:-1], each_x_gram[1:]

            # If the given `counter` is a `unigram counter`, then there's no need to wrap it with a list.
            if len(head_left) == 1:
                head_left = head_left[0]
                head_right = head_right[0]

            # Like C++, operators, overloadable, behavior properly on many built-in classes.
            left_adjacent_word_counter[head_right][each_x_gram[0]] += parent_counter[each_x_gram]
            right_adjacent_word_counter[head_left][each_x_gram[-1]] += parent_counter[each_x_gram]

        def _sort_and_padding(word, adjacent_word_counter):
            """
            The `word_counter` can be a `left_word_counter` or a `right_word_counter`.
            The `word` is not guaranteed to appear in the given `word_counter`, because not every word has both left
            or right neighbor words
            If the `word` exists in the `word_counter`, then its entry is sorted by the frequency of its neighbor words.
            Else, the `word_counter` will be assigned to an empty list, i.e. [].
            """
            if word in adjacent_word_counter:
                adjacent_word_counter[word] = sorted(adjacent_word_counter[word].items(), key=lambda x: x[1],
                                                     reverse=True)
            else:
                adjacent_word_counter[word] = []

        # Fill empty entries in `left_adjacent_word_counter` and `right_adjacent_word_counter` with an empty list.
        for each_word in counter:
            _sort_and_padding(each_word, left_adjacent_word_counter)
            _sort_and_padding(each_word, right_adjacent_word_counter)

        return left_adjacent_word_counter, right_adjacent_word_counter

    def _cal_aggregation_coef(self, x_gram):
        """
        Calculate the aggregation coefficient of a collocation. Aggregation coefficient is a variant of PMI (pair-wise 
        mutual information). 
        The elements considered to be the constituent of a collocation is the
        Only unigram and bigram are supported for now.
                             P(w1, w2)                  C(w1, w2) / #{nr_of_bigrams}
        Aggregation coef =  ----------- = -----------------------------------------------------
                             P(w1)P(w2)     C(w1) / #{nr_of_bigrams} * C(w2) / #{nr_of_bigrams}
        In case of overflow and underflow, we divide each C(w) by a #{nr_of_bigrams} to make sure the results fall into
        the acceptable interval. In fact, it doesn't matter whether you divided the occurrences by `nr_of_bigrams` or
        `nr_of_unigrams` or the like. Any number that scale the coefficient to a safe interval can be alternatives.
        """
        if isinstance(x_gram, str):
            if chinese_pattern.search(x_gram):
                numerator = self.unigram_counter[x_gram] / self.nr_of_bigrams
                denominator_vector = np.array(
                    [self.char_counter[each_char] for each_char in x_gram]) / self.nr_of_bigrams
            # fails to meet the "at least one Chinese characters" condition.
            else:
                numerator, denominator_vector = np.inf, 1
        else:
            numerator = self.bigram_counter[x_gram] / self.nr_of_bigrams
            denominator_vector = np.array(
                [self.unigram_counter[each_unigram] for each_unigram in x_gram]) / self.nr_of_bigrams
        return numerator / np.prod(np.array(denominator_vector))

    def _aggregation_coef(self, counter):
        """
        Calculate the aggregation coefficient of each word or gram in the given `counter`.
        :param counter: Any counter belonging to this class.
        :return: 
        """
        aggre_coef = list()
        for idx, each_x_gram in enumerate(counter):
            # verbose_logging('Calculating aggregation coefficients ... {} / {}', idx, len(counter), self.verbose)
            aggre_coef.append((each_x_gram, self._cal_aggregation_coef(each_x_gram)))
        return OrderedDict(sorted(aggre_coef, key=lambda x: x[1], reverse=True))

