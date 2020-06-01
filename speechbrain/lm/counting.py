"""
N-gram counting, discounting, interpolation, and backoff

Author
------
Aku Rouhe 2020
"""
import itertools
import collections


# The following functions are essentially copying the NLTK ngram counting
# pipeline with minor differences. Written from scratch, but with enough
# inspiration that I feel I want to mention the inspiration source:
# NLTK is licenced under the Apache 2.0 Licence, same as SpeechBrain
# See https://github.com/nltk/nltk
# The NLTK implementation is highly focused on getting lazy evaluation.
def pad_ends(
    sequence, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>"
):
    """
    Pad sentence ends with start- and end-of-sentence tokens

    In speech recognition it is important to predict the end of sentence
    and use the start of sentence to condition predictions. Typically this
    is done by adding special tokens (usually <s> and </s>) at the ends of
    each sentence. The <s> token should not be predicted, so some special
    care needs to be taken for unigrams.

    Arguments
    ---------
    sequence : iterator
        The sequence (any iterable type) to pad.
    pad_left : bool
        Whether to pad on the left side as well. True by default.
    left_pad_symbol : any
        The token to use for left side padding. "<s>" by default.
    right_pad_symbol : any
        The token to use for right side padding. "</s>" by deault.

    Returns
    -------
    generator
        A generator which yields the padded sequence.

    Example
    -------
    >>> for token in pad_ends(["Speech", "Brain"]):
    ...     print(token)
    <s>
    Speech
    Brain
    </s>

    """
    if pad_left:
        return itertools.chain(
            (left_pad_symbol,), tuple(sequence), (right_pad_symbol,)
        )
    else:
        return itertools.chain(tuple(sequence), (right_pad_symbol,))


def ngrams(sequence, n):
    """
    Produce all Nth order N-grams from the sequence.

    Arguments
    ---------
    sequence : iterator
        The sequence from which to produce N-grams.
    n : int
        The order of N-grams to produce

    Yields
    ------
    tuple
        Yields each ngram as a tuple.

    Example
    -------
    >>> for ngram in ngrams("Brain", 3):
    ...     print(ngram)
    ('B', 'r', 'a')
    ('r', 'a', 'i')
    ('a', 'i', 'n')

    """
    if n <= 0:
        raise ValueError("N must be >=1")
    # Handle the unigram case specially:
    if n == 1:
        for token in sequence:
            yield (token,)
        return
    iterator = iter(sequence)
    history = []
    for hist_length, token in enumerate(iterator, start=1):
        history.append(token)
        if hist_length == n - 1:
            break
    else:  # For-else is obscure but fits here perfectly
        return
    for token in iterator:
        yield tuple(history) + (token,)
        history.append(token)
        del history[0]
    return


def allgrams(sequence, max_n, skip_first_unigram=True):
    """
    Produce all N-grams upto N-th order from the sequence.

    This will generally be used in an N-gram counting pipeline.

    Arguments
    ---------
    sequence : iterator
        The sequence from which to produce N-grams.
    max_n : int
        The maximum order of N-grams to produce
    skip_first_unigram : bool
        Whether produce the first token in the sequence as a unigram or not.
        Basically, with this True, the sentence start can be avoided in
        unigrams.

    Yields
    ------
    int
        Order of the ngram
    tuple
        Ngram as a tuple.

    Example
    -------
    >>> for order, ngram in allgrams("Brain", 3):
    ...     print(order, ngram)
    1 ('B',)
    1 ('r',)
    2 ('B', 'r')
    1 ('a',)
    2 ('r', 'a')
    3 ('B', 'r', 'a')
    1 ('i',)
    2 ('a', 'i')
    3 ('r', 'a', 'i')
    1 ('n',)
    2 ('i', 'n')
    3 ('a', 'i', 'n')

    """
    if max_n <= 0:
        raise ValueError("Max N must be >=1")
    # Handle the unigram case specially:
    if max_n == 1:
        for token in sequence:
            yield (token,)
        return
    iterator = iter(sequence)
    history = []
    if skip_first_unigram:
        history.append(next(iterator))
    for token in iterator:
        history.append(token)
        if len(history) > max_n:
            del history[0]
        max_order = len(history)
        for order in range(1, max_order + 1):
            yield order, tuple(history[max_order - order :])
    return


def ngrams_for_evaluation(sequence, max_n, predict_first=False):
    """
    Produce each token with the appropriate context.

    The function produces as large N-grams as possible, so growing from
    unigrams/bigrams to max_n.

    E.G. when your model is a trigram model, you'll still only have one token
    of context (the start of sentence) for the first token.

    In general this is useful when evaluating an N-gram model.

    Arguments
    ---------
    sequence : iterator
        The sequence to produce tokens and context from.
    max_n : int
        The maximum N-gram length to produce.
    predict_first : bool
        To produce the first token in the sequence to predict (without
        context) or not. Essentially this should be False when the start of
        sentence symbol is the first in the sequence.

    Yields
    ------
    Any
        The token to predict
    tuple
        The context to predict conditional on.

    Example
    -------
    >>> for token, context in ngrams_for_evaluation("Brain", 3, True):
    ...     print(f"p( {token} |{' ' if context else ''}{' '.join(context)} )")
    p( B | )
    p( r | B )
    p( a | B r )
    p( i | r a )
    p( n | a i )
    """
    if max_n <= 0:
        raise ValueError("Max N must be >=1")
    iterator = iter(sequence)
    history = []
    if not predict_first:
        history.append(next(iterator))
    for token in iterator:
        if len(history) == max_n:
            del history[0]
        yield token, tuple(history)
        history.append(token)
    return


def count_ngrams(data, max_n, skip_first_unigram=True):
    """
    Produces N-gram counts from a list of sentences.

    """
    ngrams_by_order = {}
    for order in range(1, max_n + 1):
        ngrams_by_order[order] = collections.Counter()
    for sentence in data:
        for order, ngram in allgrams(sentence, max_n, skip_first_unigram):
            ngrams_by_order[order][ngram] += 1
    return ngrams_by_order
