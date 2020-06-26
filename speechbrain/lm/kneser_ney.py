import collections
import math

"""
Implementation tip, from Goodman and Chen:
    "[W]e added an option to the program that would compute the probability of
    all words, not just the words in the test data. We then verified that the
    sum of the probabilities of all words was within roundoff error (10^-9)
    of 1."

"""


def basic_discount_values(*args):
    discount_values = collections.defaultdict(lambda: 0.75)
    discount_values[1] = 0.5
    return discount_values


def ModKN_estimate(counts_by_order, discount_value_func=basic_discount_values):
    probs_by_order = {}
    max_order = len(counts_by_order)
    backoffs = {}
    for order in sorted(counts_by_order):
        if order < max_order:
            counts = get_continuation_counts(counts_by_order[order + 1])
        else:
            counts = counts_by_order[order]
        discount_values = discount_value_func(counts)
        discounts = get_discounts_to_apply(counts, discount_values)
        if order == 1:
            probs_by_order[order] = get_unigram_probs(counts, discounts)
            lower_counts = get_continuation_counts(counts_by_order[order - 1])
        else:
            lower_counts = get_continuation_counts(counts_by_order[order - 1])
            probs, gammas = get_interpolated_probs(
                counts, lower_counts, discounts, probs_by_order[order - 1]
            )
            probs_by_order[order] = probs
            backoffs[order - 1] = gammas
    return probs_by_order, backoffs


def get_continuation_counts(counts):
    continuations = collections.Counter()
    for starting_word, *suffix in counts:
        continuations[tuple(suffix)] += 1
    return continuations


def get_discounts_to_apply(counts, discount_values):
    discounts = collections.defaultdict(collections.Counter)
    for ngram, count in counts.items():
        *context, token = ngram
        discount = max(count - discount_values[count], 0)
        discounts[tuple(context)][token] = discount
    return dict(discounts)


def get_unigram_probs(counts, discounts_to_apply):
    # These interpolate with the uniform distribution
    # vocab_size = len(counts)
    total_counts = sum(counts.values())
    weighted_uniform_dist = sum(discounts_to_apply[tuple()].values()) / sum(
        counts.values()
    )
    probs = {}
    for token, count in counts.items():
        discount = discounts_to_apply[tuple()][
            token
        ]  # Unigrams have empty context
        probs[token] = (count - discount) / total_counts + weighted_uniform_dist
    # LMs have nested representation, first indexed by context, then token
    return {tuple(): probs}


def _pylogsumexp(*args):
    """Log-sum-exp in stdlib-only Python"""
    a_prime = max(args)
    return a_prime + math.log(sum(math.exp(a - a_prime) for a in args))


def get_counts_and_sums(counts):
    pass
    # in_order = sorted(counts.keys())
    # def sums():
    #    pass


def get_interpolated_probs(counts, discounts_to_apply, interp_lm):
    probs = collections.defaultdict(dict)
    gammas = {}
    for ngram, count in counts.items():
        *context, token = ngram
        discount = discounts_to_apply[context][token]
        # Log domain:
        numerator = math.log(counts[ngram] - discount)
        denominator = None
        # denominator = math.log(lower_counts[context])
        if context not in gammas:
            gamma = (
                math.log(sum(discounts_to_apply[context].values()))
                - denominator
            )
            gammas[context] = gamma
        else:
            gamma = gammas[context]
        interp_prob = interp_lm[context][token]
        probs[context][token] = _pylogsumexp(
            numerator - denominator, gamma + interp_prob
        )
    return probs, gammas
