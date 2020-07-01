import collections
import warnings
from speechbrain.utils.lm import LOG, EXP
import logging

logger = logging.getLogger(__name__)


class InvalidDiscountError(ValueError):
    # Discount value < 0 or otherwise invalid
    pass


def basic_discount_values(counts_by_context):
    discount_values = collections.defaultdict(lambda: 1.5)
    discount_values[2] = 1.0
    discount_values[1] = 0.5
    return discount_values


def get_counts_of_counts(counts_by_context):
    counts_of_counts = collections.Counter()
    for counts in counts_by_context.values():
        counts_of_counts.update(counts.values())
    return counts_of_counts


def ries_discount_values(
    counts_by_context, fallback_func=basic_discount_values
):
    counts_of_counts = get_counts_of_counts(counts_by_context)
    try:
        Y = counts_of_counts[1] / (counts_of_counts[1] + counts_of_counts[2])
        D1 = 1 - (2 * Y * counts_of_counts[2]) / counts_of_counts[1]
        D2 = 2 - (3 * Y * counts_of_counts[3]) / counts_of_counts[2]
        D3plus = 3 - (4 * Y * counts_of_counts[4]) / counts_of_counts[3]
        discount_values = collections.defaultdict(lambda: D3plus)
        discount_values[2] = D2
        discount_values[1] = D1
        for n in (1, 2, 3):
            if discount_values[n] < 0:
                if fallback_func is not None:
                    return fallback_func(counts_by_context)
                else:
                    raise ValueError(
                        "Ries equation would result in discount < 0"
                    )
        return discount_values
    except KeyError:
        if fallback_func is not None:
            return fallback_func(counts_by_context)
        else:
            raise ValueError("Zero count of counts encountered in Ries eqs.")


def ModKN_estimate(
    counts_by_order,
    discount_value_func=ries_discount_values,
    unk="<unk>",
    sos="<s>",
    eos="</s>",
):
    probs_by_order = {}
    backoffs = {}
    # First, setup special symbols:
    prediction_vocab = set(counts_by_order[1][tuple()].keys())
    prediction_vocab.add(unk)
    if counts_by_order[1][tuple()][sos] != 0:
        raise ValueError(
            "Start of sentence unigram included with non-zero count"
        )
    elif sos in prediction_vocab:
        prediction_vocab.discard(sos)
    if eos not in prediction_vocab:
        warnings.warn("End-of-sentence symbol not in prediction vocabulary.")
    # Validate orders:
    max_order = len(counts_by_order)
    for order in range(1, max_order):
        if order not in counts_by_order:
            raise ValueError(
                "Cannot estimate from counts of orders: "
                f"{counts_by_order.keys()}."
            )
    for order in sorted(counts_by_order):
        if order < max_order:
            counts_by_context = get_continuation_counts(
                counts_by_order, order, sos
            )
        else:
            counts_by_context = counts_by_order[order]
        discount_values = discount_value_func(counts_by_context)
        unique_ngrams = sum(
            len(counts) for counts in counts_by_context.values()
        )
        print(f"Order {order} unique ngrams: {unique_ngrams}")
        print(
            f"Order {order} discount values are "
            f"1: {discount_values[1]:.3f}, "
            f"2: {discount_values[2]:.3f}, "
            f"3+: {discount_values[3]:.3f}."
        )
        discounts = get_discounts_to_apply(counts_by_context, discount_values)
        if order == 1:
            interp_lm = get_uniform_distribution(len(prediction_vocab))
        else:
            interp_lm = probs_by_order[order - 1]
        probs, gammas = get_interpolated_probs(
            counts_by_context, discounts, interp_lm
        )
        probs_by_order[order] = probs
        backoffs[
            order - 1
        ] = gammas  # NOTE: this stored the zeroth order gamma too
    # Add special symbol probs:
    probs_by_order[1][tuple()][sos] = 0.0
    backoffs[1][eos] = 0.0
    if unk not in probs_by_order[1][tuple()]:
        interp_lm = get_uniform_distribution(len(prediction_vocab))
        # Here we use the zeroth order gamma:
        probs_by_order[1][tuple()][unk] = (
            backoffs[0][tuple()] * interp_lm[tuple()][unk]
        )
    if unk not in backoffs[1]:
        backoffs[1][unk] = 0.0
    del backoffs[0]  # Now delete the zeroth order gamma
    return probs_by_order, backoffs


def get_continuation_counts(counts_by_order, order, sos):
    continuations = collections.defaultdict(collections.Counter)
    for context, counts in counts_by_order[order + 1].items():
        prefix, *infix = context
        infix = tuple(infix)
        for token, count in counts.items():
            continuations[infix][token] += 1
    adjusted_counts = {}
    for context in counts_by_order[order]:
        if context and context[0] == sos:
            adjusted_counts[context] = counts_by_order[order][context]
        else:
            adjusted_counts[context] = continuations[context]
    return adjusted_counts


def get_discounts_to_apply(counts_by_context, discount_values):
    discounts = collections.defaultdict(collections.Counter)
    for context, counts in counts_by_context.items():
        for token, count in counts.items():
            discount = discount_values[count]
            discount = discount if count > discount else 0
            discounts[tuple(context)][token] = discount
    return dict(discounts)


def get_uniform_distribution(vocab_size):
    # Indexed first by (empty) context
    return {tuple(): collections.defaultdict(lambda: -LOG(vocab_size))}


def _pylogsumexp(*args):
    """Log-sum-exp in stdlib-only Python"""
    a_prime = max(args)
    return a_prime + LOG(sum(EXP(a - a_prime) for a in args))


def get_interpolated_probs(counts_by_context, discounts_to_apply, interp_lm):
    probs = collections.defaultdict(dict)
    gammas = {}
    for context, counts in counts_by_context.items():
        total_counts = sum(counts.values())
        denominator = LOG(total_counts)
        total_discounts = sum(discounts_to_apply[context].values())
        gamma = LOG(total_discounts) - denominator
        gammas[context] = gamma
        # print("==============")
        # print(f"Processing context {context}")
        # print(f"Denominator real: {EXP(denominator)} log: {denominator}")
        # print(f"Gamma real: {EXP(gamma)} log: {gamma}")
        for token, count in counts.items():
            discount = discounts_to_apply[context][token]
            interp_prob = interp_lm[context[1:]][token]
            # print(f"Token: {token}, Count: {count}")
            # print(f"Discount {discount}")
            # print(f"Interpolated LM real: {EXP(interp_prob)} log: {interp_prob}")
            try:
                numerator = LOG(count - discount)
                # print(f"Numerator real: {EXP(numerator)} log: {numerator}")
                probs[context][token] = _pylogsumexp(
                    numerator - denominator, gamma + interp_prob
                )
            except ValueError:
                # Adjusted count became zero
                # Could just define log(0) = -inf
                # But this is faster:
                probs[context][token] = gamma + interp_prob
    return probs, gammas
