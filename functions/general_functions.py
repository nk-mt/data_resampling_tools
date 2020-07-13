import math
from collections import Counter


def compute_some_statistics_for_the_dataset(dataset, state, is_resampling_case):
    dataset['number_of_examples'] = len(dataset['y_values'])
    unique_target_values = Counter(dataset['y_values'])
    first_mc_tuple, second_mc_tuple = unique_target_values.most_common(2)
    if not is_resampling_case:
        state.positive_tc = second_mc_tuple[0]
    else:
        first_mc_tuple, second_mc_tuple = reorder_tuple_with_positive_class(unique_target_values.most_common(2), state.positive_tc)

    dataset['y_values_as_set'] = (first_mc_tuple[0], second_mc_tuple[0])
    dataset['number_of_positive_examples'] = second_mc_tuple[1]
    dataset['positive_examples_percentage'] = "{:.1f}".format((second_mc_tuple[1] / len(dataset['y_values'])) * 100)
    dataset['imbalanced_ratio'] =  round_half_up(first_mc_tuple[1] / second_mc_tuple[1], 1)


def reorder_tuple_with_positive_class(tuples, positive_class):
    for idx, t in enumerate(tuples):
        if t[0] == positive_class:
            pos_idx = idx
    other_idx = 0 if pos_idx == 1 else 1
    return [tuples[other_idx], tuples[pos_idx]]


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier + 0.5) / multiplier





