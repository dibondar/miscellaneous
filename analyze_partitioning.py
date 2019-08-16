import numpy as np
from itertools import product, combinations_with_replacement, combinations, filterfalse
from multiprocessing import Pool

########################################################################################################################
#
#
#
########################################################################################################################


def find_partitions(x):
    """
    Get the list of all possible partitions
    :param x: numpy.array of positive integers
    :return: the set of partitions
    """
    partitions = set()

    S = x.sum()

    for mask in product((True, False,), repeat=len(x)):

        mask = np.array(mask)

        # sum of the element in the set 1
        s1 = x[mask].sum()

        # sum of the element in the set 1
        s2 = S - s1

        if s1 == s2:
            # a partitioning was found that

            # take care of the fact that if (S1, S2) is a partition, then so is (S2, S1)
            if tuple(~mask) not in partitions:
                partitions.add(tuple(mask))

    return partitions

########################################################################################################################
#
#
#
########################################################################################################################


def find_num_partitions(x):
    """
    Get the number of nonequivalent partitions
    :param x: numpy.array of positive integers
    :return: (int, numpy.array) the number of nonequivalent partitions
    """
    num = 0

    x = np.array(x)
    S = x.sum()

    for mask in product((True, False,), repeat=len(x)):

        mask = np.array(mask)

        # sum of the element in the set 1
        s1 = x[mask].sum()

        # sum of the element in the set 1
        s2 = S - s1

        if s1 == s2:
            # a partitioning was found
            num += 1

    # simple check
    assert num % 2 == 0, "we should have discovered even number of partitions since " \
                                     "if (S1, S2) is a partition, then so is (S2, S1)."

    return num // 2, (x if num else None)

########################################################################################################################
#
#
#
########################################################################################################################

# run check on different processes
with Pool() as pool:

    numbers_to_use = np.arange(1, 20)

    results = pool.imap_unordered(find_num_partitions, combinations(numbers_to_use, 10), chunksize=1000)
    results = (_ for _ in results if _[0])
    results = sorted(results, key=lambda _: _[0])
    results.reverse()

with open("partitions.csv", 'tw') as f:
    f.write("the set, number of partitions\n")
    for num, x in results:
        f.write("{}, {}\n".format(x, num))
        # print("The set {} has {} partitions".format(x, num))

########################################################################################################################
#
#   Testing particular partitions
#
########################################################################################################################

# #x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
# x = np.array([ 1,  2,  3, 13, 14, 15, 16, 17, 18, 19])
#
# r = set(
#     (tuple(x[mask]), tuple(x[~mask])) for mask in np.array(list(find_partitions(x)))
# )
#
# # simple check
# assert all(sum(s1) == sum(s2) for s1, s2 in r), "wow! Incorect partitioning."
#
#
# print("Set of partitions for {}\n\n{}".format(x, r))