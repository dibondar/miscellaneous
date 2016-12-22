import numpy as np
import networkx
from itertools import combinations_with_replacement

__doc__ = """
Analyze the connectedness of an landscape of an arbitrary dimension.
"""

def connected_components(A, threshold):
    """
    Generate indices of connected components of an array
    :param A: array of an arbitrary dimensions
    :param threshold:
    :return: set of indices to access a connected component
    """

    # create a graph of connectivity
    G = networkx.Graph()

    # Add indices of all elements that are above the threshold as nodes
    G.add_nodes_from(zip(*np.nonzero(A > threshold)))

    # Pre-calculate offset array as combination of -1, 0, +1 except all zeros
    offsets = np.array(list(
        r for r in combinations_with_replacement([-1, 0, 1], len(A.shape)) if any(r)
    ))

    # loop over all vertices in G
    for node in G:
        # loop over neighboring points
        for n in node + offsets:
            n = tuple(n)

            # Add the edge if neighboring index tuple n is in the graph (i.e., above the threshold)
            if n in G:
                G.add_edge(node, n)

    for cc in networkx.algorithms.components.connected.connected_components(G):
        yield cc


###################################################################################################
#
#   Test
#
###################################################################################################

if __name__=='__main__':

    import matplotlib.pyplot as plt

    # pick a threshold
    threshold = 0.7

    # randomly egnerate array to be analized
    A = np.random.rand(15, 15)
    indx = A > threshold
    A[indx] = 1
    A[np.logical_not(indx)] = np.nan

    # find connected componenets and color them
    colored_A = np.copy(A)

    for n, cc in enumerate(connected_components(A, threshold)):
        colored_A[zip(*cc)] = n

    # Simple consistency check
    assert np.logical_xor(np.isnan(A), np.isnan(colored_A)).sum() == 0

    plt.subplot(121)
    plt.title("Original array")
    plt.imshow(A, interpolation='nearest')

    plt.subplot(122)
    plt.title("Colored connected components")
    plt.imshow(colored_A, interpolation='nearest')

    plt.show()
