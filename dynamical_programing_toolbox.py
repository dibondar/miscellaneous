import numpy as np
import networkx as nx
from collections import namedtuple
from operator import itemgetter

__doc__ = """
This is a prototype for merging the toolbox method of Prof. Herschel Rabitz and dynamical programing (Bellman).
"""


class DyProToolbox:
    """

    """
    # data structure to save state
    CState = namedtuple('State', ['cost_func', 'node', 'field', 'state'])

    get_cost_function = itemgetter('cost_func')
    get_iteration = itemgetter('iteration')

    def __init__(self, init_state, init_field, propagator, field_switching, cost_func, cut_off):

        # Save parameters
        self.propagator = propagator
        self.field_switching = field_switching
        self.cost_func = cost_func
        self.cut_off = cut_off

        # Initialize heaps for performing the optimization
        S = self.CState(
            cost_func=self.cost_func(init_state),
            node=0,
            field=init_field,
            state=init_state
        )
        self.previous_heap = [S]

        # Number of time step in optimization iteration
        self.current_iteration = 0

        # Landscape saved as a graph, where vertices are states
        self.landscape = nx.DiGraph()
        self.landscape.add_node(S.node, cost_func=S.cost_func, iteration=self.current_iteration)

    def next_time_step(self):
        """
        Go to the next time step in the time-domain optimization
        :return:
        """
        self.current_iteration += 1

        # initialize the heap
        current_heap = []

        # Loop over all states selected at the previous step
        for S in self.previous_heap:
            # Loop over all laser fields attainable from current field
            for F in self.field_switching[S.field]:

                # update the state
                new_state = self.propagator(F, S.state)

                new_S = self.CState(
                    cost_func=self.cost_func(new_state),
                    node=len(self.landscape),
                    field=F,
                    state=new_state
                )

                self.landscape.add_node(new_S.node, cost_func=new_S.cost_func, iteration=self.current_iteration)
                self.landscape.add_edge(new_S.node, S.node, field=new_S.field)

                current_heap.append(new_S)

        # Sort list current_heap so that it is a true heap
        current_heap.sort(reverse=True)

        self.previous_heap = (current_heap[:self.cut_off] if self.cut_off > 0 else current_heap)

    def get_pos_iteration_cost(self):
        """
        Landscape plotting utility.
        :return:
        """
        return dict(
            (node, (self.get_iteration(prop), self.get_cost_function(prop)))
            for node, prop in self.landscape.node.iteritems()
        )

    def get_pos_cost_iteration(self):
        """
        Landscape plotting utility.
        :return:
        """
        return dict(
            (node, (self.get_cost_function(prop), self.get_iteration(prop)))
            for node, prop in self.landscape.node.iteritems()
        )

    def get_node_color(self):
        """
        Landscape plotting utility.
        :return:
        """
        return [self.get_cost_function(n) for n in self.landscape.node.values()]

    def get_edge_color(self):
        """
        Landscape plotting utility.
        :return:
        """
        return [d['field'] for _,_,d in self.landscape.edges(data=True) if 'field' in d]

    def get_optimal_policy(self):
        """
        Find the optimal control policy to maximize the objective function
        :return: max value of the cost function
            and list of fields that take from the initial condition to the optimal solution
        """
        # Find the maximal node
        max_cost, max_node = max(
             (self.get_cost_function(prop), node) for node, prop in self.landscape.node.iteritems()
        )

        # Initialize variables
        opt_policy_fields = []
        current_node = self.landscape[max_node]

        # Walk from best node backwards to the initial condition
        while current_node:

            assert len(current_node) == 1, "Algorithm implemented incorrectly"

            # Assertion above guarantees that there will be only one element
            next_node, prop = current_node.items()[0]

            # Add extracted value of the field
            opt_policy_fields.append(prop['field'])

            # Extract next node
            current_node = self.landscape[next_node]

        # reverse the order in the list
        opt_policy_fields.reverse()

        return max_cost, opt_policy_fields

    def get_landscape_connectedness(self, **kwargs):
        """
        :param kwargs: the same as in https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        :return: list of list. The outermost list is a list of levels. The innermost list contains the sizes of
            each connected component.
        """
        costs, nodes = zip(
            *sorted(
                (self.get_cost_function(prop), node) for node, prop in self.landscape.node.iteritems()
            )
        )

        costs = np.array(costs)

        # create the histogram of cost function values
        _, bin_edges = np.histogram(costs, **kwargs)

        levels = (
            nodes[indx:] for indx in np.searchsorted(costs, bin_edges[1:-1])
        )

        # make an undirected shallow copy of self.landscape
        landscape = nx.Graph(self.landscape)

        return [
            sorted(
                (len(c) for c in nx.connected_components(landscape.subgraph(nbunch))),
                reverse=True
            )
            for nbunch in levels
        ]

###################################################################################################
#
#   Test
#
###################################################################################################

if __name__=='__main__':

    import matplotlib.pyplot as plt

    np.random.seed(1839127)

    from itertools import product
    from scipy.linalg import expm

    ###############################################################################################
    #
    #    Crate graph for switching the values of the field
    #
    ###############################################################################################

    field_switching = nx.Graph()

    field = np.linspace(0, 9, 3)
    for k in xrange(1, field.size):
        field_switching.add_edges_from(
            product([field[k]], field[k-1:k+2])
        )
    # add separatelly the lower point
    field_switching.add_edges_from(
        [(field[0], field[0]), (field[0], field[1])]
    )

    # nx.draw_circular(
    #     field_switching,
    #     labels=dict((n, str(n)) for n in field_switching.nodes())
    # )
    # plt.show()

    ###############################################################################################
    #
    #   Create the dictionary of propagators
    #
    ###############################################################################################

    # Number of levels in the quantum system
    N = 3

    class CPropagator:
        """
        Propagator with precalculated matrix exponents
        """
        def __init__(self):
            # Generate the unperturbed hamiltonian
            H0 = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            H0 += H0.conj().T

            # Generate the dipole matrix
            V = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            V += V.conj().T

            # precalculate the matrix exponents
            self._propagators = dict(
               (f, expm(-1j * (H0 + f * V))) for f in field_switching
            )

        def __call__(self, f, state):
            return self._propagators[f].dot(state)

    ###############################################################################################
    #
    #   Create the objective (cost) function
    #
    ###############################################################################################

    class CCostFunc:
        """
        Objective function
        """
        def __init__(self):
            self.O = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            self.O += self.O.conj().T

        def __call__(self, state):
            return np.einsum('ij,i,j', self.O, state.conj(), state).real

    ###############################################################################################
    #
    #   Run the optimization
    #
    ###############################################################################################

    init_state = np.zeros(N)
    init_state[0] = 1.

    opt = DyProToolbox(
        init_state,
        field[field.size / 2],
        CPropagator(),
        field_switching,
        CCostFunc(),
        2000
    )

    for _ in xrange(11):
       opt.next_time_step()

    ###############################################################################################
    #
    #   Plot results
    #
    ###############################################################################################

    plt.title("Landscape")
    plt.xlabel("time variable (dt)")
    plt.ylabel("Value of objective function")
    nx.draw(
        opt.landscape,
        pos=opt.get_pos_iteration_cost(),
        node_color=opt.get_node_color(),
        edge_color=opt.get_edge_color(),
        arrows=False,
        alpha=0.6,
        node_shape='s',
        linewidths=0,
    )
    plt.axis('on')
    plt.show()

    # Display the connectedness analysis
    connect_info = opt.get_landscape_connectedness()

    plt.subplot(121)
    plt.title("Number of disconnected pieces")
    plt.semilogy([len(_) for _ in connect_info], '*-')
    plt.ylabel('Number of disconnected pieces')
    plt.xlabel('Level set number')

    plt.subplot(122)
    plt.title("Size of largest connected piece")
    plt.semilogy([max(_) for _ in connect_info], '*-')
    plt.ylabel("Size of largest connected piece")
    plt.xlabel('Level set number')

    plt.show()