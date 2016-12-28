__doc__ = """
This is a prototype for merging the toolbox method of Prof. Herschel Rabitz and dynamical programing (Bellman).
"""
import pydot
import networkx as nx
from collections import namedtuple

class DyProgToolbox:
    """

    """
    # data structure to save state
    CState = namedtuple('State', ['cost_func', 'node', 'field', 'state'])

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

                self.landscape.add_node(new_S.node, cost_func=S.cost_func, iteration=self.current_iteration)
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
            (node, (prop['iteration'], prop['cost_func'])) for node, prop in self.landscape.node.iteritems()
        )

    def get_pos_cost_iteration(self):
        """
        Landscape plotting utility.
        :return:
        """
        return dict(
            (node, (prop['cost_func'], prop['iteration'])) for node, prop in self.landscape.node.iteritems()
        )

    def get_node_color(self):
        """
        Landscape plotting utility.
        :return:
        """
        return [n['cost_func'] for n in self.landscape.node.values()]

    def get_optimal_policy(self):
        """

        :return:
        """
        # max_cost, best_node = max(
        #     (prop['cost_func'], node) for node, prop in self.landscape.node.iteritems()
        # )
        # print max_cost
        #
        # opt_policy_nodes = [best_node]
        # fields = []
        #
        # while True:
        #     current_node = self.landscape[opt_policy_nodes[-1]]
        #
        #     if len(current_node) == 0:
        #         break
        #
        #     current_node.pop()
        #
        #     opt_policy.append()
        print max(
             (prop['cost_func'], prop['iteration']) for node, prop in self.landscape.node.iteritems()
        )

###################################################################################################
#
#   Test
#
###################################################################################################

if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np

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
    #
    #
    ###############################################################################################

    init_state = np.zeros(N)
    init_state[0] = 1.

    opt = DyProgToolbox(
            init_state,
            field.min(),
            CPropagator(),
            field_switching,
            CCostFunc(),
            2000
    )

    for _ in xrange(11):
       opt.next_time_step()

    opt.get_optimal_policy()
    # nx.drawing.nx_pydot.to_pydot(opt.landscape).write_png('test.png')

    nx.draw(opt.landscape, pos=opt.get_pos_iteration_cost(), node_color=opt.get_node_color())
    plt.show()