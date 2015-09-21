import argparse
from finite_grid_generation.utils import StateTransitionProbability, ObservationTransitionProbability

__author__ = 'Victor Szczepanski'

"""
Main program for finite grid generation.

Offers command line interface to different finite grid generation methods.

See Also::
    GenerationHeuristics
"""

from finite_grid_generation.generationheuristics import ReachableBeliefs, GenerationHeuristicsFactory

from gt_to_pomdp.gt_to_pomdp import main as gt_translator
from gt_to_pomdp.models import POMDPModel


def main(model_filename, outfilename=None, strategy=None, verbose=False, initial_belief=(), num_points=100, truncate=False):
    # first parse model
    if verbose:
        print("Translating game theory to pomdp...")
    pomdp = gt_translator(model_filename, verbose=verbose)
    """:type : POMDPModel"""

    if verbose:
        print("Done translating. Selecting finite grid strategy {}".format(strategy))

    # if strategy is None, use default heuristic

    if strategy is None:
        if verbose:
            print("Using default strategy 'ReachableBeliefs'.")
        generation_method = ReachableBeliefs(states=pomdp.states,
                                             actions=pomdp.actions,
                                             initial_belief_state=initial_belief,
                                             observations=pomdp.observations,
                                             state_transition_function=StateTransitionProbability(pomdp.state_transition),
                                             observation_transition_function=ObservationTransitionProbability(pomdp.observation_probability))
    else:
        generation_method = GenerationHeuristicsFactory().make_generation_heuristic(strategy,
                                                                                    states=pomdp.states,
                                                                                    actions=pomdp.actions,
                                                                                    initial_belief_state=initial_belief,
                                                                                    observations=pomdp.observations,
                                                                                    state_transition_function=pomdp.state_transition,
                                                                                    observation_transition_function=pomdp.observation_probability)

    if verbose:
        print("Generating grid...")

    grid = generation_method.generate_grid(num_points=num_points, truncate=truncate, verbose=verbose)

    if verbose:
        print("Done generating grid. Generated {} points.".format(len(grid)))

    print(grid)


    if outfilename is not None:

        if verbose:
            print("Pretty printing grid to file in Cassandra format...")
        with open(outfilename, mode='w') as f:
            for belief in grid:
                for dim in belief:
                    f.write('{} '.format(dim))
                f.write('\n')
        if verbose:
            print("Done printing.")

    if verbose:
        print("Exiting...")
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parses a POMDP model and generates a finite grid of belief states.")
    parser.add_argument('gtmodel', type=str, help='The game theory model to parse')
    parser.add_argument('-gridfile', type=str, help='The filename to pretty print the grid to in Cassandra format.', default=None)
    parser.add_argument('-initial_belief', type=float, nargs='+',
                        help='The initial belief state to use for some strategies', default=())
    parser.add_argument('-strategy', type=str, help='The generation strategy.'
                        ' May be one of {}'.format(GenerationHeuristicsFactory().heuristics), default=None)
    parser.add_argument('-verbose', type=bool, help='Verbosity of output.', default=False)
    parser.add_argument('-num_points', type=int, help='Number of points to generate of the finite grid.', default=100)
    parser.add_argument('-truncate', type=bool, help='', default=False)

    args = parser.parse_args()

    main(model_filename=args.gtmodel, outfilename=args.gridfile, strategy=args.strategy, verbose=args.verbose, initial_belief=args.initial_belief, num_points=args.num_points, truncate=args.truncate)