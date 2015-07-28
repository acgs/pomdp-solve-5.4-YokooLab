# encoding: utf-8

import argparse
from gt_to_pomdp.models import *



def main(inputfilename, outputfilename=None, policygraph=None, verbose=False):
    gt = GTModel(inputfilename)
    if verbose:
        print(gt)
    ppomdp = PseudoPOMDPModel(gt)
    if verbose:
        print("PseudoPOMDP")
        print(ppomdp)
    pomdp = POMDPModel(ppomdp)
    if verbose:
        print("POMDP")
        print(pomdp)
        print(pomdp.to_Cassandra_format())
    if outputfilename is not None:
        with open(outputfilename, 'w') as f:
            f.write(pomdp.to_Cassandra_format())
    else:
        print(pomdp.to_Cassandra_format())

    V,A = pomdp.to_value_function(pomdp.players[0])
    print([(v,a) for v,a in zip(V,A)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses a Game Theory model and converts it to a POMDP model.')
    parser.add_argument('gtmodel', type=str, help='The input file name for the Game Theory model.')
    parser.add_argument('-pomdpmodel', type=str, help='The output file name for the POMDP model.', default=None)
    parser.add_argument('-policygraph', type=str, help='The policy graph (pre-FSA) to convert to a value function.', default=None)
    parser.add_argument('-verbose', type=bool, help='Verbosity of output. If true, will output in verbose mode.', default=False)

    args = parser.parse_args()
    print(args.verbose)
    main(args.gtmodel, args.pomdpmodel, args.policygraph, args.verbose)
