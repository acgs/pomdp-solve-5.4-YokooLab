# encoding: utf-8

import argparse
from gt_to_pomdp.models import *



def main(inputfilename, outputfilename=None, policygraph=None):
    gt = GTModel(inputfilename)
    print(gt)
    ppomdp = PseudoPOMDPModel(gt)
    print("PseudoPOMDP")
    print(ppomdp)
    pomdp = POMDPModel(ppomdp)
    print("POMDP")
    print(pomdp)
    print(pomdp.to_Cassandra_format())
    if outputfilename is not None:
        with open(outputfilename, 'w') as f:
            f.write(pomdp.to_Cassandra_format())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses a Game Theory model and converts it to a POMDP model.')
    parser.add_argument('gtmodel', type=str, help='The input file name for the Game Theory model.')
    parser.add_argument('pomdpmodel', type=str, help='The output file name for the POMDP model.', default=None, nargs='?')
    parser.add_argument('policygraph', type=str, help='The policy graph (pre-FSA) to convert to a value function.', default=None, nargs='?')

    args = parser.parse_args()
    main(args.gtmodel, args.pomdpmodel, args.policygraph)