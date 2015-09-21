# encoding: utf-8
"""Generate a Cassandra POMDP and Value Function from a Shun Game Theory Model.

This module defines the main entry point to the Game Theory to POMDP conversion program.
It is built to be used from the command line, but can just as easily be imported by other software.

.. seealso::
    Module `GTModel`
        The object that represents a Game Theory Model
    Module `PseudoPOMDPModel`
        The object that represents the intermediate POMDP model, converted from a GTModel.
    Module `POMDPModel`
        The object that represents the final POMDP model, converted from a PseudoPOMDPModel.

Examples:
    This software is not compatible with python 2, so all examples specify python3. Of course, if
    python 3 is the only python on your system, then the command may be run with just python.

    To have the conversion output to stdout, simply run this module with a Game Theory model::
        $ python3 gt_to_pomdp.py example.dat

    There are two optional command line arguments: -pomdpmodel and -verbose.

    pomdpmodel specifies the output file name for the POMDP model

        $ python3 gt_to_pomdp.py example.dat -pomdpmodel example_pomdp.POMDP

    -verbose enables verbose output of intermediate conversions on stdout:
        $ python3 gt_to_pomdp.py example.dat -verbose True

    Of course, they may be combined to show intermediate conversions on stdout and output the POMDP to a file:
        $ python3 gt_to_pomdp.py example.dat -pomdpmodel example_pomdp.POMDP -verbose True

The output POMDP (either to stdout or to file) is in Cassandra format,
so it may be passed directly to the pomdp-solve software, written by Cassandra.

This software makes no guarantee of the correctness of the pomdp-solve program.
The output of gt-to-pomdp is a valid pomdp-solve input as of 7/31/2015. Future versions of pomdp-solve may
break this format.
"""


import argparse
from gt_to_pomdp.models import *


def main(inputfilename, outputfilename=None, verbose=False):
    """Parse a Shun Game Theory Model.

    ..seealso::
        Class `GTModel`
            The object that represents a Game Theory Model. Documentaion describes the Shun Game Theory format.

    Parse a plaintext file `inputfilename` into a GTModel and convert it to a POMDP.
    Show intermediate steps if `verbose` is true.
    Output POMDP to `outputfilename`.

    Args:
        inputfilename (str): the path (may be relative) to a Shun formatted Game Theory file.
        outputfilename (Optional[str]): the path (may be relative) to output the POMDP to.
        verbose (bool): whether to output intermediate conversions to stdout.

    """
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

    # We use capitals here as it is the convention of the POMDP literature.
    V, A = pomdp.to_value_function(pomdp.players[0])
    print(pomdp.value_function_to_Cassandra_format(V, A))

    return pomdp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parses a Game Theory model and converts it to a POMDP model.')
    parser.add_argument('gtmodel', type=str, help='The input file name for the Game Theory model.')
    parser.add_argument('-pomdpmodel', type=str, help='The output file name for the POMDP model.', default=None)
    parser.add_argument('-verbose', type=bool, help='Verbosity of output.'
                                                    ' If true, will output in verbose mode.', default=False)

    args = parser.parse_args()
    print(args.verbose)
    main(args.gtmodel, args.pomdpmodel, args.verbose)
