__author__ = 'victor'

"""
Just defines several utility functions used by gt_to_pomdp.
"""

def flatten_tuple(t):
    """
    Flattens a 2d tuple - or returns t if it is already flat.
    :param t:
    :return:
    """
    flattened = t
    try:
        flattened[0][0] #check for multiple levels
    except IndexError:
        return flattened

    #flatten
    flattened = [element for tupl in flattened for element in tupl]

    return flattened

def to_tuple(t):
    """
    Takes a parameter t and returns it as a tuple - if t is already a tuple, returns t. Otherwise returns (t,)
    :param t:
    :return:
    """
    if type(t) is tuple:
        return t
    return (t,)

