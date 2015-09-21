__author__ = 'Victor Szczepanski'

class StateTransitionProbability(object):
    """
    Abstract accessing the state transition probability table.

    Args:
        transition_table (list[(str, str, str, float)]): A mapping from current state to action to next state to probability.

    """

    def __init__(self, transition_table):
        self.transition_table = {}
        for transition_tuple in transition_table:

            next_state, transition, probability = transition_tuple
            current_state, action = transition

            if current_state not in self.transition_table:
                self.transition_table[current_state] = {}

            if next_state not in self.transition_table[current_state]:
                self.transition_table[current_state][next_state] = {}

            self.transition_table[current_state][next_state][action] = float(probability)

    def T(self, s_prime, s, a):
        """
        Return the probability of transitioning to s_prime from s, given action a
        Args:
            s_prime (str): The next state
            s (str): The current state
            a (str): The action

        Returns:
            float: The probability of transitioning from `s` to `s_prime`, given action `a`.
        """
        return self.transition_table[s][s_prime][a]

class ObservationTransitionProbability(object):
    """
    Abstract accessing the observation probability table.

    Args:
        observation_table (dict[str, dict[str, dict[str, float]]]): A mapping from next state to action to observation to probability

    """

    def __init__(self, observation_table):
        self.observation_table = observation_table
        print(self.observation_table)

    def O(self, o, s_prime, a):
        """

        Returns:
            float: the probability of observing `o` given action `a` and state `s_prime`
        """

        return float(self.observation_table[o][(a, s_prime)])