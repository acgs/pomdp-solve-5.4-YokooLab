# encoding: utf-8

import argparse
import operator #used for reshaping a matrix
import functools
import itertools


class FSA(object):
    """Describes an FSA - finite state automata - as its states and its transitions. Note that this is really more like a
    pre-FSA, since we won't specify a starting state"""
    def __init__(self, states=[], observations=[], transitions={}):
        self.states = states
        self.observations = observations
        self.transitions = transitions
        """
        The transition function maps a state and an observation to another state - it is a 2 level dictionary
        where level 1 is a state and level 2 is an observation.
        """


def flatten_tuple(t):
    """
    Flattens a 2d tuple - or returns t if it is already flat.
    :param t:
    :return:
    """
    flattened = t
    print("Flattening tuple {}".format(t))
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

class Player(object):
    """
    A Player represents an FSA - finite state automaton.
    """

    def __init__(self, lines=None):
        self.name = ""
        self.states = []
        self.actions = []
        self.signals = []
        self.state_machine = {} #maps a state to an action
        self.state_transitions = {} #2 level dictionary that maps a state to a signal to a state.

        if lines is not None:
            p = Player.from_lines(lines)
            self.name = p.name
            self.states = p.states
            self.actions = p.actions
            self.signals = p.signals
            self.state_machine = p.state_machine
            self.state_transitions = p.state_transitions

    def join(self, other_player=None):
        """
        Joins this player's FSA with another player's FSA.
        If `other_player` is None, returns this Player - it is idempotent.
        :param other_player:
        :return:
        """
        if other_player is None:
            return self

        joint_player = Player()

        joint_player.name = self.name + other_player.name
        #We create joint states by doing the cartesian product of this player's states and the other player's states.

        for my_state in self.states:
            m = to_tuple(my_state)

            for other_state in other_player.states:
                o = to_tuple(other_state)

                joint_player.states.append(m + o) #will just append other_state to it.

        for my_action in self.actions:
            m = to_tuple(my_action)

            for other_action in other_player.actions:
                o = to_tuple(other_action)

                joint_player.actions.append(m + o) #will just append other_state to it.

        for my_signal in self.signals:
            m = to_tuple(my_signal)

            for other_signal in other_player.signals:
                o = to_tuple(other_signal)

                joint_player.signals.append(m + o) #will just append other_state to it.

        # The new state machine maps a set of states to a set of actions

        for state in joint_player.states:
            #state could be split into many 'substates' - but we know the last one is from the other player.
            my_state = state[:-1]
            if len(my_state) == 1:
                my_state = my_state[0]
            their_state = state[-1]

            m = to_tuple(my_state)
            t = to_tuple(their_state)

            joint_player.state_machine[m + t] = to_tuple(self.state_machine[my_state]) +  to_tuple(other_player.state_machine[their_state])

        #The new state transitions maps a set of 2 level state/signals to a set of states.
        for state in joint_player.states:

            my_state = state[:-1]
            if len(my_state) == 1:
                my_state = my_state[0]

            my_state_tuple = to_tuple(my_state)

            their_state = state[-1]
            t = to_tuple(their_state)

            for signal in joint_player.signals:

                my_signal = signal[:-1]

                if len(my_signal) == 1:
                    my_signal = my_signal[0]

                m_tuple = to_tuple(my_signal)


                their_signal = signal[-1]
                o = their_signal
                if type(their_signal) is not tuple and type(m_tuple) is tuple:
                    o = (their_signal,)

                if (my_state, their_state) not in joint_player.state_transitions:
                    joint_player.state_transitions[my_state_tuple + t] = {}

                my_transition = to_tuple(self.state_transitions[my_state][my_signal])


                their_transition = to_tuple(other_player.state_transitions[their_state][their_signal])

                joint_player.state_transitions[my_state_tuple + t][m_tuple + o] = \
                    (my_transition + their_transition)

        return joint_player

    @staticmethod
    def from_lines(lines):
        """
        Parses a Player configuration from a set of strings and returns a Player.
        :param lines:
        :return:
        """

        #parsing state machine
        #states: name, states, actions, signals, state_machine, state_transitions, end
        #we use integers for faster comparison: 1, 2, 3, 4, 5, 6, 7

        player = Player()

        state = 1

        for line in lines:
            #ignore all blank lines
            if len(line.strip()) is 0:
                continue
            if state is 1: #name
                player.name = line.split()[1]
                state = 2
            elif state is 2: #states
                player.states = line[line.index(':')+1:].lstrip().rstrip().split()
                state = 3
            elif state is 3: #actions
                player.actions = line[line.index(':')+1:].lstrip().rstrip().split()
                state = 4
            elif state is 4: #signals
                player.signals = line[line.index(':')+1:].lstrip().rstrip().split()
                state = 5
            elif state is 5: #state_machine
                state_action_pair = line.lstrip().rstrip().split()
                if len(state_action_pair) > 2:
                    state = 6
                else:
                    player.state_machine[state_action_pair[0].lstrip().rstrip()] = state_action_pair[1].lstrip().rstrip()

            if state is 6:
                state1, signal, state2 = line.split()
                if state1 not in player.state_transitions:
                    player.state_transitions[state1] = {}

                player.state_transitions[state1][signal] = state2

        return player

    def __str__(self):
        s = []
        s.append('Automaton {}'.format(str(self.name)))
        s.append('States: {}'.format(' '.join(flatten_tuple(self.states))))
        s.append('Actions: {}'.format(' '.join(flatten_tuple(self.actions))))
        s.append('Signals: {}'.format(' '.join(flatten_tuple(self.signals))))
        s.append('\n'.join(['{} {}'.format(key, value) for key,value in self.state_machine.items()]))
        for state_transition in self.state_transitions:
            s.append(str(state_transition) + ' '.join(str(self.state_transitions[state_transition])))
        return '\n'.join(s)


def probability_lookup(signal_distribution, players, observation, action_profile):
    """
    using the provided observation (set of signals) and action_profile (set of actions), returns the probability
    of observation given the action_profile.

    Assumes that action_profile is sorted by player (e.g. action_profile[0] is the action for player1).

    I.e. o_1 (ω_1 | (a_1 , f (θ^t ))). where (a_1 , f (θ^t ) is `action_profile` and  ω_1 is `observation`
    """
    #with the action profile, we can look up the probability table for that profile
    probability_table = signal_distribution[action_profile]

    #indexes in signal_distribution are based on the index of signals. We know the signals of other players from observation, but need to get 'our' signal
    player1_signal_index = players[0].actions.index(action_profile[0])

    # we'll collect the indicies of each other player's signal to do a lookup in the signal_distribution table.
    indicies = [player1_signal_index]
    print(observation)
    for index, obs in enumerate(observation):
        if index is 0:
            continue

        indicies.append(players[index].signals.index(obs))

    #now we can look up the value using indicies
    probability = probability_table #We'll 'reduce' probability table down to one value.
    print("Probability table: {}".format(probability))
    print("Indicies: {}".format(indicies))
    for index in indicies:
        probability = probability[index]

    return probability



class GTModel(object):


    def __init__(self, filename=None):
        self.title = ""
        self.discount = 0.0

        self.variables = {}
        """
        Dictionary that maps a variable to its value.
        """

        self.player_names = []
        """
        List of strings of player names
        """

        self.players = []
        """
        List of a player objects.
        """

        self.signal_distribution = {}
        """
        Dictionary that maps a tuple of actions (action profile) to an N dimensional matrix (list of lists) where N
         is the number of players, and each dimension is length s, where s is the number of signals (observations)
         that player can have.
        """

        self.payoff = {}
        """
        Dictionary that maps a tuple of actions (action profile) to a dictionary that maps player names to values (floats)
        """


        if filename is not None:
            model = GTModel.from_file(filename)

            self.title = model.title
            self.discount = model.discount
            self.variables = model.variables
            self.player_names = model.player_names
            self.players = model.players
            self.signal_distribution = model.signal_distribution
            self.payoff = model.payoff


    @staticmethod
    def from_file(filename):
        """
        Constructs a new GTModel from a file.
        :param filename:
        :return:
        """
        model = GTModel()


        #state machine for parsing.
        #States: title, discount, variables, player_names, player, signal, payoff, end
        #Transitions (after reading one line):
        #   title -> discount
        #   discount -> variables
        #   variables -> player_names
        #   player_names -> player
        #   player -> player
        #   player ->('Signal Distribution') signal
        #   signal -> signal
        #   signal -> ('Payoff Matrix') payoff
        #   payoff -> payoff
        #   payoff ->(empty) end
        #   end -> end #absorbs all additional input

        #We'll just use integers for faster comparison, so each state is a number 1,2,3,4,5,6,7, or 8.


        state = 1

        #player_lines collects lines to pass to the player constructor
        player_lines = []
        signal_lines = []
        payoff_lines = []

        with open(filename, 'r') as f:
            for line in f.readlines():
                if state is 1: #title
                    model.title = line[line.index(':')+1:].rstrip()
                    state = 2
                elif state is 2: #discount
                    model.discount = float(line[line.index(':')+1:].rstrip().lstrip())
                    state = 3
                elif state is 3: #variables
                    model.variables = GTModel._parse_variables(line)
                    state = 4
                elif state is 4: #player_names
                    model.player_names = [name for name in line[line.index(':')+1:].split() if name != '' and name != ' ']
                    state = 5
                elif state is 5: #player
                    if 'Signal Distribution' in line:
                        state = 6
                        continue

                    if len(player_lines) is 0 and len(line.strip()) is 0:
                        continue
                    elif len(line.strip()) is 0:
                        # build new player from collected strings and reset player_lines
                        new_player = Player(player_lines)
                        model.players.append(new_player)
                        player_lines = []
                    else:
                        player_lines.append(line.rstrip().lstrip())
                elif state is 6: #signal
                    if 'Payoff Matrix' in line:
                        state = 7
                        continue
                    if len(line.strip()) is 0:
                        #build new signal distribution matrix from collected strings and clear signal_lines
                        matrix = GTModel._parse_matrix(signal_lines, model.variables)
                        signal_lines = []


                        #we want to reshape each matrix based on the signals/observations of each player.
                        observation_sizes = [len(player.signals) for player in model.players]

                        for key in matrix:
                            matrix[key] = GTModel._shape(matrix[key], observation_sizes)

                        model.signal_distribution = matrix

                    else:
                        signal_lines.append(line)
                elif state is 7: #payoff
                    if len(line.strip()) is 0:
                        state = 8
                        continue
                    payoff_lines.append(line) #We hold off on parsing payoff_lines until end of file.
                elif state is 8: #end
                    continue

        #build new payoff matrix
        matrix = GTModel._parse_matrix(payoff_lines, model.variables)

        #We don't need to reshape payoff matrix, since values are always 1-D
        model.payoff = matrix

        return model

    @staticmethod
    def _parse_variables(line):
        """
        Takes a variables line formated q=0.001 s=0.001 g=0.3 ...
        and parses it into a dictionary as described by GTModel.variables
        :param line:
        :return variables: a dictionary that maps string variable names to float values
        """
        variables = {}
        vars = line.split()
        for assignment in vars:
            if '=' in assignment:
                split_assignment = assignment.split('=')
                variables[split_assignment[0]] = float(split_assignment[1])

        return variables


    @staticmethod
    def _parse_matrix(lines, variables):
        """
        Takes a collection of lines of the form:
        C,C : 1-4*s-4*q q q q s s q s s
        C,D : q s s 1-4*s-4*q q q q s s
        C,E : q s s q s s 1-4*s-4*q q q
        D,C : q 1-4*s-4*q q s q s s q s
        D,D : s q s q 1-4*s-4*q q s q s
        D,E : s q s s q s q 1-4*s-4*q q
        E,C : q q 1-4*s-4*q s s q s s q
        E,D : s s q q q 1-4*s-4*q s s q
        E,E : s s q s s q q q 1-4*s-4*q

        or

        C,C: 1 1
        C,D: -l 1+g
        C,E: 1-b 1
        D,C: 1+g -l
        D,D: 0 0
        D,E: 1+g-b x
        E,C: 1 1-b
        E,D: x 1+g-b
        E,E: 1-b 1-b

        and builds a dictionary that maps the action tuples to flat matricies of values, where the values are converted based on
        the input variables.

        Assumes any variable encountered in lines can be found in variables' keyset.
        :param lines: list of strings representing the matrix
        :param variables: dictionary that maps a variable name to a floating point value.
        :return mapping: dictionary that maps an action tuple (action profile) to a flat matrix of floating point values.
        """
        mapping = {}
        for line in lines:
            actions = tuple(line[:line.index(':')].rstrip().lstrip().split(',')) #make tuple of actions
            values = line[line.index(':')+1:].rstrip().lstrip().split() #make list of value expressions

            #replace all variables in each expression with values
            for i in range(len(values)):
                values[i] = ''.join([str(variables[c]) if c in variables else c for c in values[i]])

                values[i] = eval(values[i]) #actually evaluate any expressions

            mapping[actions] = values

        return mapping



    @staticmethod
    def _shape(flat, dims):
        subdims = dims[1:]
        subsize = functools.reduce(operator.mul, subdims, 1)
        if dims[0]*subsize!=len(flat):
            raise ValueError("Size does not match or invalid")
        if not subdims:
            return flat
        return [GTModel._shape(flat[i:i+subsize], subdims) for i in range(0,len(flat), subsize)]

    def __str__(self):
        s = []
        s.append('Title: {}'.format(self.title))
        s.append('Discount Rate: {}'.format(self.discount))
        s.append('Variables: {}'.format(self.variables))
        s.append('Players: {}'.format(self.player_names))
        s.append('\n')
        for player in self.players:
            s.append(str(player))
            s.append('\n')
        s.append('Signal Distribution')
        s.append(str(self.signal_distribution))
        s.append('\n')
        s.append('Payoff Matrix')
        s.append(str(self.payoff))
        return '\n'.join(s)


class PseudoPOMDPModel(object):
    """
    Describes a PseudoPOMDP model, as defined in Automated Equilibrium Analysis of Repeated Games with
Private Monitoring: A POMDP Approach by YongJoon Joe.
    """

    def __init__(self, gt_model=None):
        self.title = ''
        self.discount = 0.0
        self.gt = None
        self.states = set()  # A set of states of other players (player 2 in a 2 player game)
        self.actions = set()  # A set of actions for player 1
        self.observations = set()  # A set of observations/signals of player 1.
        # Note that this is an observation of the entire world - that is, the joint observations of all other players.


        self.observation_probability = {}  # A function that maps an observation given an action/state tuple to a probability
        self.state_transition = {}  # A function that represents the conditional probability that the next state
        # is θ^t+1 when the current state is θ^t and the action of player 1 is a_1
        self.payoff = {}  # A function that maps an action/state tuple to a real value.

        #Additional data to help translation
        self.players = []
        self.signal_distribution = []

        if gt_model is not None:
            self.from_gt(gt_model)

    def from_gt(self, gt_model):
        """
        Translates a GTModel into this PseudoPOMDPModel. Sets all of this model's attributes.
        :param gt_model: the GTModel to translate from.
        :return:
        """
        self.title = gt_model.title
        self.discount = gt_model.discount
        self.gt = gt_model
        # states are the cartesian product of all states in gt_model of players except player 1

        self.players = gt_model.players
        self.signal_distribution = gt_model.signal_distribution

        player1 = gt_model.players[0]

        #We need to build a joint-FSA from all other players
        joint_player = gt_model.players[1]
        for player in gt_model.players:

            joint_player = joint_player.join(player)

        print(joint_player)

        states = [player.states for player in gt_model.players if player is not player1]

        self.states = [s for s in itertools.product(*states)]

        # self.states is a list of tuples (maybe of length 1)
        # actions are the set of actions of player 1. Again, if all players use FSA M, we can pick any player.
        self.actions = set(player1.actions)

        # observations are the set of observations of player 1.
        self.observations = set(player1.signals)

        # Observation probability maps an observation given an action/state tuple to a probability
        # We represent this as a tuple (observation, (action, state), probability)
        # Then there are |observations| x |actions| x |states| many entries

        self.observation_probability = {}
        action_state_tuples = [(action_state[0], action_state[1]) for action_state in itertools.product(self.actions, self.states)]

        #Note that each action_state_tuple in action_state_tuples is a tuple (maybe of length 1)

        observation_profiles = [observation_profile for observation_profile in itertools.product(self.observations, repeat=len(gt_model.players))]

        for observation in observation_profiles:
            self.observation_probability[observation] = {}

        for (action, state) in action_state_tuples:

            # we'll loop over each state to find the action profile (a_1, a_2, ...)
            action_profile = self._to_action_profile( state, action)

            for observation in observation_profiles: #gets the combinations of observations possible.
                # O(ω_1 | a_1 , θ^t ) = o_1 (ω_1 | (a_1 , f (θ^t ))).

                probability = probability_lookup(gt_model.signal_distribution, self.players, observation, action_profile)

                #now we can make the tuple
                self.observation_probability[observation][action_profile] = probability
                #self.observation_probability[(observation, action, state)] = probability

        # state_transition function P (θ^t+1 | θ^t , a_1 ) represents the conditional probability that
        # the next state is θ^t+1 when the current state is θ^t and the
        # action of player 1 is a_1
        # P (θ^t+1 | θ^t , a_1 ) = sum_{ω_2 in Omega | T(θ t, ω_2) = θ t+1}  o_2 (ω_2 | (a_1 , f (θ^t ))).
        # So, we make a tuple (state2, (state1, action), value), where state2 =  θ^t+1 and state1 = θ^t
        # Note that ω_2 is the observation of player 2

        #initialize self.state_transition:
        for state1 in self.states:
            self.state_transition[state1] = {}
            for state2 in self.states:
                self.state_transition[state1][state2] = {}

        for (action, state1) in action_state_tuples:


            # we'll loop over each state to find the action profile (a_1, a_2, ...)
            action_profile = self._to_action_profile(state1, action)

            for state2 in self.states: #state2 is θ^t+1

                # now loop over all observations to do summation
                # sum_{ω_2 in Omega | T(θ t, ω_2) = θ t+1}  o_2 (ω_2 | (a_1 , f (θ^t ))).
                #TODO: What do we do when there are more than 2 players? Multiply the sums of probabilities?


                #For right now, assume we only have 2 players. Must abstract this somehow to n players
                probability = 0

                for observation in gt_model.players[1].signals:

                    probability += probability_lookup(gt_model.signal_distribution, self.players, observation, action_profile)[1] if (state1[0], observation, state2[0]) in gt_model.players[1].state_transitions else 0



                #probability_tuple = (state2, (state1, action), probability)
                self.state_transition[state1][state2][action] = probability


                # probability = 0
                # for obs in observation_profiles:
                #     observation = obs[0]
                #
                #     player_probability = 0
                #
                #     print("Looking up transition probabilities for pairs {} and observation {}".format([(index, (s1, s2)) for (index, (s1,s2)) in enumerate(zip(state1, state2))], observation))
                #
                #     for index, (s1, s2) in enumerate(zip(state1, state2)): #We iterate over pairs of single states for each player
                #         i = index + 1
                #
                #         if (s1, observation, s2) not in gt_model.players[i].state_transitions:
                #             continue  # ignore any observations that do not take us from state1 to state2 for player 1.
                #
                #         print("****\nGot probability of observation {} given action profile {} for player {} as {}\n****".format(observation, action_profile, i+1, gt_model.probability_lookup(observation, action_profile)[i]))
                #
                #         player_probability += gt_model.probability_lookup(observation, action_profile)[i]
                #
                #     probability += player_probability # For now, just add each players' probabilities. This only makes sense for 2 players, though.
                #
                # #now we can make the tuple
                # probability_tuple = (state2, (state1, action), probability)
                # self.state_transition.append(probability_tuple)

        # payoff R : A × S → R is given as:
        # R(a_1 , θ^t ) = g_1 ((a_1 , f (θ^t ))).
        # g_i (a) = sum_{ω∈Ω^2} π_i (a_i , ω_i )o(ω | a)
        # So, we make a tuple (action, state, payoff) where action = a_1, state = θ^t, and payoff is a real value (floating point)
        for (action, state) in action_state_tuples:
            action_profile = self._to_action_profile(state, action)
            payoff_matrix = gt_model.payoff[action_profile]

            payoff = 0
            for observation in observation_profiles:
                probability = probability_lookup(gt_model.signal_distribution, self.players, observation, action_profile)
                #TODO: Check if this is correct. The GT model doesn't seem to provide realized payoff related to a player's observation.
                payoff += payoff_matrix[0] * probability  # π_i (a_i , ω_i )o(ω | a)

            self.payoff[(action, state)] = payoff



    def _to_action_profile(self, state, action):
        #print("In action profile with state {} and action {}".format(state, action))
        action_profile = [action]
        for i in range(len(state)):
            #print("Looking up action for player {}.".format(i+1))
            other_action = self.players[i+1].state_machine[state[i]]
            action_profile.append(other_action)
        action_profile = tuple([action for action in action_profile])

        return action_profile



    def __str__(self):
        """
        self.states = set()  # A set of states of other players (player 2 in a 2 player game)
        self.actions = set()  # A set of actions for player 1
        self.observations = set()  # A set of observations/signals of player 1.
        # Note that this is an observation of the entire world - that is, the joint observations of all other players.


        self.observation_probability = []  # A function that maps an observation given an action/state tuple to a probability
        self.state_transition = []  # A function that represents the conditional probability that the next state
        # is θ^t+1 when the current state is θ^t and the action of player 1 is a_1
        self.payoff = []  # A function that maps an action/state tuple to a real value.
        :return:
        """
        s = []
        s.append('States: {}'.format(self.states))
        s.append('Actions: {}'.format(self.actions))
        s.append(('Observations: {}'.format(self.observations)))
        s.append('Observation probabilities: {}'.format(self.observation_probability))
        s.append('State Transition probabilities: {}'.format(self.state_transition))
        s.append('Payoffs: {}'.format(self.payoff))
        return '\n'.join(s)

class POMDPModel(object):

    def __init__(self, pseudo_pomdp_model=None):
        self.title = ''
        self.discount = 0.0
        self.gt = None
        self.states = set()  # A set of states of other players (player 2 in a 2 player game)
        self.actions = set()  # A set of actions for player 1
        self.observations = set()  # A set of observations/signals of player 1.
        # Note that this is an observation of the entire world - that is, the joint observations of all other players.


        self.observation_probability = {}  # A function that maps an observation given an action/state tuple to a probability
        self.state_transition = []  # A function that represents the conditional probability that the next state
        # is θ^t+1 when the current state is θ^t and the action of player 1 is a_1
        self.payoff = {}  # A function that maps an action/state tuple to a real value.
        self.players = []

        if pseudo_pomdp_model is not None:
            self.from_pseudo_pomdp(pseudo_pomdp_model)

    def from_pseudo_pomdp(self, pseudo_pomdp_model):
        self.title = pseudo_pomdp_model.title
        self.discount = pseudo_pomdp_model.discount
        self.gt = pseudo_pomdp_model.gt

        self.players = pseudo_pomdp_model.players

        # actions and observations are identical
        self.actions = pseudo_pomdp_model.actions
        self.observations = pseudo_pomdp_model.observations

        # The key idea of this translation is to introduce a set of new
        # combined states Θ' , where Θ' = Θ^2 . Namely, we assume
        # that a state θ'^t in the standard POMDP model represents
        # the combination of the previous and current states (θ^t−1 , θ^t )
        # in our model present in the previous subsection.
        self.states = [s for s in itertools.product(pseudo_pomdp_model.states, repeat=2)]
        #remove any unreachable states (i.e. pairs that do not exist in the state_transition function of pseudo_pomdp_model.

        removed_states = []
        for state in self.states:
            try:
                _ = pseudo_pomdp_model.state_transition[state[0]][state[1]]
            except KeyError:
                removed_states.append(state)
                print(removed_states)

        for state in removed_states:
            self.states.remove(state)

        # A new state transition function P' (θ'^t+1 | θ'^t , a_1 ) is equal
        # to P (θ^t+1 | θ^t , a_1 ) in the original model if θ'^t+1 = (θ^t , θ^t+1 )
        # and θ'^t = (θ^t−1 , θ^t ), i.e., the previous state in θ'^t+1 and the
        # current state in θ'^t are identical. Otherwise, it is 0.

        for action, theta_t_plus_one, theta_t in itertools.product(self.actions, self.states, self.states):
            probability = 0
            if theta_t_plus_one[0] is theta_t[1]:

                probability = pseudo_pomdp_model.state_transition[theta_t_plus_one[0]][theta_t_plus_one[1]][action]

                """for state_transition in pseudo_pomdp_model.state_transition:
                    if state_transition[0] is theta_t[1] and state_transition[1][0] is theta_t[0] and\
                        state_transition[1][1] is action:
                            probability = state_transition[2]"""
            self.state_transition.append((theta_t_plus_one, (theta_t, action), probability))


        """
        for action, theta_t_plus_one, theta_t in itertools.product(self.actions, self.states, self.states):
            probability = 0
            if theta_t_plus_one[0] is theta_t[1]:
                for state_transition in pseudo_pomdp_model.state_transition:
                    if state_transition[0] is theta_t[1] and state_transition[1][0] is theta_t[0] and\
                        state_transition[1][1] is action:
                            probability = state_transition[2]
            self.state_transition.append((theta_t_plus_one, (theta_t, action), probability))
        """


        # Next,
        # let us examine how to define O'(ω_1 | a_1 , (θ^t , θ^t+1 )). This
        # is identical to the posterior probability that the observation
        # was ω 1 , when the state transits from θ t to θ t+1 . Thus, this
        # is defined as:
        #     O'(ω_1 | a_1 , (θ^t , θ^t+1 )) =
        #         (sum_{ω_2 ∈Ω'} O(ω_1 , ω_2 | (a_1 , f (θ^t ))) )
        #         ----------------------------------------------
        #         (sum_{ω∈Ω} sum_{ω_2 ∈Ω'} O(ω, ω_2 | (a_1 , f (θ^t ))))
        # ,
        # where Ω' = {ω 2 | T (θ t , ω 2 ) = θ t+1 }

        #Initialize observation_probability table
        for observation in itertools.product(self.observations,repeat=len(self.gt.players)):
            self.observation_probability[observation] = {}

            print("Observation: {}".format(observation))

        for observation1, action, theta_prime in itertools.product(itertools.product(self.observations,repeat=len(self.gt.players)), self.actions, self.states):
            theta_t = theta_prime[0]
            theta_t_plus_one = theta_prime[1]
            upper = 0
            lower = 0
            print(observation1)

            omega_prime = set()#[omega_2 for (t, omega_2, t1) in self.players[1].state_transitions] # state ->signal state

            for state in self.players[1].state_transitions:
                for obs in self.players[1].state_transitions[state]:
                    s = to_tuple(state)
                    o = to_tuple(obs)

                    if theta_t == s and theta_t_plus_one == to_tuple(self.players[1].state_transitions[state][obs]):
                        omega_prime.add(o)



            print("omega prime: {}".format(omega_prime))

            # (sum_{ω_2 ∈Ω} O(ω_1 , ω_2 | (a_1 , f (θ^t ))) )
            # (observation, (action, state), probability)
            # Since it is O(ω_1 , ω_2 | (a_1 , f (θ^t ))) ), we will split it to o_1(ω_1 | a_1, f(θ^t )) x o_2(ω_2 | a_1, f(θ^t ))
            for observation2 in omega_prime:

                obs2 = probability_lookup(self.gt.signal_distribution, self.gt.players, observation2,
                                          pseudo_pomdp_model._to_action_profile(theta_t, action))
                obs1 = probability_lookup(self.gt.signal_distribution, self.gt.players, observation1,
                                          pseudo_pomdp_model._to_action_profile(theta_t, action))
                print("obs2: {}, obs1: {}".format(obs2, obs1))

                upper += obs2 * obs1

            # (sum_{ω∈Ω} sum_{ω_2 ∈Ω'} O(ω, ω_2 | (a_1 , f (θ^t ))))
            for observation in self.observations:
                for observation2 in omega_prime:
                    lower += pseudo_pomdp_model.observation_probability[((observation, observation2), action, theta_t)]
                    """for observations, observation_tuple in pseudo_pomdp_model.observation_probability.items():
                        if (observation, observation2) == observations and action == observation_tuple[0][0] and theta_t == observation_tuple[0][1]:
                            lower += observation_tuple[1]"""


            self.observation_probability[observation1][(action, theta_prime)] = upper/lower


        #Finally, the expected payoff function, R' (a_1 , (θ^t−1 , θ^t )), is
        #given as R(a_1 , θ^t ).
        for action, theta_prime in itertools.product(self.actions, self.states):
            theta_t = theta_prime[1]
            self.payoff[(action, theta_prime)] = pseudo_pomdp_model.payoff[(action, theta_t)]


    def to_Cassandra_format(self):
        """
        Returns a string formatted in Cassandra file format. This string may be directly output to a file (i.e. it contains all whitespace necessary)
        :return string: this POMDP formatted in Cassandra format.
        """
        s = []
        s.append('# TITLE: {}'.format(self.title))
        s.append('# Automatically generated by gt-to-pomdp script.')
        s.append('discount: {}'.format(self.discount))
        s.append('values: reward') #Right now, we'll hard code this, since the input from GT doesn't state whether we minimize or maximize
        state_string = []
        for state in self.states:
            state_string.append(self._statetuple_to_Cassandra(state))

        s.append('states: {}'.format(' '.join(state_string)))
        s.append('actions: {}'.format(' '.join(self.actions)))
        s.append('observations: {}'.format(' '.join(self.observations)))
        s.append('\n')

        #We don't have a start state from GT.
        #s.append('start: ')

        # T: <action> : <start-state> : <end-state> %f

        for theta_t_plusone, (theta_t, action), probability in self.state_transition:
            s.append('T: {} : {} : {} {}'.format(action, self._statetuple_to_Cassandra(theta_t), self._statetuple_to_Cassandra(theta_t_plusone), probability))

        for observation, action, state in itertools.product(self.observations, self.actions, self.states):
            #O: action : end-state : observation %f
            s.append('O: {} : {} : {} {}'.format(action, self._statetuple_to_Cassandra(state), observation, self.observation_probability[observation][(action, state)]))

        for action, state in self.payoff:
            state_string = ''
            substate_string = []
            for substate in state:
                substate_string.append(''.join(substate))
            state_string = (''.join(substate_string))
            s.append('R: {} : {}: * : * {}'.format(action, state_string, self.payoff[(action, state)]))

        return '\n'.join(s)

    def _statetuple_to_Cassandra(self, tuple):
        """
        Formats a POMDP tuple to a combined state that Cassandra format likes.
        :param tuple:
        :return:
        """
        state_string = ''
        substate_string = []
        for substate in tuple:
            substate_string.append(''.join(substate))
        state_string = (''.join(substate_string))
        return state_string

    def __str__(self):
        s = []
        s.append('States: {}'.format(self.states))
        s.append('Actions: {}'.format(self.actions))
        s.append(('Observations: {}'.format(self.observations)))
        s.append('Observation probabilities: {}'.format(self.observation_probability))
        s.append('State Transition probabilities: {}'.format(self.state_transition))
        s.append('Payoffs: {}'.format(self.payoff))
        return '\n'.join(s)

def main(inputfilename, outputfilename=None):
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

    args = parser.parse_args()
    main(args.gtmodel, args.pomdpmodel)