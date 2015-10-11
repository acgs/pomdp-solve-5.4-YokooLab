# encoding: utf-8
"""Define the various models used by gt_to_pomdp.

This module defines the objects that represent the various models needed to convert a Game Theory model to a POMDP.

The objects are intended to have as weak coupling as possible, but each model exposes a function to build itself from a 'previous' model.

The general order of conversion is GTModel -> PseudoPOMDPModel -> POMDPModel
"""

__author__ = 'Victor Szczepanski'
import operator  # used for reshaping a matrix
import functools
import itertools
from decimal import *
getcontext().prec = 10
import numpy as np
from scipy.linalg import block_diag

from utils import flatten_tuple, to_tuple



# Just convenient names to make it more clear, mainly in documentation
STATE = str
SIGNAL = str
ACTION = str


class Player(object):
    """
    Represent a pre-FSA - finite state automaton without initial starting state.

    Note:
        `observation_marginal_distribution` is only initialized as a flat, empty Dict by constructor.
        It should be filled in by something else that knows about the marginal distribution, like a `GTModel`.
        `payoff` is only initialized as a flat, empty Dict by constructor.
        It should be filled in by something else that knows about its (marginal) payoff, like a `GTModel`.

    Args:
        lines (Optional[List[str]]): ordered sequence of strings to parse and build this Player from.

    Attributes:
        name (str): name of the Player

        states (List[`STATE`] | List[tuple]): ordered sequence of states (or set of states, in a joint-player)

        actions (List[`ACTION`]): ordered sequence of actions

        signals (List[`SIGNAL`]): ordered sequence of signals

        state_machine (Dict[`STATE`, `ACTION`]): mapping from states to actions - this specifies the action to take in a state

        state_transitions (Dict[`STATE`, Dict[`SIGNAL`, `STATE`]]): mapping from states to signals to states - this specifies the transitions (edges) of the pre-FSA.

        observation_marginal_distribtuion (Dict[Tuple[`ACTION`], Dict[`SIGNAL`, Decimal]]): mapping from sets of actions to observations to probabilities - this specifies the marginal distribution for this player's observation probability, given an action profile.

        payoff (Dict[Tuple[`ACTION`], Decimal]): mapping from sets of actions to a payoff - this is the immediate reward for this player given an action profile.
    """

    def __init__(self, lines=None):
        self.name = ""
        self.states = []
        self.actions = []
        self.signals = []
        self.state_machine = {}  # maps a state to an action
        self.state_transitions = {}  # 2 level dictionary that maps a state to a signal to a state.
        self.observation_marginal_distribution = {}  # 2 level dictionary that maps an action profile (set of actions) to observations (single element, in the case of 2 players) to probabilities.
        self.payoff = {}  # maps a pair of actions to a payoff (real value). Not initialized by calling from_lines.

        if lines is not None:
            p = Player.from_lines(lines)
            self.name = p.name
            self.states = p.states
            self.actions = p.actions
            self.signals = p.signals
            self.state_machine = p.state_machine
            self.state_transitions = p.state_transitions

            # something else needs to fill in our observation_marginal_distribution, since we need the joint distribution.

    def build_marginal_distribution(self, joint_distribution, my_dimension: int):
        """Using a observation joint distribution table of probabilities, construct the marginal distribution for this player.
        Set this player's `observation_marginal_distribution`

        Args:
            joint_distribution (Dict[tuple[`ACTION`], list[list[float]]]):
                The joint distribution (n-dimensional matrix, where the number of players is n)
                 of observation probabilities for all players.
                 Maps an action tuple to a matrix where the length of each row is len(`self.signals`)
                 and there are len(`action tuple`) == len(`players`) dimensions.

            action_profiles (tuple[`ACTION`]): The set of action profiles to consider.
            my_dimension (int): The dimension to consider the marginal distribution for - for player 1, this is 0 (the first dimension). For player n, this is n-1 (the last dimension)
        """
        self.observation_marginal_distribution = {}
        for action_profile in joint_distribution:
            self.observation_marginal_distribution[action_profile] = {}
            for observation in range(len(self.signals)):  # joint_distribution is indexed by signal index.
                sum_observation_probabilities = Decimal(0.0)
                # now iterate over joint_distribution.
                # we know how many dimensions there are from action_profile
                # we sum over all others except our dimension, which we fix.
                dims = []
                for dim in range(len(action_profile)):
                    if dim == my_dimension:
                        dims.append([observation])
                    else:
                        dims.append([i for i in range(len(self.signals))])

                dims = [d for d in itertools.product(*dims)]

                for dim in dims:
                    distribution = joint_distribution[action_profile]
                    for d in dim:
                        distribution = distribution[d]

                    sum_observation_probabilities += distribution

                self.observation_marginal_distribution[action_profile][self.signals[observation]] = sum_observation_probabilities

    def join(self, other_player=None):
        """Join this player's FSA with another player's FSA.

        Create a joint pre-FSA from this player and `other_player`.
        If `other_player` is None, returns this `Player`.

        Note:
            This function is not idempotent - calling it with the same player as `other_player` will create a new joint player.
        Args:
            other_player (Player): The Player to join with this Player.
        Returns:
            Player: The `Player` representing the joint pre-FSA of this `Player` and `other_player`.
        """
        if other_player is None:
            return self

        joint_player = Player()

        joint_player.name = self.name + other_player.name
        # We create joint states by doing the cartesian product of this player's states and the other player's states.

        for my_state in self.states:
            m = to_tuple(my_state)

            for other_state in other_player.states:
                o = to_tuple(other_state)

                joint_player.states.append(m + o)  # will just append other_state to it.

        for my_action in self.actions:
            m = to_tuple(my_action)

            for other_action in other_player.actions:
                o = to_tuple(other_action)

                joint_player.actions.append(m + o)  # will just append other_state to it.

        for my_signal in self.signals:
            m = to_tuple(my_signal)

            for other_signal in other_player.signals:
                o = to_tuple(other_signal)

                joint_player.signals.append(m + o)  # will just append other_state to it.

        # The new state machine maps a set of states to a set of actions

        for state in joint_player.states:
            # state could be split into many 'substates' - but we know the last one is from the other player.
            my_state = state[:-1]
            if len(my_state) == 1:
                my_state = my_state[0]
            their_state = state[-1]

            m = to_tuple(my_state)
            t = to_tuple(their_state)

            joint_player.state_machine[m + t] = to_tuple(self.state_machine[my_state]) + to_tuple(other_player.state_machine[their_state])

        # The new state transitions maps a set of 2 level state/signals to a set of states.
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
        """Parse a `Player` configuration from a set of strings and return a `Player`.

        .. seealso::
            Class :gt_to_pomdp:models:GTModel
                Describes the full Shun format for an input Game Theory model.
                The description of an Automaton is contained in this format.

        Parse an automaton description in Shun format and create a new `Player` from it.

        Some expressions may be given multiple times -
        e.g. the same STATE ACTION pair may be given multiple times.
        The parser will only use the last one, if an overwriting would occur.
        So, if R, P are states, g is an observation, and there are two lines::

            R g R
            R g P

        The parser will only use R g P.

        Note:
            The automaton description itself does not provide the `payoff` information - this should be filled in by
            an external program.
        Args:
            lines (List[str]): ordered sequence of strings that describe the automaton in Shun format to parse.
        Returns:
            Player: a new Player that represents the automaton parsed from `lines`

        Raises:
            SyntaxError: If some line in `lines` is not formatted correctly.
        """

        # parsing state machine
        # states: name, states, actions, signals, state_machine, state_transitions, end
        # we use integers for faster comparison: 1, 2, 3, 4, 5, 6, 7

        player = Player()

        state = 1

        for line in lines:
            # ignore all blank lines
            if len(line.strip()) is 0:
                continue
            if state is 1:  # name
                player.name = line.split()[1]
                state = 2
            elif state is 2:  # states
                player.states = line[line.index(':')+1:].lstrip().rstrip().split()
                state = 3
            elif state is 3:  # actions
                player.actions = line[line.index(':')+1:].lstrip().rstrip().split()
                state = 4
            elif state is 4:  # signals
                player.signals = line[line.index(':')+1:].lstrip().rstrip().split()
                state = 5
            elif state is 5:  # state_machine
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
        s = list()
        s.append('Automaton {}'.format(str(self.name)))
        s.append('States: {}'.format(' '.join(flatten_tuple(self.states))))
        s.append('Actions: {}'.format(' '.join(flatten_tuple(self.actions))))
        s.append('Signals: {}'.format(' '.join(flatten_tuple(self.signals))))
        s.append('\n'.join(['{} {}'.format(key, value) for key, value in self.state_machine.items()]))
        for state_transition in self.state_transitions:
            s.append(str(state_transition) + ' '.join(str(self.state_transitions[state_transition])))
        return '\n'.join(s)


class GTModel(object):
    """Represent a Game Theory Model.

    A Game Theory Model consists of a discount rate, a set of variables for substitution, a set of automaton (players),
    a set of signal distributions for every action profile, and a payoff function for every action profile.

    A GTModel can be parsed from a ``Shun Game Theory Model``. The format of the input file can be given as a regular expression as follows::

        Title[ ]*:[ ]*[a-z|A-Z|0-9|' '|\\t]*[\\n]
        Discount Rate[ ]*:[ ]*FLOAT[\\n]
        Variables[ ]*:[ ]*([a-z|A-Z|0-9]+'='FLOAT)*[\\n]
        Players[ ]*:[ ]*([a-z|A-Z|0-9]+' ')+[\\n]
        [\\n]
        (
        Automaton[ ]+[a-z|A-Z|0-9]+[\\n]
        States[ ]*:[ ]*([a-z|A-z]+' ')+[\\n]
        Actions[ ]*:[ ]*([a-z|A-z]+' ')+[\\n]
        Signals[ ]*:[ ]*([a-z|A-z]+' ')+[\\n]
        (STATE ACTION[\\n])+
        (STATE SIGNAL STATE[\\n])+
        ){Players} # number of players
        [\\n]
        Signal Distribution[\\n]
        (
        ACTION{players}[ ]*:[ ]*(FLOAT[ ]){Product_{p in players} (p.signals)}[\\n] # Product of number of signals of each player.
        ){Product_{p in players} (p.actions)} #Product of number of actions for each player.
        [\\n]
        Payoff Matrix[\\n]
        (
        ACTION{players} : (FLOAT[ ]){Players}[\\n] # number of players
        ){Product_{p in players} (p.actions)} #Product of number of actions for each player.

    Note that newlines in the regex are explicitly listed above. Some expressions may be given multiple times -
    e.g. in the Automaton description, the same STATE ACTION pair may be given multiple times.
    The parser will only use the last one, if an overwriting would occur. So, if R, P are states, g is an observation,
    and there are two lines::

        R g R
        R g P

    The parser will only use R g P.

    Args:
        filename (str): The path to a Shun Game Theory Model file to parse.

    Attributes:
        name (str): The title of this model.

        discount (float): The discount factor

        variables (Dict[str, float]): mapping of variable name to its value

        player_names (List[str]): sequence of player names.

        players (List[Player]): sequence of Players.

        signal_distribution (Dict[tuple(Action), List[List[float]]): Dictionary that maps a tuple of actions (action profile) to an N dimensional matrix (list of lists) where N is the number of players, and each dimension is length s, where s is the number of signals (observations) that player can have.

        payoff (Dict[tuple(Action), Dict[str, float]]): mapping of a tuple of actions (action profile) to a dictionary that maps player names to values (floats)
    """

    def __init__(self, filename=None):
        self.title = ""
        self.discount = 0.0

        self.variables = {}

        self.player_names = []

        self.players = []

        self.signal_distribution = {}

        self.payoff = {}

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
        """Parse a `Shun Game Theory Model`` and build a new `GTModel` from it.

        Returns:
            GTModel : The result of parsing the Game Theory file.
        Raises:
            SyntaxError : If the input file at `filename` has a syntax error.
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

        # player_lines collects lines to pass to the player constructor
        player_lines = []
        signal_lines = []
        payoff_lines = []

        with open(filename, 'r') as f:
            for line in f.readlines():
                if state is 1: #title
                    model.title = line[line.index(':')+1:].rstrip()
                    state = 2
                elif state is 2: #discount
                    model.discount = Decimal(line[line.index(':')+1:].rstrip().lstrip())
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

        #build each player's marginal distribution

        for dim, player in enumerate(model.players):
            player.build_marginal_distribution(model.signal_distribution, dim)

        return model

    @staticmethod
    def _parse_variables(line):
        """
        Take a variables line and parse it into a dictionary as described by GTModel.variables

        Just a convenience function. Parsed numbers are converted to Decimal.
        :param line: string formatted as q=0.001 s=0.001 g=0.3 ...
        :return variables: a dictionary that maps string variable names to float values
        """
        variables = {}
        vs = line.split()
        for assignment in vs:
            if '=' in assignment:
                split_assignment = assignment.split('=')
                variables[split_assignment[0]] = Decimal(split_assignment[1])

        return variables


    @staticmethod
    def _parse_matrix(lines, variables):
        """
        Parse an iterable of strings that represent an action pair -> matrix mapping.

        Take a collection of lines of the form:
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

        and build a dictionary that maps the action tuples to flat matricies of values, where the values are converted based on
        the input variables.

        any variable encountered in `lines` is assumed to be found in the keyset of `variables`.
        :param lines: iterable of strings representing the matricies
        :param variables: dict that maps a variable name to a floating point value.
        :return mapping: dict that maps an action tuple (action profile) to a flat matrix of floating point values.
        """
        mapping = {}
        for line in lines:
            actions = tuple(line[:line.index(':')].rstrip().lstrip().split(',')) #make tuple of actions
            values = line[line.index(':')+1:].rstrip().lstrip().split() #make list of value expressions

            #replace all variables in each expression with values
            for i in range(len(values)):
                values[i] = ''.join([str(variables[c]) if c in variables else c for c in values[i]])

                values[i] = Decimal(eval(values[i])) #actually evaluate any expressions

            mapping[actions] = values

        return mapping



    @staticmethod
    def _shape(flat, dims):
        """
        Shape a flat matrix into a len(`dims`)-D matrix along dimensions `dims`.

        Note - this function is recursive, so it can be slow for large dimensions.
        Just used as a helper for parsing payoff or signal distribution matricies.

        :param flat: the flat matrix to shape
        :param dims: sequence of dimensions along which to shape the matrix - specifies the size of each dimension of resulting matrix
        :return: a len(`dims`) dimensional matrix (nested lists) where the ith dimension has size (length) `dims`[i]
        """
        subdims = dims[1:]
        subsize = functools.reduce(operator.mul, subdims, 1)
        if dims[0]*subsize != len(flat):
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
    """Describe a PseudoPOMDP model.

    A PseudoPOMDPModel is the intermediate POMDP described in "Automated Equilibrium Analysis of Repeated Games with Private Monitoring: A POMDP Approach" by YongJoon Joe.
    Provides a method to convert from a GTModel.

    Args:
        gt_model (Optional[GTModel]): A Game Theory model to convert from.
    Attributes:
        title (str): the title of this PseudoPOMDPModel.
        discount (float): the discount factor
        gt (GTModel): the GTModel this PseudoPOMDPModel was converted from.
        states (List[STATE]): the states of other players
        actions (List[ACTION]): the actions of player 1
        observations (List[str]): the observations of player 1
        observation_probability (Dict[tuple[ACTION,STATE], Dict[str, float]): mapping from an action/state tuple to an observation to a probability. Represents the probability of an observation given an action/state.
        state_transition (Dict[STATE], Dict[ACTION, Dict[STATE, float]]]): mapping from a state θ^t to an action to a state θ^t+1 to a probability. Represents the conditional probability of transitioning to state θ^t+1 given θ^t and an action.
        payoff (Dict[tuple[ACTION,STATE], float): mapping from an action/state tuple to a payoff (real number). Represents the immediate payoff of taking ACTION in STATE.
        players (List[Player]): list of Players in the PseudoPOMDPModel - this structure is just used to help translate from GTModel.
        signal_distribution (Dict[tuple(Action), List[List[float]]): Dictionary that maps a tuple of actions (action profile) to an N dimensional matrix (list of lists) where N is the number of players, and each dimension is length s, where s is the number of signals (observations) that player can have.
        player1 (Player): the Player to consider as the agent in a POMDP. Other players get joined into a joint Player.
    """

    def __init__(self, gt_model=None):
        self.title = ''
        self.discount = 0.0
        self.gt = None
        self.states = []  # A set of states of other players (player 2 in a 2 player game)
        self.actions = []  # A set of actions for player 1
        self.observations = []  # A set of observations/signals of player 1.
        # Note that this is an observation of the entire world - that is, the joint observations of all other players.

        self.observation_probability = {}  # A function that maps an observation given an action/state tuple to a probability
        self.state_transition = {}  # A function that represents the conditional probability that the next state
        # is θ^t+1 when the current state is θ^t and the action of player 1 is a_1
        self.payoff = {}  # A function that maps an action/state tuple to a real value.

        # Additional data to help translation
        self.players = []
        self.signal_distribution = {}
        self.player1 = None

        self._V = {}  # The expected discounted payoff for player 1. Maps a state pair (θ_1, θ_2) to a real value.
        # Calculated as:
        # V_{θ_1,θ_2} = g_1 ((f(θ_1 ), f(θ_2))) +
        # δ * Sum_{ω_1, ω_2 in Ω} (o((ω_1, ω_2 ) | (f(θ_1 ), f(θ_2 ))) · V_{T(θ_1, ω_1 ),T(θ_2, ω_2 )} .

        if gt_model is not None:
            self.from_gt(gt_model)

    @property
    def V(self):
        """dict[tuple[STATE,STATE], float]: The expected payoff of player 1 in this pseudoPOMDPModel.

        Note:
            Access to V assumes this PseudoPOMDPModel has been properly built
            - it makes use of most of this PseudoPOMDPModel's attributes.
            Accessing V prior to building this PseudoPOMDPModel may
            result in undefined behaviour (including Exceptions).

        This is only calculated once, on the first access to V. Repeated accesses will just return the 'cached' calculation.
        V is calculated as::

            V_{θ_1,θ_2} = g_1 ((f(θ_1), f(θ_2))) +
            δ * Sum_{ω_1, ω_2 in Ω} (o((ω_1, ω_2) | (f(θ_1), f(θ_2))) ·
            V_{T(θ_1, ω_1),T(θ_2, ω_2)}

        Where f(STATE) is the action to take in STATE,
        o((ω_1, ω_2 ) | (f(θ_1), f(θ_2)) is the joint probability distribution for observation pair (ω_1, ω_2) given an action tuple,
        and T(θ_1, ω_1) is the state transition function given observation ω_1 and current state θ_1.

        """
        if len(self._V) == 0:
            self._V = self._calculate_expected_payoff()
        return self._V

    def from_gt(self, gt_model):
        """Translates a GTModel into this PseudoPOMDPModel.

        Sets all of this model's attributes.
        Follows the procedure described in "Automated Equilibrium Analysis of Repeated Games with Private Monitoring: A POMDP Approach" by YongJoon Joe.

        Args:
            gt_model (GTModel): the GTModel to translate from.
        """
        self.title = gt_model.title
        """:type : str"""
        self.discount = gt_model.discount
        self.gt = gt_model
        # states are the cartesian product of all states in gt_model of players except player 1

        self.players = gt_model.players
        self.signal_distribution = gt_model.signal_distribution

        player1 = gt_model.players[0]
        """:type : Player"""
        self.player1 = player1
        """:type : Player"""

        #We need to build a joint-FSA from all other players
        opponent = gt_model.players[1]
        """:type : Player"""
        for player in gt_model.players:
            if player is not player1 and player is not gt_model.players[1]:
                opponent = opponent.join(player)

        #Θ is a set of states of player 2
        self.states = opponent.states
        """:type : list[tuple]"""

        # self.states is a list of tuples (maybe of length 1)
        # actions are the set of actions of player 1. Again, if all players use FSA M, we can pick any player.
        self.actions = player1.actions
        """:type : list[str]"""

        # observations are the set of observations of player 1.
        self.observations = player1.signals
        """:type : list[str]"""

        # Observation probability maps an observation given an action/state tuple to a probability
        # Then there are |observations| x |actions| x |states| many entries

        self.observation_probability = {}
        """:type : dict[str, dict[str, dict[str, float]]]"""
        action_state_tuples = [(action_state[0], action_state[1]) for action_state in itertools.product(self.actions, self.states)]
        """:type : list[tuple[str, str]]"""

        for action in self.actions:
            self.observation_probability[action] = {}
            for state in self.states:
                self.observation_probability[action][state] = {}

        for (action, state) in action_state_tuples:

            # we'll loop over each state to find the action profile (a_1, a_2, ...)
            action_profile = self._to_action_profile(state, action)
            """:type : tuple[str]"""

            for observation in self.observations: #gets the combinations of observations possible.
                # O(ω_1 | a_1 , θ^t ) = o_1 (ω_1 | (a_1 , f (θ^t ))).
                probability = player1.observation_marginal_distribution[action_profile][observation]
                self.observation_probability[action][state][observation] = probability

        # state_transition function P (θ^t+1 | θ^t , a_1 ) represents the conditional probability that
        # the next state is θ^t+1 when the current state is θ^t and the
        # action of player 1 is a_1
        # P (θ^t+1 | θ^t , a_1 ) = sum_{ω_2 in Omega | T(θ t, ω_2) = θ t+1}  o_2 (ω_2 | (a_1 , f (θ^t ))).
        # So, we make a tuple (state2, (state1, action), value), where state2 =  θ^t+1 and state1 = θ^t
        # Note that ω_2 is the observation of player 2


        #initialize self.state_transition:
        for theta_t in self.states:
            self.state_transition[theta_t] = {}
            for action in self.actions:
                self.state_transition[theta_t][action] = {}

        for (action, theta_t) in action_state_tuples:
            action_profile = self._to_action_profile(theta_t, action)
            for theta_t_plusone in self.states:
                probability = 0
                for omega2 in self.observations:
                    if player1.state_transitions[theta_t][omega2] != theta_t_plusone:
                        continue
                    probability += opponent.observation_marginal_distribution[action_profile][omega2]
                self.state_transition[theta_t][action][theta_t_plusone] = probability

        # payoff R : A × S → R is given as:
        # R(a_1 , θ^t ) = g_1 ((a_1 , f (θ^t ))).
        # g_i (a) = sum_{ω∈Ω^2} π_i (a_i , ω_i )o(ω | a)
        # So, we make a tuple (action, state, payoff) where action = a_1, state = θ^t, and payoff is a real value (floating point)
        #Player i’s realized payoff is determined by her own action and signal and denoted π_i (a_i , ω_i )
        for (action, state) in action_state_tuples:
            action_profile = self._to_action_profile(state, action)
            payoff_matrix = gt_model.payoff[action_profile]

            payoff = payoff_matrix[0] # g_1 ((a_1 , f (θ^t ))).
            #We'll pull out the payoff function for player1 - this will be used to generate the expected payoff later.
            self.player1.payoff[action_profile] = payoff

            self.payoff[(action, state)] = payoff

    def _calculate_expected_payoff(self) -> dict:
        """
        Computes the expected payoff function V_{θ_1,θ_2} for player 1

        In particular, computes the following function::
            V_{θ_1,θ_2} = g_1 ((f(θ_1 ), f(θ_2))) +
            δ * Sum_{ω_1, ω_2 in } (o((ω_1, ω_2 ) | (f(θ_1 ), f(θ_2 ))) · V_{T(θ_1, ω_1 ),T(θ_2, ω_2 )} .

        While this function can be computed by the GTModel, we'll just do it here,
        since it is more relevant to the PseudoPOMDP and POMDP models.
        Note:
            This function uses the numpy linalg package to solve the system of linear equations.
        Returns:
            Dict[tuple(STATE,STATE), float]: a dictionary with keys (θ_1,θ_2) (i.e. state pair) that maps to a real value.
        """
        #equations will become our matrix of coefficients
        equations = {}
        answer_vector = {}
        state_pairs = [sp for sp in itertools.product(self.states, repeat=2)]

        #Now, we iterate over all state pairs
        for theta1, theta2 in state_pairs:
            coefficients_dict = {}
            for state_pair in state_pairs:
                coefficients_dict[state_pair] = 0

            f_theta1 = self.player1.state_machine[theta1]
            f_theta2 = self.player1.state_machine[theta2]
            # each coefficient is given by δ * (o((ω_1, ω_2 ) | (f(θ_1 ), f(θ_2 ))) * V_{T(θ_1, ω_1 ),T(θ_2, ω_2 )}.
            # It is possible to get multiple values for a single coefficient,
            # so we'll sum them (since the coefficents are generated in a sum)
            for observation1, observation2 in itertools.product(self.observations, repeat=2):
                coefficients_dict[(self.player1.state_transitions[theta1][observation1],
                                  self.player1.state_transitions[theta2][observation2])] +=\
                    float(self.discount *
                          self.signal_distribution[(f_theta1, f_theta2)][self.observations.index(observation1)]
                          [self.observations.index(observation2)]
                          )

            #We need to subtract 1 from the coefficient corresponding to V_{θ_1,θ_2}
            coefficients_dict[(theta1, theta2)] -= 1

            #add coefficients_dict as a row to equations
            equations[(theta1, theta2)] = coefficients_dict
            #Now, we make the answer vector from -g_1(f(θ_1), f(θ_2))
            answer_vector[(theta1, theta2)] = float(-1 * self.player1.payoff[(f_theta1, f_theta2)])
            if answer_vector[(theta1, theta2)] == -0.0 or answer_vector[(theta1, theta2)] == 0.0:
                answer_vector[(theta1, theta2)] = 0.0

        #Now, we can solve the system of linear equations. First, lets make lists out of the dictionaries so we can be sure they iterate in the same order.
        equation_matrix = []
        answer_matrix = []
        for state_pair in state_pairs:
            coefficient_row = []
            for state_pair2 in state_pairs:
                coefficient_row.append(equations[state_pair][state_pair2])
            equation_matrix.append(coefficient_row)
            answer_matrix.append(answer_vector[state_pair])

        coefficients = np.array(equation_matrix)
        answers = np.array(answer_matrix)

        solution = np.linalg.solve(coefficients, answers)

        V = {}
        for i, state_pair in enumerate(state_pairs):
            V[state_pair] = solution[i]

        return V

    def _to_action_profile(self, state: str, action: str) -> tuple:
        """Returns the action profile for taking action `action` in state `state`.

        In particular, this function finds f(`state`) for all other players (i.e. the action to take in state `state`) and returns a tuple of those actions appended to `action`.
        This is used to look up things like the payoff or observation probability.

        Args:
            state (STATE): a state to look up the best action for
            action (ACTION): the action of player 1 - just used as a convenience when building the full action profile. Not actually needed to compute f(`state`) for other players.

        Returns:
            tuple[ACTION, ACTION, ...]: A tuple of actions, where the first action is `action` and the remaining are f(`state`) for each other player in this PseudoPOMDPModel.
        """
        action_profile = [action]
        for i in range(len(state)):
            other_action = self.players[i+1].state_machine[state[i]]
            action_profile.append(other_action)
        action_profile = tuple([action for action in action_profile])

        return action_profile

    def __str__(self):
        s = list()
        s.append('States: {}'.format(self.states))
        s.append('Actions: {}'.format(self.actions))
        s.append(('Observations: {}'.format(self.observations)))
        s.append('Observation probabilities: {}'.format(self.observation_probability))
        s.append('State Transition probabilities: {}'.format(self.state_transition))
        s.append('Payoffs: {}'.format(self.payoff))
        return '\n'.join(s)


class POMDPModel(object):
    """
    Describe a Partially Observable Markov Decision Process Model.

    This object encapsulates a POMDP Model and provides some functions to convert from a PseudoPOMDP and output itself in Cassandra format.

    Args:
        pseudo_pomdp_model (Optional[PseudoPOMDPModel]): if not None, constructs this POMDPModel by converting from `pseudo_pomdp_model`.

    Attributes:
        title (str): The title of this POMDPModel
        discount (float): The discount factor
        gt (GTModel): A reference to a GTModel - used to get at things like the joint distribution table.
        states (List[tuple[STATE]]): A list of sets of states of the other player.
        actions (List[ACTION]): List of actions for player 1
        observations (List[str]): List of observations of player 1. Note that this is an observation of the entire world - that is, the joint observations of all other players.
        observation_probability (Dict[tuple[ACTION,STATE], Dict[str,float]): Mapping from action/state tuple to mapping from observation to float.
        state_transition (List[tuple[STATE,STATE,ACTION,float]]): List of tuples (θ^t, θ^t+1, a_1, probability) that represents the probability that the next state is θ^t+1 when the current state is θ^t and the action is a_1.
        payoff (Dict[tuple[ACTION,STATE], float]): Mapping from action/state tuple to a payoff real value. Represents the immediate payoff of taking an action in a state.
        players (List[Player]): The players in this POMDP - usually this is just two players: player 1 and a joint player of all other players.
        V : The Expected Payoff for player 1, assuming other players use the same FSA as player 1.
    """

    def __init__(self, pseudo_pomdp_model: PseudoPOMDPModel=None):
        self.title = ''
        self.discount = 0.0
        self.gt = None
        self.states = []  # A set of states of other players (player 2 in a 2 player game)
        self.actions = []  # A set of actions for player 1
        self.observations = []  # A set of observations/signals of player 1.
        # Note that this is an observation of the entire world - that is, the joint observations of all other players.


        self.observation_probability = {}  # A function that maps an observation given an action/state tuple to a probability
        self.state_transition = []  # A function that represents the conditional probability that the next state
        # is θ^t+1 when the current state is θ^t and the action of player 1 is a_1
        self.payoff = {}  # A function that maps an action/state tuple to a real value.
        self.players = []
        self.V = {} #  The Expected Payoff function for player1 in the POMDP - assuming other players use the same FSA.

        if pseudo_pomdp_model is not None:
            self.from_pseudo_pomdp(pseudo_pomdp_model)

    def from_pseudo_pomdp(self, pseudo_pomdp_model: PseudoPOMDPModel):
        """Converts `pseudo_pomdp_model` into this `POMDPModel`.

        Follows the procedure in "Automated Equilibrium Analysis of Repeated Games with Private Monitoring: A POMDP Approach" by YongJoon Joe.

        Args:
            pseudo_pomdp_model (PseudoPOMDPModel): The PseudoPOMDPModel to convert from.

        """
        self.title = pseudo_pomdp_model.title
        self.discount = pseudo_pomdp_model.discount
        self.gt = pseudo_pomdp_model.gt

        self.players = pseudo_pomdp_model.players

        # actions and observations are identical
        self.actions = pseudo_pomdp_model.actions
        self.observations = pseudo_pomdp_model.observations

        #We also just take V from pseudo_pomdp.
        self.V = pseudo_pomdp_model.V

        # The key idea of this translation is to introduce a set of new
        # combined states Θ' , where Θ' = Θ^2 . Namely, we assume
        # that a state θ'^t in the standard POMDP model represents
        # the combination of the previous and current states (θ^t−1 , θ^t )
        # in our model present in the previous subsection.
        self.states = [s for s in itertools.product(pseudo_pomdp_model.states, repeat=2)]
        #remove any unreachable states (i.e. pairs that do not exist in the state_transition function of pseudo_pomdp_model.

        removed_states = []
        for state in self.states:
            found = False
            for action in self.actions:
                try:
                    a = pseudo_pomdp_model.state_transition[state[0]][action][state[1]]
                    if a == 0:
                        raise KeyError
                except KeyError:
                    continue

                found = True
                break
            if not found:
                removed_states.append(state)

        for state in removed_states:
            self.states.remove(state)

        # A new state transition function P' (θ'^t+1 | θ'^t , a_1 ) is equal
        # to P (θ^t+1 | θ^t , a_1 ) in the original model if θ'^t+1 = (θ^t , θ^t+1 )
        # and θ'^t = (θ^t−1 , θ^t ), i.e., the previous state in θ'^t+1 and the
        # current state in θ'^t are identical. Otherwise, it is 0.
        for action, theta_t_plus_one, theta_t in itertools.product(self.actions, self.states, self.states):
            probability = 0
            if theta_t_plus_one[0] is theta_t[1]:
                probability = pseudo_pomdp_model.state_transition[theta_t_plus_one[0]][action][theta_t_plus_one[1]]

            self.state_transition.append((theta_t_plus_one, (theta_t, action), probability))

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
        for observation in self.observations:
            self.observation_probability[observation] = {}

        for observation1, action, theta_prime in itertools.product(self.observations, self.actions, self.states):
            theta_t = theta_prime[0]
            theta_t_plus_one = theta_prime[1]
            upper = 0
            lower = 0

            # (sum_{ω_2 ∈Ω} O(ω_1 , ω_2 | (a_1 , f (θ^t ))) )

            # Since it is O(ω_1 , ω_2 | (a_1 , f (θ^t ))) ), we will split it to o_1(ω_1 | a_1, f(θ^t )) x o_2(ω_2 | a_1, f(θ^t ))
            for observation2 in self.observations:
                if self.players[0].state_transitions[theta_t][observation2] != theta_t_plus_one:
                        continue
                obs2 = self.gt.players[1].observation_marginal_distribution[pseudo_pomdp_model._to_action_profile(theta_t, action)][observation2]
                obs1 = self.gt.players[0].observation_marginal_distribution[pseudo_pomdp_model._to_action_profile(theta_t, action)][observation1]

                upper += obs2 * obs1

            # (sum_{ω∈Ω} sum_{ω_2 ∈Ω'} O(ω, ω_2 | (a_1 , f (θ^t ))))
            for observation in self.observations:
                for observation2 in self.observations:
                    if self.players[0].state_transitions[theta_t][observation2] != theta_t_plus_one:
                        continue
                    obs2 = self.gt.players[1].observation_marginal_distribution[pseudo_pomdp_model._to_action_profile(theta_t, action)][observation2]
                    obs1 = self.gt.players[0].observation_marginal_distribution[pseudo_pomdp_model._to_action_profile(theta_t, action)][observation]
                    lower += obs2 * obs1

            self.observation_probability[observation1][(action, theta_prime)] = upper/lower


        #Finally, the expected payoff function, R' (a_1 , (θ^t−1 , θ^t )), is
        #given as R(a_1 , θ^t ).
        for action, theta_prime in itertools.product(self.actions, self.states):
            theta_t = theta_prime[1]
            self.payoff[(action, theta_prime)] = pseudo_pomdp_model.payoff[(action, theta_t)]

    def to_value_function(self, player1: Player):
        """Return the value function for player1's pre-FSA.

        See "A Variance Analysis for POMDP Policy Evaluation", Fard, Pineau, and Sun, AAAI-2008 for a description of the translation procedure.

        pomdp-solve expects each an action paired with each alpha vector, so we return a list of actions
        corresponding to the respective alpha vector in `V`.

        Args:
            player1 (Player): The player representing the policy graph / pre-FSA to generate the value function for.

        Returns:
            V, A:
                V - the len(`S`) * len(`K`) dimensional vector that is the value function, where each row is an alpha vector corresponding to its respective state in the pre-FSA of `player1`.

                A - the len(`S`) dimensional vector that lists the actions of each alpha vector in `V`.
        """
        K = np.array([k for k in player1.state_machine.keys()], ndmin=1)
        S = self.states
        """:type : list[str]"""
        A = np.array(self.actions, ndmin=1)

        # set up matricies to use: V, R, T, O, and Π, and vectors a(k) and r^k.

        a = {}
        """:type : dict[str, str]
        maps a state k to an action"""
        for k in K:
            a[k] = player1.state_machine[k]

        r = {}
        """:type : dict[str, list[float]]
        maps a state k to a list of payoffs, indexed by state index."""
        for k in K:
            r[k] = {}

        for k in K:
            payoff = []
            for s in S:
                payoff.append(float(self.payoff[(a[k], s)]))
            r[k] = payoff

        R = np.hstack((r[k] for k in K))

        T = self._make_T(S, K, a)

        O = self._make_O(A, S, K, a)

        Pi = self._make_Pi(K, S, player1)

        # V = (I−γTOΠ)^{−1}R
        temp = np.multiply(float(self.discount), T)  # multiply does scalar multiplication
        temp = np.dot(temp, O)  # dot does matrix multiplication
        temp = np.dot(temp, Pi)
        temp = np.subtract(np.identity(temp.shape[0]), temp)  # we can use temp.shape[0], because it is square.
        temp = np.linalg.inv(temp)
        V = np.dot(temp, R)

        # we'll split V up into sublists based on |S|: every |S| elements in V belong to one list.
        temp = []

        i = 0
        while i < V.shape[1]:
            t = []
            z = 0
            while z < (len(S)):
                t.append(V[0, i])
                i += 1
                z += 1
            temp.append(t)

        V = temp
        ak = [a[k] for k in K]
        return V, ak

    def _make_T(self, S: list, K: list, a: dict) -> np.array:
        """Compute the conditional probability of state transition in POMDP.

        T is an len(`S`)*len(`K`) x len(`S`)*len(`K`) dimensional block diagonal matrix of len(`K`) x len(`K`) blocks,
        with T_a(k) as the kth diagonal sub-matrix. T_a(k) is the transition probability matrix when selecting action a(k).

        Args:
            S (list[str]): a list of states
            K (list[str]): a list of player 1 state machine states.
            a (dict[str,str]): a mapping from state to action.
        Returns:
            np.array: mapping from action to 2-D array indexed
            first by theta_t index and then by theta_t_plus_one index.
        """
        T_a = {}  # maps action to state to state to real
        """:type : dict[str, np.array]
        conditional probability of moving from one state to another given an action.
        accessed as T_a[action][theta_t_index][theta_t_plus_one_index]"""

        # self.state_transition.append((theta_t_plus_one, (theta_t, action), probability))
        # Rearrange self.state_transition so we can index by action.
        for (theta_t_plus_one, (theta_t, action), probability) in self.state_transition:
            theta_t_index = S.index(theta_t)
            theta_t_plus_one_index = S.index(theta_t_plus_one)
            if action not in T_a:
                T_a[action] = np.zeros((len(S), len(S)))

            T_a[action][theta_t_index][theta_t_plus_one_index] = probability

        #We'll make the sub-matricies ot give to T. They are |K| x |K| blocks, with T_a(k) as the kth diagonal submatrix
        t_submatricies = []
        """:type : list[np.array]"""
        for k in K:
            t_submatricies.append(T_a[a[k]])

        T = block_diag(*t_submatricies)  # * black magic unpacks the t_submatricies list.
        """:type : np.array"""
        return T

    def _make_O(self, A: list, S: list, K: list, a: dict) -> np.array:
        """Compute the observation probability matrix.

        Computes a 2-d matrix where the kth block, sth sub-block contains the sth row of O_a(k).

        O_a(k) is a list of probabilities, indexed by `Z` (the observations), given action a(k).

        Args:
            A (list[str]): A list of actions
            K (list[str]): A list of state machine states.
            S (list[str]): A list of states
            a (dict[str, str]): A mapping from machine state k to an action.
        Returns:
            np.array : a len(`S`)*len(`K`) x len(`Z`)*len(`S`)*len(`K`) dimensional block diagonal matrix.
             The kth block, sth sub-block contains the sth row of O_a(k)
        """
        # Make O_a(k) - the observation probability indexed by actions
        O_a = {}
        for action in A:
            if action not in O_a:
                O_a[action] = []
                for i in range(len(S)):
                    O_a[action].append([])
            for i, state in enumerate(S):
                for j, observation in enumerate(self.observations):
                    O_a[action][i].append(self.observation_probability[observation][(action, state)])

        #First, we build the sub-blocks of size |S| x |S| from the |Z| vectors from O_a(k)
        observation_subblocks = []
        for k in K:
            action = a[k]
            for i, state in enumerate(self.states):
                observation_subblocks.append(block_diag([float(o) for o in O_a[action][i]]))

        # now, we have |K| * |K| diagonal block matricies that are diagonal block matricies of the observation_subblocks
        O = block_diag(*observation_subblocks)  # * black magic unpacks the observation_subblocks list.

        return O

    def _make_Pi(self, K: list, S: list, player1: Player) -> np.array:
        """Compute the transition function of `player1`'s pre-FSA in matrix form.

        Compute an len(`Z`)*len(`S`)*len(`K`) x len(`S`)*len(`K`) dimensional block matrix of len(`K`) x len(`K`) blocks.
        Each block is a len(`Z`)*len(`S`) x len(`S`) block diagonal sub-matrix of len(`S`) x len(`S`) sub-blocks.
        Each sub-block is a len(`Z`) vector.
        Then, for all s, the zth component of the sth diagonal block of the (k1,k2) submatrix
        is 1 if k2 is a successor of k1 given observation z in `player1`'s pre-FSA, else 0.

        Args:
            K (list[str]): a list of state machine states.
            S (list[str]): a list of states
            player1 (Player): the player whose pre-FSA should be used.

        Returns:
            np.array: an len(`Z`)*len(`S`)*len(`K`) x len(`S`)*len(`K`) dimensional block matrix
            representing the transition function of `player1`'s pre-FSA.
        """
        # Pi is made of sub-matricies Pi_{k1,k2}.
        # Each sub-matrix is made of |S| diagonal blocks (i.e. Pi_{k1,k2} is diagonal block matrix).
        # Each diagonal block is made of a vector |Z|.
        # The components of the vectors of size |Z| are 1 if k_2 is the next state of the FSA when the FSA is in k_1 and observes z. Otherwise 0.

        # We build the submatricies for all s in S from the sub-matrix [(Pi_{k1,k2})_s]_z
        pi_submatricies = {}


        for k1, state2 in enumerate(K):
            pi_submatricies[k1] = {}
            for k2, state3 in enumerate(K):
                #pi_submatricies[k1][k2] selects a Pi_{k1,k2}
                subsubmatrix = []
                for state in S:  # we don't use state here, because we're just collecting diagonal blocks.
                    #pi_submatricies[k1][k2][s] selects a diagonal block
                    diagonal_block = np.zeros((len(self.observations), 1))
                    for z, observation in enumerate(self.observations):
                        # z selects an element of pi_submatrices[k1][k2][s];pi_submatrices[k1][k2][s][z] is a real number.
                        diagonal_block[z][0] = ((1 if player1.state_transitions[state2][observation] == state3 else 0))
                    subsubmatrix.append(diagonal_block)
                pi_submatricies[k1][k2] = block_diag(*subsubmatrix)

        # we need to build a list of the submatricies to pass to np.bmat, and need to do it in column major order
        pi = []
        for k1 in range(len(K)):
            sub_pi = []
            for k2 in range(len(K)):
              sub_pi.append(pi_submatricies[k2][k1])
            pi.append(sub_pi)

        Pi = np.bmat(pi)
        return Pi

    def value_function_to_Cassandra_format(self, V: list, ak: list) -> str:
        """Takes a Value Function V and associated actions ak and outputs them in the Cassandra alpha file format.

        pomdp-solve expects an alpha file format as::

            action_number
            alpha-vector_coefficients

            action_number
            alpha-vector_coefficients

            ...


        So, we look up the index of each action in ak and prepend that to each vector.
        The return string may be directly output to a file.

        Args:
            V (list[list[float]]): the value function - list of vectors ordered by ak
            ak (list[str]): the actions associated with each vector in V
        Returns:
            str: The action/vector pairs formatted in Cassandra alpha file format.
        """
        if len(V) != len(ak):
            raise IndexError("Lengths of V and ak must be equal.")

        s = []
        for action, vector in zip(ak, V):
            s.append(str(self.actions.index(action)))
            s.append(' '.join([str(v) for v in vector]))
            s.append('')
        return '\n'.join(s)

    def to_Cassandra_format(self)-> str:
        """Returns a string formatted in Cassandra file format.

        This string may be directly output to a file (i.e. it contains all whitespace necessary)

        Returns:
            str: this POMDP formatted in Cassandra format.
        """
        s = list()
        s.append('# TITLE: {}'.format(self.title))
        s.append('# Automatically generated by gt_to_pomdp script.')
        s.append('discount: {}'.format(self.discount))
        s.append('values: reward') #Right now, we'll hard code this, since the input from GT doesn't state whether we minimize or maximize
        state_string = []
        for state in self.states:
            state_string.append(self._statetuple_to_Cassandra(state))

        s.append('states: {}'.format(' '.join(state_string)))
        s.append('actions: {}'.format(' '.join(self.actions)))
        s.append('observations: {}'.format(' '.join(self.observations)))
        s.append('')

        #We don't have a start state from GT.
        #s.append('start: ')

        # T: <action> : <start-state> : <end-state> %f

        for theta_t_plusone, (theta_t, action), probability in self.state_transition:
            s.append('T: {} : {} : {} {}'.format(action, self._statetuple_to_Cassandra(theta_t),
                                                 POMDPModel._statetuple_to_Cassandra(theta_t_plusone), probability))

        for observation, action, state in itertools.product(self.observations, self.actions, self.states):
            #O: action : end-state : observation %f
            s.append('O: {} : {} : {} {}'.format(action, POMDPModel._statetuple_to_Cassandra(state), observation,
                                                 self.observation_probability[observation][(action, state)]))

        for action, state in self.payoff:
            state_string = ''
            substate_string = []
            for substate in state:
                substate_string.append(''.join(substate))
            state_string = (''.join(substate_string))
            s.append('R: {} : {}: * : * {}'.format(action, state_string, self.payoff[(action, state)]))

        return '\n'.join(s)

    @staticmethod
    def _statetuple_to_Cassandra(t: tuple)-> str:
        """Formats a POMDP tuple to a combined state that Cassandra format likes.
        Args:
            t (tuple[STATE]): a tuple of STATEs that should be combined (concatenated)
        Returns:
            str: The concatenation of STATEs in `t`
        """
        substate_string = []
        for substate in t:
            substate_string.append(''.join(substate))
        state_string = (''.join(substate_string))
        return state_string

    def __str__(self):
        s = list()
        s.append('States: {}'.format(self.states))
        s.append('Actions: {}'.format(self.actions))
        s.append(('Observations: {}'.format(self.observations)))
        s.append('Observation probabilities: {}'.format(self.observation_probability))
        s.append('State Transition probabilities: {}'.format(self.state_transition))
        s.append('Payoffs: {}'.format(self.payoff))
        s.append('Expected Payoff: {}'.format(self.V))
        return '\n'.join(s)

