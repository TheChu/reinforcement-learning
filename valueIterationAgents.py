# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for i in range(self.iterations):
            newValues = self.values.copy()
            for state in states:
                possibleActions = self.mdp.getPossibleActions(state)
                if len(possibleActions) == 0:
                    newValues[state] = 0
                else:
                    newValues[state] = max(
                        [self.computeQValueFromValues(state, action) \
                         for action in possibleActions])
            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        return sum([t * (self.mdp.getReward(state, action, sPrime) \
                    + self.discount * self.values[sPrime]) for (sPrime, t) \
                    in self.mdp.getTransitionStatesAndProbs(state, action)])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        if not possibleActions:
            return None

        # Returns the action with the maximum sum of the products of next
        # possible states and corresponding probabilities
        return max(possibleActions, key = lambda action:
                    sum([self.values[sPrime] * prob for (sPrime, prob) in \
                         self.mdp.getTransitionStatesAndProbs(state, action)]))

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        statesLength = len(states)

        for i in range(self.iterations):
            state =  states[i % statesLength]
            possibleActions = self.mdp.getPossibleActions(state)
            if self.mdp.isTerminal(state):
                pass
            elif len(possibleActions) == 0:
                self.values[state] = 0
            else:
                self.values[state] = max(
                    [self.computeQValueFromValues(state, action) \
                     for action in possibleActions])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value
        iteration for a given number of iterations using the supplied
        parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = self.getPredecessors()

        # Initialize an empty priority queue
        queue = util.PriorityQueue()

        # Push each state in the priority queue
        for state in [s for s in states if not self.mdp.isTerminal(s)]:
            queue.push(state, -self.getDiff(state))

        for i in range(self.iterations):
            if queue.isEmpty():
                break

            # Pop a state off the pririty queue and update its value in
            # self.values
            state = queue.pop()
            self.values[state] = self.getHighestQValue(state)

            # Push predecessors in the priority queue
            for predecessor in predecessors[state]:
                diff = self.getDiff(predecessor)

                if diff > self.theta:
                    queue.update(predecessor, -diff)

    def getPredecessors(self):
        states = self.mdp.getStates()

        # Initialize predecessors as a set of sets for all states
        predecessors = {}
        for state in states:
            predecessors[state] = {}

        # Compute predecessors of all states
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for (nextState, _) in \
                        self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState][state] = True

        return predecessors

    def getDiff(self, state):
        return abs(self.values[state] - self.getHighestQValue(state))

    def getHighestQValue(self, state):
        return max([self.computeQValueFromValues(state, action) \
                    for action in self.mdp.getPossibleActions(state)])
