# solutions.py
# ------------
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

'''Implement the methods from the classes in inference.py here'''

import util
from util import raiseNotDefined
from util import manhattanDistance
import random
import busters

def normalize(self):
    """
    Normalize the distribution such that the total value of all keys sums
    to 1. The ratio of values for all keys will remain the same. In the case
    where the total value of the distribution is 0, do nothing.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> dist.normalize()
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
    >>> dist['e'] = 4
    >>> list(sorted(dist.items()))
    [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
    >>> empty = DiscreteDistribution()
    >>> empty.normalize()
    >>> empty
    {}
    """
    "*** YOUR CODE HERE ***"

    #get sum of all keys
    keySum = self.total()

    #normalize each key's value
    if keySum != 0:
        for item in self.items():
            key = item[0]
            val = self.__getitem__(key) #get the value
            self[key] = val/keySum

def sample(self):
    """
    Draw a random sample from the distribution and return the key, weighted
    by the values associated with each key.

    >>> dist = DiscreteDistribution()
    >>> dist['a'] = 1
    >>> dist['b'] = 2
    >>> dist['c'] = 2
    >>> dist['d'] = 0
    >>> N = 100000.0
    >>> samples = [dist.sample() for _ in range(int(N))]
    >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
    0.2
    >>> round(samples.count('b') * 1.0/N, 1)
    0.4
    >>> round(samples.count('c') * 1.0/N, 1)
    0.4
    >>> round(samples.count('d') * 1.0/N, 1)
    0.0
    """
    "*** YOUR CODE HERE ***"

    x = random.random() #random between 0.0 and 1.0
    self.normalize() #normalize so that the weights are between 0.0 and 1.0
    startVal = 0.0
    # this loop essentially divides up the keys so that each
    # key covers a range of values between 0.0 and 1.0
    # if x was within the range that a certain key covers, return it
    for item in self.items():
        if x <= startVal + item[1]:
            return item[0]
        startVal += item[1]
    return None

def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
    """
    Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
    """
    "*** YOUR CODE HERE ***"
    # find trueDistance
    trueDistance = manhattanDistance(pacmanPosition, ghostPosition)
    
    # check if ghost is observed in jail
    inJail = False
    if noisyDistance == None:
        inJail = True
    
    # cover case when ghost not observed in jail
    if not inJail:
        # consider case where jailPosition and ghostPosition are not equal
        # (ghost is accurately observed to not be in jail)
        if ghostPosition != jailPosition:
            probND = busters.getObservationProbability(noisyDistance, trueDistance)
        # consider case where jailPosition and ghostPosition are equal
        # (ghost is falsely observed to not be in jail, should not happen)
        else:
            probND = 0.0

    # cover case where the ghost is in jail
    if inJail:
        # consider case where jailPosition and ghostPosition are not equal
        # (ghost is falsely observed to be in jail, should not happen)
        if ghostPosition != jailPosition:
            probND = 0.0
        # consider case where jailPosition and ghostPosition are equal
        # (ghost is accurately observed to be in jail)
        else:
            probND = 1.0

    return probND
    
def observeUpdate(self, observation, gameState):
    """
    Update beliefs based on the distance observation and Pacman's
    position.

    The observation is the noisy Manhattan distance to the ghost you are
    tracking.

    self.allPositions is a list of the possible ghost positions, including
    the jail position. You should only consider positions that are in
    self.allPositions.

    The update model is not entirely stationary: it may depend on Pacman's
    current position. However, this is not a problem, as Pacman's current
    position is known.
    """
    "*** YOUR CODE HERE ***"

    # get the known positions of pacman and jail
    pacPos = gameState.getPacmanPosition()
    jailPos = self.getJailPosition()
    
    # iterate through all positions, the last element is jail position
    for pos in self.allPositions:
        
        # probability of the observation being accurate given pacPos, ghostPos = pos,
        # and jailPos --> assuming that ghostPos is certain
        obsProb = self.getObservationProb(observation, pacPos, pos, jailPos)

        # pacPos and jailPos are known, ghostPos is not certain though
        # so must account for this
        new = (self.beliefs[pos])*obsProb
        self.beliefs[pos] = new

    #normalize the new beliefs
    self.beliefs.normalize()
    
def elapseTime(self, gameState):
    """
    Predict beliefs in response to a time step passing from the current
    state.

    The transition model is not entirely stationary: it may depend on
    Pacman's current position. However, this is not a problem, as Pacman's
    current position is known.
    """
    "*** YOUR CODE HERE ***"

    # make an zeroed copy of self.beliefs based on all positions of game
    # this will initialize all probabilites for all positions to 0
    beliefCopy = {}
    for belief in self.allPositions:
        beliefCopy[belief] = 0

    # iterate through all positions, adjusting the beliefs about each position
    for pos in self.allPositions:
    
        # predictions of "how the ghost will/can move in t+1"
        newPosDist = self.getPositionDistribution(gameState, pos)

        # go through all the positions the ghost can go to in t+1
        # and adjust the belief
        # the new beliefs in beliefCopy +
        # (whatever the belief was before x probability ghost goes there (gotten from newPosDist)
        for item in newPosDist.items():
            nextPos = item[0]
            nextProb = item[1]
            beliefCopy[nextPos] = beliefCopy[nextPos] + (self.beliefs[pos]*nextProb)

    # move the new beliefs into the self.beliefs and normalize
    for belief in beliefCopy.items():
        self.beliefs[belief[0]] = belief[1]
        
    self.beliefs.normalize()


        













    
