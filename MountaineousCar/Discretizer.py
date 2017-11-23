import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from multiprocessing import Process, Lock , Queue


class Discretizer:


    discretization = None
    base = None
    ranges   = None
    validDims = None

    task = None
    env = None


    def __init__(self, task):
        self.env = gym.make(task)
        self.task = task
        self.updateRangesAndSupports()


    def updateRangesAndSupports(self):
        global discretization, ranges, validDims , base
        env = self.env

        upLim = env.observation_space.high
        lowLim = env.observation_space.low

        if(self.task == 'CartPole-v0'):
            ranges = np.zeros((len(upLim), 2))
            ranges[:, 0] = lowLim[:]
            ranges[:, 1] = upLim[:]

            ranges[1, :] = [-1, 1]
            ranges[3, :] = [-4, 4]
            discretization = np.array([10, 10, 10, 10])

        else:
            if(self.task == 'MountainCarContinuous-v0'):
                ranges = np.zeros((len(upLim), 2))
                ranges[:, 0] = lowLim[:]
                ranges[:, 1] = upLim[:]
                discretization = np.array([10, 10])


        base = np.flip(np.cumprod(np.flip(discretization, 0)), 0)
        base[discretization > 1] = base[discretization > 1] / discretization[discretization > 1]
        base[discretization == 1] = 0


    def updateRangesAndSupportsOld(self):

        global discretization, ranges, validDims , base

        env = self.env

        upLim  = env.observation_space.high
        lowLim = env.observation_space.low

        ranges = np.zeros((len(upLim),2))
        ranges[:,0] = lowLim[:]
        ranges[:,1] = upLim[:]

        ranges[1,:] = [-1,1]
        ranges[3,:] = [-4,4]

        discretization = np.array([10, 10, 10, 10])

        base = np.flip(np.cumprod(np.flip(discretization, 0)), 0)
        base[discretization > 1] = base[discretization > 1] / discretization[discretization > 1]
        base[discretization == 1] = 0

    def getStateActionInfo(self):

        global discretization, validDims

        if(self.task=='CartPole-v0'):
            numActions = self.env.action_space.n
        else:
            if(self.task == 'MountainCarContinuous-v0'):
                numActions = 2

        numStates = np.prod(discretization)
        return int(numStates), int(numActions)



    def getStateFromObservation(self,observation):
        global discretization, ranges, validDims , base
        reps = (observation-ranges[:,0])/(ranges[:, 1] - ranges[:, 0])
        reps = np.clip(reps,0,1)
        reps = (reps * (discretization-1)).astype(int)

        state = np.sum(reps*base)

        return state



    def getAcionFromIndex(self,idx):
        if (self.task == 'CartPole-v0'):
            return idx
        else:
            if(self.task == 'MountainCarContinuous-v0'):
                return [2*(idx-.5)]