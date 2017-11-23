import numpy as np
import matplotlib.pyplot as plt
from QLearningAgent import QLearningAgent
from ContinousQLearningAgent import QLearningAgentC
from EvolutionaryLearner1 import EvolutionaryLearner1
from EvolutionaryLearner2 import EvolutionaryLearner2
from EvolutionaryLearner3 import EvolutionaryLearner3

import math
import gym

from multiprocessing import Process, Lock , Queue



def getRepresentatiVeEpRewards(task, learningMethod, nrstages, nrRuns, repMethod, nrParellel=1):

    def putRewardProgress(task, nrStages, queue, lock=None):
        epRewards, agent = getRewardProgress(task, learningMethod, nrStages)
        queue.put(epRewards)

    queues = []
    lock = Lock()
    r = 0
    while r<nrRuns:
        processes = []

        while (len(processes)<nrParellel) & (r<nrRuns):
            queues.append(Queue())
            process = Process(target=putRewardProgress, args=(task, nrstages, queues[r], lock))
            processes.append(process)
            r+=1


        for process in processes: process.start()
        for process in processes: process.join()

        print(r,' runs completed')

    rewards = np.zeros((nrRuns, nrstages))

    for run in range(nrRuns):
        epReward = queues[run].get()
        rewards[run,:] = epReward

    if(repMethod == 'median'):
        repReward = np.median(rewards,0)
    else:
        repReward = np.mean(rewards,0)

    return repReward

def getRewardProgress(task, learningMethod, nrStages):

    if learningMethod == 'QL':
        learningAgent = QLearningAgent(task, .9)
    else:
        if learningMethod == 'EL1':
            learningAgent = EvolutionaryLearner1(task)
        else:
            if learningMethod == 'EL2':
                learningAgent = EvolutionaryLearner2(task)
            else:
                if learningMethod == 'EL3':
                    learningAgent = EvolutionaryLearner3(task)

    episodicRewards = learningAgent.learn(nrStages)

    return episodicRewards, learningAgent

if __name__ == "__main__":
    np.random.seed(27)
    task = 'CartPole-v0'
    #task = 'ReversedAddition-v0'
    #task = 'Pendulum-v0'
    #task = 'MountainCarContinuous-v0'
    learningMethod = 'EL3'

    episodicRewards, agent = getRewardProgress(task, learningMethod , nrStages=1000)

    #episodicRewards = getRepresentatiVeEpRewards(task, 'EL1', nrstages=16000, nrRuns=16, repMethod='median',nrParellel=8)
    np.save('episodicRewards', episodicRewards)

    #plt.plot(angles*180/math.pi)


    agent.saveVideo('../Videos/' + task + '-'+ learningMethod + '.avi',3)
    plt.plot(episodicRewards, '.')
    plt.legend()
    plt.xlabel('Number of Episodes')
    plt.ylabel('Approximated value ')
    plt.show()


