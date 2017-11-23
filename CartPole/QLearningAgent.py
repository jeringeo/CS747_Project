import numpy as np
import random as random
from Discretizer import Discretizer
from LearningAgent import LearningAgent
import gym
import cv2 as opencv

class QLearningAgent(LearningAgent):
    numStates = 0
    numActions = 2
    state = 0
    gamma = 0
    Q = None
    epsilon = .5 #turned out to be best
    attenuation = .999

    discretizer = None

    def __init__(self, task,  gamma):
        env = gym.make(task)
        LearningAgent.__init__(self, env)
        self.discretizer = Discretizer(task)
        self.numStates, self.numActions = self.discretizer.getStateActionInfo()
        self.gamma = gamma
        self.Q = np.random.random((self.numStates, self.numActions))

    def getAction(self, para = None):

        return self.discretizer.getAcionFromIndex(self.action)

    def observe(self, observation, r):
        # return
        sP = self.discretizer.getStateFromObservation(observation)
        aP = self.getEGAction(sP)

        alpha = .25  # update Later
        self.Q[self.state, self.action] +=  (alpha * (r + self.gamma * np.max(self.Q[sP,:]) - self.Q[self.state,self.action]))
        self.state = sP
        self.action = aP



    def getEGAction(self, sP):

        if (random.random() < self.epsilon):
            return random.randint(0, self.numActions - 1)
        else:
            return np.argmax(self.Q[sP, :])


    def startNewEpisode(self, observation):
        self.state = self.discretizer.getStateFromObservation(observation)
        self.action = self.getEGAction(self.state)
        self.epsilon *= self.attenuation

    def runEpisode(self, actionPara=None):
        env = self.env
        epReward = 0
        observation = env.reset()
        self.startNewEpisode(observation)

        done = False
        nrSteps = 0
        while done is False:

            a = self.getAction(actionPara)
            observation, reward, done, info = env.step(a)
            nrSteps += 1
            epReward += 1

            if (done):
                if (nrSteps != self.maxSteps):
                    reward = -self.maxSteps

            self.observe(observation, reward)

        return epReward

    def learn(self, nrStages):
        nrEpisodes = nrStages
        epRewards = np.zeros(nrEpisodes, 'float32')

        for episode in range(nrEpisodes):
            epRewards[episode] = self.runEpisode()

        return epRewards

    def finishLearning(self):
        self.epsilon = 0



    def play(self, nrEpisodes):
        env = self.env

        for ep in range(nrEpisodes):
            done = False
            observation = env.reset()
            self.startNewEpisode(observation)

            while done is False:
                env.render()
                a = self.getAction()
                observation, reward, done, _ = env.step(a)
                self.observe(observation, reward)


    def saveVideo(self, fileName, nrEpisodes):
        env = self.env
        fourcc = opencv.VideoWriter_fourcc(*'MJPG')
        videoWriter = opencv.VideoWriter(fileName, fourcc, 50, (600, 400))

        for ep in range(nrEpisodes):
            done = False
            observation = env.reset()
            self.startNewEpisode(observation)

            while done is False:
                img = env.render('rgb_array')
                img[:, :, :] = img[:, :, [2, 1, 0]]
                videoWriter.write(img.astype('uint8'))
                a = self.getAction()
                observation, reward, done, _ = env.step(a)
                self.observe(observation, reward)





