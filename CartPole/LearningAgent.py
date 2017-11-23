import numpy as np
import matplotlib.pyplot as plt
import math
import gym
from multiprocessing import Process, Lock , Queue
import cv2 as opencv


class LearningAgent:
    maxSteps = 200
    env = None
    action = None

    def __init__(self, env):
        self.env = env


    def observe(self, observation, r):
        pass

    def startNewEpisode(self, observation):
        pass


    def learn(self, nrStages):
       pass

    def play(self, nrEpisodes):
        pass