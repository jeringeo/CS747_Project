from LearningAgent import LearningAgent
import gym
import numpy as np

class EvolutionaryLearner1(LearningAgent):
    tF = None
    sig = .2
    task = None
    env = None
    nrFeatures = None
    nrActions = None

    alpha = .2

    pop = 25

    def __init__(self, task):

        self.env = gym.make(task)
        LearningAgent.__init__(self, self.env)
        self.nrFeatures = getNrFeaturesForTask(task)
        self.nrActions = self.env.action_space.n
        self.tF = np.zeros((self.nrActions, self.nrFeatures),'float32')

    def runEpisode(self, env, tF):
        epReward = 0
        state = env.reset()
        self.startNewEpisode(state)

        done = False
        nrSteps = 0
        while done is False:

            a = self.getAction(tF, state)
            state, reward, done, info = env.step(a)
            nrSteps += 1
            epReward += 1

        return epReward


    def getAction(self, tF, feature):
        actions = np.dot(tF, feature)
        return np.argmax(actions)

    def getRandomPertubation(self):
        return self.sig * np.random.randn(self.nrActions, self.nrFeatures)


    def getWeightedPertubationForPopulation(self, pop):

        wtPertub = 0
        totalReward = 0

        for entity in range(pop):
            env = self.env
            pertub = self.getRandomPertubation()
            r = self.runEpisode(env, self.tF+pertub)
            wtPertub+= (r*pertub/(entity +1))
            totalReward+=r

        wtPertub = wtPertub/totalReward

        return wtPertub

    def learn(self, nrStages):
        nrGens = nrStages

        rewards = np.zeros(nrGens)

        for gen in range(nrGens):
            pop = self.pop
            wtPertub = self.getWeightedPertubationForPopulation(pop)

            self.tF += ((self.alpha/(pop*self.sig)) * wtPertub)

            rewards[gen] = self.runEpisode(self.env,self.tF)


        return rewards

    def play(self, nrEpisodes):
        env = self.env


        for ep in range(nrEpisodes):
            done = False
            state = env.reset()

            while done is False:
                env.render()
                a = self.getAction(self.tF, state)
                state, _, done, _ = env.step(a)


    def saveVideo(self, fileName, nrEpisodes):
        import cv2 as opencv
        env = self.env

        fourcc = opencv.VideoWriter_fourcc(*'MJPG')
        videoWriter = opencv.VideoWriter(fileName, fourcc, 50, (600, 400))
        for ep in range(nrEpisodes):
            done = False
            state = env.reset()

            while done is False:
                img = env.render('rgb_array')
                img[:, :, :] = img[:, :, [2, 1, 0]]
                videoWriter.write(img.astype('uint8'))
                a = self.getAction(self.tF, state)
                state, _, done, _ = env.step(a)


def getNrFeaturesForTask(task):
    env = gym.make(task)

    if task=='CartPole-v0':
        nrFeatures = env.observation_space.shape[0]
    else:
        if(task=='ReversedAddition-v0'):
            nrFeatures = env.observation_space.n

    return nrFeatures
