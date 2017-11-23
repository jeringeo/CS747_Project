from LearningAgent import LearningAgent
import gym
import numpy as np
import cv2 as opencv

class EvolutionaryLearner2(LearningAgent):
    tF = None
    sig = .1
    task = None
    env = None
    nrFeatures = None
    nrActions = None

    population = None

    alpha = .2

    pop = 5
    nrSurvivors = int(pop*.4)
    selectionTable = None

    def __init__(self, task):

        self.env = gym.make(task)
        LearningAgent.__init__(self, self.env)
        self.nrFeatures = getNrFeaturesForTask(task)
        self.nrActions = self.env.action_space.n
        self.tF = np.zeros((self.nrActions, self.nrFeatures),'float32')
        self.population = np.random.rand(self.pop,self.nrActions, self.nrFeatures)


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

    def mutateAndUpdatePopulation(self):
        rewards = np.zeros(self.pop)
        for i in range(self.pop):
            rewards[i] = self.runEpisode(self.env,self.population[i,:,:])

        selec = np.argsort(-rewards)[:self.nrSurvivors]
        survivors = self.population[selec,:,:]
        self.tF = survivors[0,:,:]

        self.population = self.getMutatedPopulation(survivors, self.pop)

    def getMutatedPopulation(self,base,popSize):
        indices = np.random.randint(self.nrSurvivors,size=(self.pop-self.nrSurvivors,self.nrActions,self.nrFeatures))
        newGen = np.zeros_like(self.population)
        newGen[:self.nrSurvivors,:,:] = base
        newGen[self.nrSurvivors:, 0, 0] = base[indices[:, 0, 0], 0, 0]
        newGen[self.nrSurvivors:, 0, 1] = base[indices[:, 0, 1], 0, 1]
        newGen[self.nrSurvivors:, 1, 0] = base[indices[:, 1, 0], 1, 0]
        newGen[self.nrSurvivors:, 1, 1] = base[indices[:, 1, 1], 1, 1]

        newGen[self.nrSurvivors:,:,:] += self.sig * np.random.randn(self.pop-self.nrSurvivors,self.nrActions,self.nrFeatures)
        return newGen

    def learn(self, nrStages):
        nrGens = nrStages

        rewards = np.zeros(nrGens)

        for gen in range(nrGens):
            self.mutateAndUpdatePopulation()
            rewards[gen] = self.runEpisode(self.env,self.tF)
            if((gen%1000) == 0):
                print(gen)


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

    def saveVideo(self, fileName,nrEpisodes):
        env = self.env
        fourcc = opencv.VideoWriter_fourcc(*'MJPG')
        videoWriter = opencv.VideoWriter(fileName, fourcc, 50, (600, 400))


        for ep in range(nrEpisodes):
            done = False
            state = env.reset()

            while done is False:
                img = env.render('rgb_array')
                img[:,:,:] = img[:,:,[2,1,0]]
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
