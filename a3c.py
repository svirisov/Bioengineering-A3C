import torch, os, matplotlib, math
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

os.environ['OMP_NUM_THREADS'] = '1'

GLOBAL_ITER = 32
GAMMA = 0.99
MAX_EPISODES = 1500

env = gym.make('CartPole-v1') #create cartpole environment

NUMSTATES = env.observation_space.shape[0]
NUMACTIONS = env.action_space.n

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weightDecay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weightDecay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # share memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

## Actor-Critic def
class ActorCritic(nn.Module):
    def __init__(self, nStates, nActions):
        super(ActorCritic, self).__init__()
        self.nStates = nStates
        self.nActions = nActions
        # create nn layers
        self.pi1 = nn.Linear(nStates, 256)
        self.pi2 = nn.Linear(256, nActions)
        self.v1 = nn.Linear(nStates, 256)
        self.v2 = nn.Linear(256, 1)
        nnLayers = [self.pi1, self.pi2, self.v1, self.v2]
        for layer in nnLayers: # initialize layers parameters
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)
        self.distribution = torch.distributions.Categorical
    
    def forward(self, state):
        pi1 = torch.tanh(self.pi1(state))
        v1 = torch.tanh(self.v1(state))
        logits = self.pi2(pi1)
        values = self.v2(v1)
        return logits, values
    
    def choose_action(self, obs):
        self.eval()
        #print(f'obs {obs.shape}')
        #epsilon = 2.5e-1
        pi, _ = self.forward(obs)
        #pi += 20
        probs = f.softmax(pi, dim=1).data
        dist = self.distribution(probs)
        #action = dist.sample().numpy()[0]
        return dist.sample().numpy()[0]
    
    def calcLoss(self, state, action, value):
        self.train()
        pi, values = self.forward(state)
        td = value - values
        criticLoss = (td.pow(2) * 1)
        probs = f.softmax(pi, dim=1)
        dist = self.distribution(probs)
        expectedVal = dist.log_prob(action) * td.detach().squeeze()
        actorLoss = torch.mean(((-expectedVal) + (value*0.0001).pow(2)))
        totalLoss = (criticLoss + actorLoss).mean()
        return totalLoss
    
class Agent(mp.Process):
    def __init__(self, globalNet, optimizer, globalEpisodeIdx, globalEpisodeRewards, resultsQ, name):
        super(Agent, self).__init__()
        self.name = 'w%02i' % name
        self.globalEpisodeIdx = globalEpisodeIdx
        self.globalEpisodeRewards = globalEpisodeRewards
        self.resultsQ = resultsQ
        self.globalNet = globalNet
        self.optimizer = optimizer
        self.net = ActorCritic(NUMSTATES, NUMACTIONS)
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')

    def run(self):
        step = 1
        while self.globalEpisodeIdx.value < MAX_EPISODES:
            state, _ = self.env.reset()
            bufferStates, bufferActions, bufferRewards = [], [], []
            episodeReward = 0
            while True:
                if self.name == 'w00':
                    self.env.render()
                    pass
                action = self.net.choose_action(wrapVals(state[None, :]))
                #print(action)
                nextState, reward, done, _, _ = self.env.step(action.item())
                if done:
                    reward = -1
                episodeReward += reward
                bufferActions.append(action)
                bufferStates.append(state)
                bufferRewards.append(reward)
                if step % GLOBAL_ITER == 0 or done:
                    # sync global and local net
                    syncNets(self.optimizer, 
                                  self.net, 
                                  self.globalNet, 
                                  done, 
                                  nextState, 
                                  bufferStates, 
                                  bufferActions, 
                                  bufferRewards, 
                                  GAMMA)
                    bufferStates, bufferActions, bufferRewards = [], [], []
                    if done:
                        getResults(self.globalEpisodeIdx, 
                                        self.globalEpisodeRewards, 
                                        episodeReward, 
                                        self.resultsQ, 
                                        self.name)
                        break
                state = nextState
                step += 1
                #self.env.render()
        self.resultsQ.put(None)

def wrapVals(array, dtype=np.float32):
        if array.dtype != dtype:
            array = array.astype(dtype)
        return torch.from_numpy(array)

def syncNets(optimizer, localNet, globalNet, done, nextState, bufferState, bufferAction, bufferReward, gamma):
        if done:
            valNextState = 0.               # terminal
        else:
            valNextState = localNet.forward(wrapVals(nextState[None, :]))[-1].data.numpy()[0, 0]
        bufferValTarget = []
        for r in bufferReward[::-1]:    # reverse buffer r
            valNextState = r + gamma * valNextState
            bufferValTarget.append(valNextState)
        bufferValTarget.reverse()
        loss = localNet.calcLoss(
            wrapVals(np.vstack(bufferState)),
            wrapVals(np.array(bufferAction), dtype=np.int64) if bufferAction[0].dtype == np.int64 else wrapVals(np.vstack(bufferAction)),
            wrapVals(np.array(bufferValTarget)[:, None]))
        # calculate local gradients and push local parameters to global
        optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(localNet.parameters(), globalNet.parameters()):
            gp._grad = lp.grad
        optimizer.step()
        # get global parameters
        localNet.load_state_dict(globalNet.state_dict())

def getResults(globalEpisodeIdx, globalEpisodeReward, episodeReward, resultsQ, workerName):
        with globalEpisodeIdx.get_lock():
            globalEpisodeIdx.value += 1
        with globalEpisodeReward.get_lock():
            if globalEpisodeReward.value == 0.:
                globalEpisodeReward.value = episodeReward
            else:
                globalEpisodeReward.value = globalEpisodeReward.value * 0.95 + episodeReward * 0.05
        resultsQ.put(globalEpisodeReward.value)
        print(
            workerName,
            "Ep:", globalEpisodeIdx.value,
            "| local reward:  %.0f" % episodeReward,
            "| global reward: %.0f" % globalEpisodeReward.value,
        )

if __name__ == '__main__':
    globalNet = ActorCritic(NUMSTATES, NUMACTIONS)
    globalNet.share_memory()
    optimizer = SharedAdam(globalNet.parameters(), lr=5e-5, betas=(0.6, 0.85))
    globalEpisodeIdx, globalEpisodeRewards, resultsQ = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    nonAve = mp.Queue()
    # parallel operation
    workers = [Agent(globalNet, optimizer, globalEpisodeIdx, globalEpisodeRewards, resultsQ, i) for i in range(4)]
    [w.start() for w in workers]
    results = []
    nonAverage = []
    while True:
        res = resultsQ.get()
        #nonAv = 
        if res is not None:
            results.append(res)
        else: 
            break
    [w.join() for w in workers]
    plt.plot(results)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

    import csv
    with open('./data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(results)
