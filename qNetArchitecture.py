#%%
import gymnasium as gym
from IPython import display
import math
import os
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from stateEvaluationFunctions import stateMaker,dummyGame, getPacmanLocation, getGhostDistances, superCoinDistance
#%%
# replay memory stores the tranistion information during the game play 
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize weights
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # initialization
        if m.bias is not None:
            nn.init.zeros_(m.bias)  # Set biases to 0

# define Q-net with 3 layers, which takes in states and returns a Q value for each action
class DQN_2HL(nn.Module):
    def __init__(self, n_observations, n_actions, n_nodes):
        super(DQN_2HL, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_nodes)
        self.layer2 = nn.Linear(n_nodes,n_nodes)
        self.layer3 = nn.Linear(n_nodes, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_3HL(nn.Module):
    def __init__(self, n_observations, n_actions, n_nodes):
        super(DQN_3HL, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_nodes)
        self.layer2 = nn.Linear(n_nodes,n_nodes)
        self.layer3 = nn.Linear(n_nodes,n_nodes)
        self.layer4 = nn.Linear(n_nodes,n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

class DQN_4HL_2(nn.Module):
    def __init__(self, n_observations, n_actions, n_nodes):
        super(DQN_4HL_2, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_nodes)
        self.layer2 = nn.Linear(n_nodes,n_nodes)
        self.layer3 = nn.Linear(n_nodes,20)
        self.layer4 = nn.Linear(20,n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
class DQN_4HL(nn.Module):
    def __init__(self, n_observations, n_actions, n_nodes):
        super(DQN_4HL, self).__init__()
        self.layer1 = nn.Linear(n_observations, n_nodes)
        self.layer2 = nn.Linear(n_nodes,n_nodes)
        self.layer3 = nn.Linear(n_nodes,n_nodes)
        self.layer4 = nn.Linear(n_nodes,n_nodes)
        self.layer5 = nn.Linear(n_nodes, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)

class MsPacmanGame():
    def __init__(self, N_nodes=24, seed = random.randint(0,100), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), DQN=DQN_3HL, weightsFile=''):
        # set up pacman game
        #env = gym.make("MsPacmanNoFrameskip-v4", full_action_space = False, render_mode = "human")
        #self.env = gym.make("ALE/MsPacman-v5", full_action_space = False) 
        #env = gym.make("MsPacmanNoFrameskip-v4", full_action_space = False) 

        self.movementMask = np.load("pacmanMovement_Mask.npy")
        self.coinLoc = np.load("coinLocations.npy")
        self.device = device
        # Get number of actions from gym action space
        self.n_actions = 9 #self.env.action_space.n # 9 possible moves
        # Get the number of state observations

        self.DQN = DQN

        # length of state vector
        self.n_observations = 24
        # make neural net for Q determination
        self.N_nodes = N_nodes
        self.policy_net = self.DQN(self.n_observations, self.n_actions, N_nodes).to(self.device)

        if weightsFile == '':
            # initalize weights for each layer
            self.policy_net.apply(initialize_weights)
        else:
            # Import previously trainned weight values
            self.policy_net.load_state_dict(torch.load(weightsFile)) 

        self.seed = seed

    def trainingParameters(self, batchsize, gamma, eps_start, eps_end, eps_decay, tau, lr, ucb):
        self.BATCH_SIZE = batchsize # BATCH_SIZE is the number of transitions sampled from the replay buffer
        self.GAMMA = gamma # GAMMA is the discount factor as mentioned in the previous section
        self.EPS_START = eps_start # EPS_START is the starting value of epsilon
        self.EPS_END = eps_end # EPS_END is the final value of epsilon
        self.EPS_DECAY = eps_decay # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        self.TAU = tau # TAU is the update rate of the target network
        self.LR = lr # LR is the learning rate of the ``AdamW`` optimizer
        self.steps_done = 0
        self.randomPicked = 0
        self.maxActionPicked = 0
        self.UCB_C = ucb


    # select which action to take by applying the Q-Net and choosing a producing Qmax
    def select_action_Greedy(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        #(f"epsilon threshold = {eps_threshold}")

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.maxActionPicked += 1
                Q = self.policy_net(state)
                return Q.argmax().view(1,1), Q.max()
        else: # randomly sample
            self.randomPicked += 1
            with torch.no_grad():
                return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long), self.policy_net(state).max()

    # select which action to take by applying the Q-Net and choosing a producing Qmax
    def select_action_UCB(self, state):
        self.steps_done += 1

        bonus = self.UCB_C * np.sqrt(np.log(self.steps_done)/(self.Na + 1e-5))
        #print(print(f"bonus = {bonus}"))

        Q = self.policy_net(state)

        ucb_values = Q + torch.tensor(bonus.copy(), device=Q.device)

        # pick action
        with torch.no_grad(): 
            A = ucb_values.argmax().view(1,1)

        # increment count 
        self.Na[A] += 1

        if A == Q.argmax():
            self.maxActionPicked += 1
        else:
            self.randomPicked += 1

        return A, Q.max()

    def regular_action_select(self, state):
        with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                Q = self.policy_net(state)
                return Q.argmax().view(1,1), Q.max()

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE: # only update weights of netword when batch size is reached
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) -> the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # gather collects only the Q values at the actions which were taken 
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values  --performing--> GAMMA*Q(s_{t+1}) + R 
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute L1 (Huber loss):  Q(s_t, a) - [GAMMA*Q(s_{t+1}) + R ]
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad() # set gradients to zero from last step
        loss.backward() # calculate gradients for smoothL1Loss
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step() # perform gradient decent to update Q values using Q - lr*loss
        return loss.item()

    def playGame(self, trainMode, GhostReward=True, plotInline=False):
        # keeping track of previous boards for ghost search 
        boardTime = []
        blankBoard = np.zeros([171,160,3])
        for i in range(0,8):
            boardTime.append(blankBoard)

        # start a new game
        self.env.reset(seed=self.seed)

        # game doesn't start for a few moves so wait a little
        observation, reward, terminated, truncated, info = dummyGame(self.env)

        #  region of interest of board 
        board = deepcopy(observation[:171, :, :])

        # keep track of board through time (for ghosts)
        boardTime.pop(0)
        boardTime.append(board)
            
        # assess state vector (based on many different factors)
        state = stateMaker(np.stack(boardTime,axis=-1), self.coinLoc, self.movementMask, self.device)

        errorGame = []
        scoreGame = 0
        done = False

        lifePrev = 3

        if plotInline:
            img = plt.imshow(self.env.render()) # only call this once
            plt.axis('off')
        # starting game loop

        dur = 0 # keep track of how many rounds were played 
        gameRewards = []
        Qs = []

        while not(done):
            if plotInline: # plot as image
                img.set_data(self.env.render()) # just update the data in the plotted image
                display.display(plt.gcf())
                display.clear_output(wait=True)

            # choose action to take
            action, Q = self.actionFunc(state)
            #print(f"action = {action}, Q = {Q}")

            Qs.append(float(Q))

            dur += 1 # game step was taken

            # make action
            observation, reward, terminated, truncated, info = self.env.step(action.item())

            # record reward of that action state pair
            scoreGame += reward

            # intensify reward
            reward /= 10

            # region of interest of board 
            board = deepcopy(observation[:171, :, :])

            # keep track of board through time (for ghosts)
            boardTime.pop(0)
            boardTime.append(board)

            done = terminated or truncated

            # lost a life (must suffer)
            if info['lives'] < lifePrev:
                reward -= 1
                # Remove the last 3 elements from training data memory (heuristic bc packman can't move for a few turns after being eaten by ghost)
                '''
                if trainMode:
                    for _ in range(3):
                        self.memory.memory.pop()
                '''
            # check ghost distances to run away from them 
            if GhostReward:
                y, x = getPacmanLocation(board)
                d_ghost1, d_ghost2, d_ghost3, d_ghost4, d_ghostGood  = getGhostDistances(np.stack(boardTime,axis=-1), y, x, self.movementMask, ghostDist = True)

                d_ghost1 = 1 if d_ghost1 == 0 else d_ghost1
                d_ghost2 = 1 if d_ghost2 == 0 else d_ghost2
                d_ghost3 = 1 if d_ghost3 == 0 else d_ghost3
                d_ghost4 = 1 if d_ghost4 == 0 else d_ghost4
                d_ghostGood = 1 if d_ghostGood == 0 else d_ghostGood

                # reward is decreased if too close to a ghost, and increased if close to a flashing ghost (normalize between 0 and 1)
                reward -= (1/d_ghost1 + 1/d_ghost2 + 1/d_ghost3 + 1/d_ghost4) /4
                reward += 1/d_ghostGood

                # min distance between pacman and supercoin
                dSuperCoinMin = superCoinDistance(np.stack(boardTime,axis=-1), y, x)

                dSuperCoinMin = 1 if dSuperCoinMin == 0 else dSuperCoinMin
                
                # rewarded for being close to super coin 
                reward += 1/dSuperCoinMin

            gameRewards.append(reward)
            reward = torch.tensor([reward], device=self.device)

            # calculate state after taking that action
            if terminated:
                next_state = None
            else:
                next_state = stateMaker(np.stack(boardTime,axis=-1), self.coinLoc, self.movementMask, self.device)

            if trainMode:
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if trainMode:
                err = self.optimize_model()
                if err:
                    errorGame.append(err)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

            # lost a life (takes some time to start game again)
            if done: 
                self.env.close()
                return scoreGame, errorGame, dur, sum(gameRewards)/len(gameRewards), sum(Qs)/len(Qs)
            elif info['lives'] < lifePrev:
                observation, reward, terminated, truncated, info = dummyGame(self.env)
                state = stateMaker(np.stack(boardTime,axis=-1), self.coinLoc, self.movementMask, self.device)

            # update life 
            lifePrev = info['lives']

    def train(self, actionSelected = "", epochs=10, batchsize=100, gamma=0.6, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=1, lr=0.001, ucb=0.5, saveName="", show=1, GhostReward=True, save=True):
        # initalize game
        if show:
            self.env = gym.make("ALE/MsPacman-v5", full_action_space = False, render_mode = "human")
        else:
            self.env = gym.make("ALE/MsPacman-v5", full_action_space = False)

        # action selection alg
        if actionSelected == "greedy":
            self.actionFunc = self.select_action_Greedy
        elif actionSelected == "ucb":
            self.actionFunc = self.select_action_UCB
            self.Na = np.ones(self.n_actions)
        else:
            self.actionFunc = self.regular_action_select

        # initalize training parameters
        self.trainingParameters(batchsize, gamma, eps_start, eps_end, eps_decay, tau, lr, ucb)

        # transition is the data saved for each training step (state, action, nextState, reward)
        global Transition
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        # make target net (copy of policy net)
        self.target_net = self.DQN(self.n_observations, self.n_actions, self.N_nodes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        print('training...')
        print(f"Using {self.DQN} with {self.N_nodes} nodes, with policy net {id(self.policy_net)} , and {id(self.target_net)}. lr = {self.LR}, gamma = {self.GAMMA}, tau = {self.TAU}. Action = {self.actionFunc}. Memory address for model is {id(self.memory)}")

        errorAll = [] # a list of scores per game, for every game played
        scoreAll = [] # one score per game
        durAll =[]
        avrgRewardAll = []
        avrgQAll = []

        for i_episode in range(epochs):
            
            print("episode: ", i_episode)

            score, error, dur, avrgReward, avrgQ = self.playGame(True, GhostReward=GhostReward)
            
            print('Complete with score: ', score)

            errorAll.append(sum(error)/float(len(error)))
            scoreAll.append(score)
            durAll.append(dur)
            avrgRewardAll.append(avrgReward)
            avrgQAll.append(avrgQ)

        # save nn's weights
        # save weights
        if save:
            fileName = saveName + '_policy_net_weights_'+ str(epochs) + 'epochs.pth'
            torch.save(self.policy_net.state_dict(), fileName)
            
            # saving error and scores for this step
            fileName = saveName + '_error_' + str(epochs) + 'epochs.pkl'
            with open(fileName, 'wb') as file:
                pickle.dump(errorAll, file)

            fileName = saveName + '_score_' + str(epochs) + 'epochs.pkl'
            with open(fileName, 'wb') as file:
                pickle.dump(scoreAll, file)

        print(f"max action picked = {self.maxActionPicked}, random action picked = {self.randomPicked}")

        return scoreAll, errorAll, durAll, avrgRewardAll, avrgQAll
        
        
    def play(self, show = 1, plotInline=False):
        # initalize game
        if show:
            if plotInline:  # will plot it inline for jupyter notebook
                self.env = gym.make("ALE/MsPacman-v5", full_action_space = False, render_mode = "rgb_array")
            else: # will create a window to play the game
                self.env = gym.make("ALE/MsPacman-v5", full_action_space = False, render_mode = "human")
        else:
            self.env = gym.make("ALE/MsPacman-v5", full_action_space = False)

        self.actionFunc = self.regular_action_select

        score, error, dur, avrgReward, avrgQ = self.playGame(False, plotInline=plotInline)
            
        return score, error, dur, avrgReward, avrgQ


#%%
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## USER DEFINED PARAMETERS:
    # Name your model:
    name = "QNET_3HL"
    # choose a seed for your game:
    #seedSet = 42
    seedSet = random.randint(0, 100)
    # will you be training or just playing?:
    trainMode = True
    # what type of exploration action selection do you want to use? (will be set to none if not training)
    actionSelected = 'greedy'
    #actionSelected = 'ucb'
    #actionSelected = 'regular'
    # type of nn for Q evaluation:
    #DQN = DQN_2HL
    DQN = DQN_3HL

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    pacmanGame = MsPacmanGame(DQN = DQN)

    # train Q-net by playing the game and learning from trial and error
    scoreAll, errorGame = pacmanGame.train(saveName=name, actionSelected = actionSelected, epochs=20, gamma=0.9, eps_start=0.9, eps_end=0.2, eps_decay=1000, tau=0.01, lr=0.001, show=0)

    # plotting
    plt.plot(errorGame)
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.title("Q-Net's error")
    plt.show()

    plt.plot(scoreAll)
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.title("Game Score")
    plt.show()
    
    score, error = pacmanGame.play()
    print("Game ended with score: ", score)

    
    weightsFile = 'pacmanAutonomousGame/QNET_3HL_BEST_policy_net_weights_200epochs.pth'
    
    pacmanGame = MsPacmanGame(DQN = DQN, weightsFile = weightsFile)
    
    score, error = pacmanGame.play()

    print("Game ended with score: ", score)
    
    Nepochs = 20
    
    # train Q-net by playing the game and learning from trial and error
    scoreAll, errorGame = pacmanGame.train(saveName=name + '_lr1e-6_onlyCoinReward', GhostReward=False, actionSelected = actionSelected, epochs=Nepochs, gamma=0.7, eps_start=0.9, eps_end=0.3, eps_decay=5000, tau=0.01, lr=1e-6, show=0)

    # plotting
    plt.plot(errorGame)
    plt.ylabel("Error")
    plt.xlabel("Epoch")
    plt.title("Q-Net's error")
    plt.show()

    plt.plot(scoreAll)
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.title("Game Score")
    plt.show()
    
    score, error = pacmanGame.play()
    print("Game ended with score: ", score)
    '''

    pacmanGame = MsPacmanGame(DQN = DQN_2HL)

    #avrgReward, avrgQ 
    scoreAll, errorGame, gameDur, rewardsAll, QsAll = pacmanGame.train(actionSelected='ucb')