import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import numpy as np
from copy import deepcopy
import torch
import torch.optim as optim
import pickle
from stateEvaluationFunctions import stateMaker, dummyGame, getPacmanLocation, getGhostDistances, superCoinDistance

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## USER DEFINED PARAMETERS:
# Name your model:
name = "QNET_2HL_seed100"
# choose a seed for your game:
#seedSet = 42
seedSet = random.randint(0, 100)
# will you be training or just playing?:
trainMode = False
# what type of exploration action selection do you want to use? (will be set to none if not training)
#actionSelected = select_action_Greedy
actionSelected = select_action_UCB
#actionSelected = regular_action_select
# type of neural network for Q evaluation:
DQN = DQN_2HL
#DQN = DQN_3HL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# set up pacman game ( you can set render_mode to human if you want to see the game play, but don't define it if you are just training)
#env = gym.make("MsPacmanNoFrameskip-v4", full_action_space = False, render_mode = "human")
env = gym.make("ALE/MsPacman-v5", full_action_space = False, render_mode = "human") 
#env = gym.make("MsPacmanNoFrameskip-v4", full_action_space = False) 

#movementMask = makeMask(pacmanMovement)
movementMask = np.load("pacmanMovement_Mask.npy")

# load coin locations 
coinLoc = np.load("coinLocations.npy")

# use GPU is avaible 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()

plt.ion()

if trainMode:
        # select which type of action selection you want 
        #actionFunc = select_action_UCB
        actionFunc = actionSelected

        # transition is the data saved for each training step (state, action, nextState, reward)
        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
else:
    #actionFunc = regular_action_select
    actionFunc = actionSelected

BATCH_SIZE = 100 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.05 # EPS_END is the final value of epsilon
EPS_DECAY = 1000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 1e-4 # LR is the learning rate of the ``AdamW`` optimizer

# Get number of actions from gym action space
n_actions = env.action_space.n 

# length of state vector
n_observations = 24

# make neural net for Q determination
policy_net = DQN(n_observations, n_actions).to(device)

# ~~~~~~~Import previously trainned weight values~~~~~~~~~~~
if not(trainMode):
    policy_net.load_state_dict(torch.load('QNET_1HL_15N_policy_net_weights_300epochs.pth')) 
    
# make target net 
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

if trainMode:
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)


steps_done = 0
Na = np.ones(n_actions)

episode_durations = []
errorAll = [] # a list of scores per game, for every game played
scoreAll = [] # one score per game

# actual play and train loop
if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 10

# keeping track of previous boards for ghost search 
boardTime = []
blankBoard = np.zeros([171,160,3]) # initalize to zeros with same board shape
for i in range(0,8):
    boardTime.append(blankBoard)

lifePrev = 0

for i_episode in range(num_episodes):
    env.reset(seed=seedSet)
    print("episode: ", i_episode)

    # game doesn't start for a few moves so wait a little
    observation, reward, terminated, truncated, info = dummyGame(env)

    # region of interest of board 
    board = deepcopy(observation[:171, :, :])

    # keep track of board through time (for ghosts)
    boardTime.pop(0) # get rid of t-2 time point
    boardTime.append(board) # add to t time point
    
    # make state vector from the current board (based on many different factors)
    state = stateMaker(np.stack(boardTime, axis=-1), coinLoc, movementMask)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    errorGame = []
    scoreGame = 0
    for t in count():
        #print("game step: ", t)

        action = actionFunc(state)
        observation, reward, terminated, truncated, info = env.step(action.item())

        scoreGame += reward

        # region of interest of board 
        board = deepcopy(observation[:171, :, :])

        # keep track of board through time (for ghosts)
        boardTime.pop(0)
        boardTime.append(board)

        done = terminated or truncated

        # lost a life (must suffer)
        if info['lives'] < lifePrev:
            reward = reward - 100
        
        # check ghost distances to run away from them 
        y, x = getPacmanLocation(board)
        d_ghost1, d_ghost2, d_ghost3, d_ghost4, d_ghostGood  = getGhostDistances(np.stack(boardTime,axis=-1), y, x, movementMask, ghostDist = True)

        # punish for being too close to ghost
        if 5 < d_ghost1 < 13 or 5 < d_ghost2 < 13 or 5 < d_ghost3 < 13 or 5 < d_ghost4 < 13:
            reward -= 20
        elif d_ghost1 < 5 or d_ghost2 < 5 or d_ghost3 < 5 or d_ghost4 < 5:
            reward -= 50

        # reward for being close to good ghosts
        if 7 < d_ghostGood < 13:
            reward += 30
        elif d_ghostGood < 7:
            reward += 60
        
        # min distance between pacman and supercoin
        dSuperCoinMin = superCoinDistance(np.stack(boardTime,axis=-1), y, x)
        if dSuperCoinMin < 7:
            reward += 60

        # determine reward
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = stateMaker(np.stack(boardTime,axis=-1), coinLoc, movementMask)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        if trainMode:
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if trainMode:
            err = optimize_model()
            if err:
                errorGame.append(err)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            #plot_durations(episode_durations)
            break

        # lost a life (takes some time to start game again)
        if info['lives'] < lifePrev:
            observation, reward, terminated, truncated, info = dummyGame(env)
            state = stateMaker(np.stack(boardTime,axis=-1), coinLoc, movementMask)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # update life 
        lifePrev = info['lives']

    print('Complete with score: ', scoreGame)

    errorAll.append(errorGame)
    scoreAll.append(scoreGame)
    #plot_durations(episode_durations, show_result=True)
    #plt.ioff()
    #plt.show()

if trainMode:
    # save weights
    fileName = name + '_policy_net_weights_50epochs.pth'
    torch.save(policy_net.state_dict(), fileName)
    
    # saving error and scores for this step
    fileName = name + '_error_50epochs.pkl'
    with open(fileName, 'wb') as file:
        pickle.dump(errorAll, file)

fileName = name + '_score_50epochs.pkl'
with open(fileName, 'wb') as file:
    pickle.dump(scoreAll, file)

# plotting
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()

plt.plot(scoreGame)
plt.ylabel("Game's Final Score")
plt.xlabel("Game Epoch")

print("here")