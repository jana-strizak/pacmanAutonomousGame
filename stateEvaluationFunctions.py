import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from coinDirectionFinding import getCoinsDirections, getGhostDirections, getSuperCoinsDirections
from copy import deepcopy
import math
import torch
# agent must choose action between 1 and 4 
    # 0 -> stay 
    # 1 -> up
    # 2 -> Right 
    # 3 -> Left 
    # 4 -> Down 
    # 5 -> diagonal right up
    # 6 -> diagonal left up
    # 7 -> diagonal right down
    # 8 -> diagonal left down 

global pacmanTouched 
pacmanTouched = []
'''
tot = np.vstack(pacmanTouched)

originalLocations = tot[0]
originalLocations = np.vstack([originalLocations, tot[1]])

for i in range(0, len(tot)):
    if not(np.any(np.all((originalLocations  == tot[i]),axis=-1))):
        originalLocations = np.vstack([originalLocations, tot[i]])

boardNew = np.zeros([171,160])
for j in range(0,len(originalLocations)):
    boardNew[originalLocations[j][0], originalLocations[j][1]] = 1

plt.imshow(boardNew)

np.save("pacmanMovement", originalLocations)
np.save("pacmanMovement_Mask", boardNew)
'''
def getPacmanLocation(board, save = False):
    # pacman color
    target_color = [210, 164, 74]
    
    # Find the indices where the image matches the target color
    matching_indices = np.argwhere(np.all(board == target_color,axis = -1))
    pacmanTouched.append(matching_indices)

    # Calculate the center point
    center_point = np.round(np.mean(matching_indices, axis=0))
    y, x = int(center_point[0]), int(center_point[1])
    #y, x = matching_indices[0][0], matching_indices[0][1]

    # test to see that the location is right by putting a red dot
    board[y,x,:] = [1,0,0]

    '''
    plt.imshow(board, cmap='gray')  # 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis labels and ticks
    # Show the image
    plt.show()
    '''
    if save:
        return matching_indices
    return y, x

def getWall(board, y, x ,movementMask):
    wallColor = [228, 111, 111]
    movementMask3D = np.repeat((np.logical_not(movementMask))[:,:,np.newaxis], 3, axis=2)
    try:
        rightWall = np.all((movementMask3D*board)[y, x+9, :] == wallColor)
        #rightWall = np.all(board[y+1, x+8, :] == wallColor)
    except:
        rightWall = True
    try:
        leftWall = np.all((movementMask3D*board)[y, x-9, :] == wallColor)
        #leftWall = np.all(board[y+1, x-4, :] == wallColor)
    except:
        leftWall = True
    try:
        downWall = np.all((movementMask3D*board)[y+9, x, :] == wallColor)
        #downWall = np.all(board[y+13, x-4, :] == wallColor)
    except:
        downWall = True
    try:
        upWall = np.all((movementMask3D*board)[y-8, x, :] == wallColor)
        #upWall = np.all(board[y-5, x-4, :] == wallColor)
    except:
        upWall = True
    
    '''
    # check if these pixels are walls
    boardCheck = deepcopy(board)
    boardCheck[y+1, x+8] = [1, 0, 0]
    boardCheck[y+1, x-4] = [1, 0, 0]
    boardCheck[y+13, x-4] = [1, 0, 0]
    boardCheck[y-5, x-4] = [1, 0, 0]
    plt.imshow(boardCheck, cmap='gray')  # 'gray' colormap for grayscale images
    plt.axis('off')  # Turn off axis labels and ticks
    # Show the image
    plt.show()
    '''
    return upWall, downWall, rightWall, leftWall
"""
def getCoinLocations(board, y, x):
    # coin colors
    coinColor = [228, 111, 111]
    movementMask3D = np.repeat(movementMask[:,:,np.newaxis], 3, axis=2)
    coinLoc =  np.argwhere(np.all(board*movementMask3D == coinColor, axis=-1))

    '''
    boardNew = deepcopy(board)
    for i in range(0, len(coinLoc)):
        boardNew[coinLoc[i,0],coinLoc[i,1]] = [1,0,0]
    plt.imshow(boardNew)
    
    '''
    return coinLoc
"""

def getCoins(board, yo, xo, coinLoc):

    numCoins_UP, distNearestCoin_UP = getCoinsDirections(board, coinLoc, yo, xo, "above")
    numCoins_DOWN, distNearestCoin_DOWN = getCoinsDirections(board, coinLoc, yo, xo, "below")
    numCoins_LEFT, distNearestCoin_LEFT = getCoinsDirections(board, coinLoc, yo, xo, "left")
    numCoins_RIGHT, distNearestCoin_RIGHT = getCoinsDirections(board, coinLoc, yo, xo, "right")

    return numCoins_UP, distNearestCoin_UP, numCoins_DOWN, distNearestCoin_DOWN, numCoins_LEFT, distNearestCoin_LEFT, numCoins_RIGHT, distNearestCoin_RIGHT

def getGhosts(board, yo, xo, ghostLoc):

    _, dghost_UP = getGhostDirections(board, ghostLoc, yo, xo, "above")
    _, dghost_DOWN = getGhostDirections(board, ghostLoc, yo, xo, "below")
    _, dghost_LEFT = getGhostDirections(board, ghostLoc, yo, xo, "left")
    _, dghost_RIGHT = getGhostDirections(board, ghostLoc, yo, xo, "right")

    return dghost_UP, dghost_DOWN, dghost_LEFT, dghost_RIGHT

def superCoinDistance(boardTime, y, x):
    # super coin locations
    allCoinLoc = [[149, 10],[17, 9],[17, 149],[148, 149]]

    # coin color
    coinColor = np.array([228, 111, 111])
    coinColor = np.repeat(coinColor[:,np.newaxis], boardTime.shape[3], axis=1)

    # check if locations have coins there (or if it's already eated)
    d_all = [300]
    for c in allCoinLoc:
        # is it present
        if np.any(np.any((boardTime[c[0], c[1], :, :] == coinColor), axis = -1)):
            d = math.sqrt((y - c[0])**2 + (x - c[1])**2)
            d_all.append(d)
    
    return min(d_all)

def findSuperCoins(boardTime, y, x):
    # super coin locations
    allCoinLoc = [[149, 10],[17, 9],[17, 149],[148, 149]]
    presentCoinLoc = [] 

    # coin color
    coinColor = np.array([228, 111, 111])
    coinColor = np.repeat(coinColor[:,np.newaxis], boardTime.shape[3], axis=1)

    # check if locations have coins there (or if it's already eated)
    for c in allCoinLoc:
        if np.any(np.any((boardTime[c[0], c[1], :, :] == coinColor), axis = -1)):
            presentCoinLoc.append(c)
    
    presentCoinLoc = np.array(presentCoinLoc)
    _, dSuperCoin_UP = getSuperCoinsDirections(np.squeeze(boardTime[:,:,:,-1]), presentCoinLoc, y, x, "above")
    _, dSuperCoin_DOWN = getSuperCoinsDirections(np.squeeze(boardTime[:,:,:,-1]), presentCoinLoc, y, x, "below")
    _, dSuperCoin_LEFT = getSuperCoinsDirections(np.squeeze(boardTime[:,:,:,-1]), presentCoinLoc, y, x, "left")
    _, dSuperCoin_RIGHT = getSuperCoinsDirections(np.squeeze(boardTime[:,:,:,-1]), presentCoinLoc, y, x, "right")

    return dSuperCoin_UP, dSuperCoin_DOWN, dSuperCoin_LEFT, dSuperCoin_RIGHT

def getGhostDistances(boardTime, y, x, movementMask, ghostDist = False):
    redGhostColor = np.array([200,72,72])
    redGhostColor = np.repeat(redGhostColor[:,np.newaxis], boardTime.shape[3], axis=1)
    pinkGhostColor = np.array([198,89,179])
    pinkGhostColor = np.repeat(pinkGhostColor[:,np.newaxis],boardTime.shape[3], axis=1)
    blueGhostColor = np.array([84,184,153])
    blueGhostColor = np.repeat(blueGhostColor[:,np.newaxis],boardTime.shape[3], axis=1)
    orangeGhostColor = np.array([180,122,48])
    orangeGhostColor = np.repeat(orangeGhostColor[:,np.newaxis],boardTime.shape[3], axis=1)
    goodGhostColor = np.array([66,114,194]) # good ghost 
    goodGhostColor = np.repeat(goodGhostColor[:,np.newaxis],boardTime.shape[3], axis=1)

    # find locations where board  is equal to ghost colors
    #movementMask3D = np.repeat(movementMask[:,:,np.newaxis], 3, axis=2)
    #movementMask4D = np.repeat(movementMask3D[:,:,:,np.newaxis], 4, axis=3)

    redLoc = np.argwhere(np.any(np.all(boardTime  == redGhostColor, axis= 2), axis=-1))
    redLoc = np.argwhere(np.all(boardTime  == redGhostColor, axis= 2)) # returns (yghost, xghost, frameNum)
    pinkLoc =  np.argwhere(np.any(np.all(boardTime  == pinkGhostColor, axis= 2), axis=-1))
    blueLoc =  np.argwhere(np.any(np.all(boardTime  == blueGhostColor, axis= 2), axis=-1))
    orangeLoc =  np.argwhere(np.any(np.all(boardTime  == orangeGhostColor, axis= 2), axis=-1))
    goodLoc =  np.argwhere(np.any(np.all(boardTime  == goodGhostColor, axis= 2), axis=-1))

    # testing 
    #newBoard = deepcopy(boardTime[:,:,:,-1])
    #newBoard[redLoc_y, redLoc_x,:] = [1,0,0]
    #plt.imshow(newBoard)
    #plt.show()

    # distance from pacman to ghosts
    ghostLoc = []
    try:
        # ghost midpoints 
        center_point = np.round(np.mean(redLoc, axis=0))
        redLoc_y, redLoc_x = int(center_point[0]), int(center_point[1])
        ghostLoc.append([redLoc_y, redLoc_x])

        d_red = math.sqrt((y - redLoc_y)**2 + (x - redLoc_x)**2)
    except: 
        d_red = 300
    try:
        # ghost midpoints 
        center_point = np.round(np.mean(pinkLoc, axis=0))
        pinkLoc_y, pinkLoc_x = int(center_point[0]), int(center_point[1])
        ghostLoc.append([pinkLoc_y, pinkLoc_x])

        d_pink = math.sqrt((y - pinkLoc_y)**2 + (x - pinkLoc_x)**2)
    except: 
        d_pink = 300
    try:
        # ghost midpoints 
        center_point = np.round(np.mean(blueLoc, axis=0))
        blueLoc_y, blueLoc_x = int(center_point[0]), int(center_point[1])
        ghostLoc.append([blueLoc_y, blueLoc_x])

        d_blue = math.sqrt((y - blueLoc_y)**2 + (x - blueLoc_x)**2)
    except: 
        d_blue = 300
    try:
        center_point = np.round(np.mean(orangeLoc, axis=0))
        orangeLoc_y, orangeLoc_x = int(center_point[0]), int(center_point[1])
        ghostLoc.append([orangeLoc_y, orangeLoc_x])

        d_orange = math.sqrt((y - orangeLoc_y)**2 + (x - orangeLoc_x)**2)
    except: 
        d_orange = 300
    try:
        center_point = np.round(np.mean(goodLoc, axis=0))
        goodLoc_y, goodLoc_x = int(center_point[0]), int(center_point[1])
    
        d_good = math.sqrt((y - goodLoc_y)**2 + (x - goodLoc_x)**2)
    except: 
        d_good = 300

    if ghostDist:
        return d_red, d_pink, d_blue, d_orange, d_good
    
    ghostLoc = np.array(ghostLoc)
    # closest distance in all 4 directions of 
    # bad ghosts 
    dghost_UP, dghost_DOWN, dghost_LEFT, dghost_RIGHT = getGhosts(np.squeeze(boardTime[:,:,:,-1]), y, x, ghostLoc)
    # good ghosts
    dgoodghost_UP, dgoodghost_DOWN, dgoodghost_LEFT, dgoodghost_RIGHT = getGhosts(np.squeeze(boardTime[:,:,:,-1]), y, x,goodLoc)

    #return d_red, d_pink, d_blue, d_orange, d_good
    return dghost_UP, dghost_DOWN, dghost_LEFT, dghost_RIGHT, dgoodghost_UP, dgoodghost_DOWN, dgoodghost_LEFT, dgoodghost_RIGHT 

def stateMaker(boardTime, coinLoc, movementMask, device):
    # latest board 
    board = np.squeeze(boardTime[:,:,:,-1])

    # get location of pacman
    cont = True
    count = 2
    while cont:
        try:
            y, x = getPacmanLocation(board)
            cont = False
        except:
            if count < 5:
                board = np.squeeze(boardTime[:,:,:,-count])
                count += 1
            else:
                y, x = 0, 0
                cout = False

    # is there walls on any sides
    up_wall, down_wall, right_wall, left_wall = getWall(board, y, x, movementMask)
    #print("up_wall: ", up_wall)
    #print("down_wall: ", down_wall)
    #print("right_well: ", right_well)
    #print("left_well: ", left_well)

    # coin locations 
    numCoins_UP, distNearestCoin_UP, numCoins_DOWN, distNearestCoin_DOWN, numCoins_LEFT, distNearestCoin_LEFT, numCoins_RIGHT, distNearestCoin_RIGHT = getCoins(board, y, x, coinLoc)

    # Blinking coins 
    dSuperCoin_UP, dSuperCoin_DOWN, dSuperCoin_LEFT, dSuperCoin_RIGHT = findSuperCoins(boardTime, y, x)

    # distance between ghosts 
    dghost_UP, dghost_DOWN, dghost_LEFT, dghost_RIGHT, dgoodghost_UP, dgoodghost_DOWN, dgoodghost_LEFT, dgoodghost_RIGHT  = getGhostDistances(boardTime[:,:,:,-4:], y, x, movementMask)
   
    # seperate variables for normalization 
    state_binary = [up_wall, down_wall, right_wall, left_wall]
    state_num = [numCoins_UP, numCoins_DOWN, numCoins_LEFT, numCoins_RIGHT]
    state_dist = [dSuperCoin_UP, dSuperCoin_DOWN, dSuperCoin_LEFT, dSuperCoin_RIGHT, distNearestCoin_UP, distNearestCoin_DOWN, distNearestCoin_LEFT, distNearestCoin_RIGHT, dghost_UP, dghost_DOWN, dghost_LEFT, dghost_RIGHT, dgoodghost_UP, dgoodghost_DOWN, dgoodghost_LEFT, dgoodghost_RIGHT]
    
    state = [state_binary, state_num, state_dist]

    maxVal = [1, 155, 300]

    # normalize between 0 and 1
    for i in range(len(state)):
        s = state[i]
        v = maxVal[i]
        s = torch.tensor(s, dtype=torch.float32, device=device)
        s /= v
        state[i] = s
    
    state = torch.cat(state)
    state = state.reshape(1,-1)

    if state[0,0].isnan():
        print("here")
    return state

def dummyGame(env):
    
    cont = True
    
    try:
        action = env.action_space.sample()  # insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        # get pacman location
        y_last, x_last = getPacmanLocation(observation)
    except:
        pass
    
    #for _ in range(0, 271):
    while cont:
        action = env.action_space.sample()  # insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        try:
            y, x = getPacmanLocation(observation)
        except:
            continue
    
        if abs(y - y_last) > 2 or abs(x - x_last) > 2:
            cont = False
        
        y_last, x_last = y, x

    return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    env = gym.make("MsPacmanNoFrameskip-v4", full_action_space = False, render_mode = "human") 
    env.reset(seed=42) 

    # load movement spots
    #pacmanMovement = np.load("pacmanMovement.npy")
    #movementMask = makeMask(pacmanMovement)
    movementMask = np.load("pacmanMovement_Mask.npy")

    # load coin locations 
    coinLoc = np.load("coinLocations.npy")

    # game doesn't start for a few moves so wait a little
    observation, reward, terminated, truncated, info = dummyGame(env)

    cont = True
    count = 0

    # keeping track of previous boards for ghost search 
    boardTime = []
    blankBoard = np.zeros([171,160,3])
    for i in range(0,8):
        boardTime.append(blankBoard)
    
    lifePrev = 0
    
    while cont:
        # region of interest of board 
        board = deepcopy(observation[:171, :, :])

        # temporal board (for ghosts)
        boardTime.pop(0)
        boardTime.append(board)

        lives = info['lives']
        # determine agent's move
        action = agentMove(np.stack(boardTime,axis=-1), reward, lives, coinLoc, movementMask)

        # take action
        observation, reward, terminated, truncated, info = env.step(action)

        # display board
        env.render()
        '''
        plt.imshow(board, cmap='gray')  # 'gray' colormap for grayscale images
        plt.axis('off')  # Turn off axis labels and ticks
        # Show the image
        plt.show()
        '''
        count += 1
        #print("action: ", action)
        print("reward: ", reward)
        print("truncated: ", truncated)
        print("info: ", info)

        if terminated or truncated:
            print("here")
            #cont = False
        
        if info['lives'] < lifePrev:
                observation, reward, terminated, truncated, info = dummyGame(env)

    env.close()
