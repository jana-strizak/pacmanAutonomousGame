import math
import numpy as np
import matplotlib.pyplot as plt 
#from ECE526_A3 import getPacmanLocation
from copy import deepcopy 

def getCoinsDirections(board, coinLoc, yo, xo, direction):
    # RGB value of coin color
    coinColor = [228, 111, 111]

    cone_angle = math.radians(100)  # Cone angle in radians
    max_radius = math.sqrt(board.shape[0]**2 + board.shape[1]**2)  # Maximum radius

    numCoins = 0
    distNearestCoin = 300 # initalize to large number

    # Initialize a list to store points within the cone
    points_within_cone = []
    
    for i in range(len(coinLoc)):
        # coin locations
        y, x = coinLoc[i,0], coinLoc[i,1]

        if (x, y) != (xo, yo):  # Exclude the origin point
            # Calculate the vector from (xo, yo) to (x, y)
            dx = x - xo
            dy = y - yo

            # Calculate the angle between the vector and the reference vector (1, 0)
            # to the right
            #angle = math.atan2(dy, dx)
            # above
            if direction == "above":
                angle = math.atan2(dy, dx) + math.pi/2
            elif direction == "right" or direction == "left":
                angle = math.atan2(dy, dx)
            elif direction == "below":
                angle = math.atan2(dy, dx) - math.pi/2

            # checking if there is a coin there 
            if direction == "above" or direction == "right" or direction == "below":
                if -cone_angle / 2 <= angle <= cone_angle / 2 and math.hypot(dx, dy) <= max_radius:
                    if all(board[y,x] == coinColor):
                        # add to count
                        numCoins += 1

                        # check dist
                        distNearestCoin = min(math.sqrt((x - xo)**2 + (y - yo)**2), distNearestCoin)

                        # just to check 
                        #board[y,x] = 1

            elif direction == "left":
                if (-cone_angle / 2 + math.pi) <= angle or angle <= (cone_angle / 2 - math.pi)  and math.hypot(dx, dy) <= max_radius:
                    if  all(board[y,x] == coinColor):
                        # add to count
                        numCoins += 1

                        # check dist
                        distNearestCoin = min(math.sqrt((x - xo)**2 + (y - yo)**2), distNearestCoin)

                        # just to check 
                        #board[y,x] = 1
    return numCoins, distNearestCoin #, board

def getSuperCoinsDirections(board, coinLoc, yo, xo, direction):

    cone_angle = math.radians(100)  # Cone angle in radians
    max_radius = math.sqrt(board.shape[0]**2 + board.shape[1]**2)  # Maximum radius

    numCoins = 0
    distNearestCoin = 300 # initalize to large number

    # Initialize a list to store points within the cone
    points_within_cone = []
    
    for i in range(len(coinLoc)):
        # coin locations
        y, x = coinLoc[i,0], coinLoc[i,1]

        if (x, y) != (xo, yo):  # Exclude the origin point
            # Calculate the vector from (xo, yo) to (x, y)
            dx = x - xo
            dy = y - yo

            # Calculate the angle between the vector and the reference vector (1, 0)
            # to the right
            #angle = math.atan2(dy, dx)
            # above
            if direction == "above":
                angle = math.atan2(dy, dx) + math.pi/2
            elif direction == "right" or direction == "left":
                angle = math.atan2(dy, dx)
            elif direction == "below":
                angle = math.atan2(dy, dx) - math.pi/2

            # checking if there is a coin there 
            if direction == "above" or direction == "right" or direction == "below":
                if -cone_angle / 2 <= angle <= cone_angle / 2 and math.hypot(dx, dy) <= max_radius:
                    # check dist
                    distNearestCoin = min(math.sqrt((x - xo)**2 + (y - yo)**2), distNearestCoin)

                    # just to check 
                    #board[y,x] = 1

            elif direction == "left":
                if (-cone_angle / 2 + math.pi) <= angle or angle <= (cone_angle / 2 - math.pi)  and math.hypot(dx, dy) <= max_radius:

                    # check dist
                    distNearestCoin = min(math.sqrt((x - xo)**2 + (y - yo)**2), distNearestCoin)

                    # just to check 
                    #board[y,x] = 1
    return numCoins, distNearestCoin #, board

def getGhostDirections(board, coinLoc, yo, xo, direction):
    # RGB value of coin color
    ghostColor = [200,72,72]
    ghostColor2 = [198,89,179]
    ghostColor3 = [84,184,153]
    ghostColor4 = [180,122,48]

    cone_angle = math.radians(100)  # Cone angle in radians
    max_radius = math.sqrt(board.shape[0]**2 + board.shape[1]**2)  # Maximum radius

    numCoins = 0
    distNearestCoin = 300 # initalize to large number

    # Initialize a list to store points within the cone
    points_within_cone = []
    
    for i in range(len(coinLoc)):
        # coin locations
        y, x = coinLoc[i,0], coinLoc[i,1]

        if (x, y) != (xo, yo):  # Exclude the origin point
            # Calculate the vector from (xo, yo) to (x, y)
            dx = x - xo
            dy = y - yo

            # Calculate the angle between the vector and the reference vector (1, 0)
            # to the right
            #angle = math.atan2(dy, dx)
            # above
            if direction == "above":
                angle = math.atan2(dy, dx) + math.pi/2
            elif direction == "right" or direction == "left":
                angle = math.atan2(dy, dx)
            elif direction == "below":
                angle = math.atan2(dy, dx) - math.pi/2

            # checking if there is a coin there 
            if direction == "above" or direction == "right" or direction == "below":
                if -cone_angle / 2 <= angle <= cone_angle / 2 and math.hypot(dx, dy) <= max_radius:
                    if all(board[y,x] == ghostColor) or all(board[y,x] == ghostColor2) or all(board[y,x] == ghostColor3) or all(board[y,x] == ghostColor4):
                        # add to count
                        numCoins += 1

                        # check dist
                        distNearestCoin = min(math.sqrt((x - xo)**2 + (y - yo)**2), distNearestCoin)

                        # just to check 
                        #board[y,x] = 1

            elif direction == "left":
                if (-cone_angle / 2 + math.pi) <= angle or angle <= (cone_angle / 2 - math.pi)  and math.hypot(dx, dy) <= max_radius:
                    if  all(board[y,x] == ghostColor) or all(board[y,x] == ghostColor2) or all(board[y,x] == ghostColor3) or all(board[y,x] == ghostColor4):
                        # add to count
                        numCoins += 1

                        # check dist
                        distNearestCoin = min(math.sqrt((x - xo)**2 + (y - yo)**2), distNearestCoin)

                        # just to check 
                        #board[y,x] = 1
    return numCoins, distNearestCoin #, board

if __name__ == "__main__":
    # Define the cone parameters
    #xo, yo = 20, 20  # Origin point
    cone_angle = math.radians(100)  # Cone angle in radians

    board = np.load("testBoard.npy")
    plt.imshow(board)
    plt.show()

    coinLoc = np.load("coinLocations.npy")
    #yo, xo = getPacmanLocation(board)
    movementMask = np.load("pacmanMovement_Mask.npy")

    numCoins, distNearestCoin = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "above")
    numCoins, distNearestCoin = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "below")
    numCoins, distNearestCoin = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "left")
    numCoins, distNearestCoin = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "right")

    '''
    numCoins, distNearestCoin, boardUp = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "above")
    numCoins, distNearestCoin, boardDown = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "below")
    numCoins, distNearestCoin, boardLeft = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "left")
    numCoins, distNearestCoin, boardRight = getCoinsDirections(deepcopy(board), coinLoc, yo, xo, "right")
    '''
    # Create a grid (2D array)
    #grid = np.zeros([30,30])
    #gridAll = np.repeat(grid[:,:,np.newaxis], 4, axis= 2) 
        
    '''
    # right
    plt.imshow(boardUp)
    plt.show()

    # up
    plt.imshow(boardDown)
    plt.show()

    # down
    plt.imshow(boardLeft)
    plt.show()
    '''
    # left
    plt.imshow(boardRight)
    plt.show()
