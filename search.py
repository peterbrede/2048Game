# Peter Brede MP2

import numpy as np 
import random
from typing import List
import copy

def printGrid(Grid: List[List[int]]):
    print("----------")
    for row in range(4):
        print(str(Grid[row][0])+ "|" + str(Grid[row][1]) + "|" + str(Grid[row][2]) + "|" + str(Grid[row][3]))
    print("----------")


def swipeLeft(Grid: List[List[int]]):
    # initialize
    newGrid = copy.deepcopy(Grid)
    # Condense first
    for row in range(4):
        for col in range(3):
            colno = col + 1
            while (colno > 0 and newGrid[row][colno-1] == 0):
                newGrid[row][colno-1] = newGrid[row][colno]
                newGrid[row][colno] = 0
                colno = colno - 1
    # Now combine
    for row in range(4):
        for sq in range(3):
            if (newGrid[row][sq] == newGrid[row][sq+1]):
                newGrid[row][sq] = 2*newGrid[row][sq]
                if (sq+1 == 3):
                    newGrid[row][sq+1] = 0
                else:
                    newGrid[row][sq+1] = newGrid[row][sq+2]
                    if (sq+2 == 3):
                        newGrid[row][sq+2] = 0
                    else:
                        newGrid[row][sq+2] = newGrid[row][sq+3]
                        newGrid[row][sq+3] = 0
    return newGrid

def swipeUp(Grid: List[List[int]]):
    # initialize
    newGrid = copy.deepcopy(Grid)
    # Condense first
    for col in range(4):
        for row in range(3):
            rowno = row + 1
            while (rowno > 0 and newGrid[rowno-1][col] == 0):
                newGrid[rowno-1][col] = newGrid[rowno][col]
                newGrid[rowno][col] = 0
                rowno = rowno - 1
    # Now combine
    for col in range(4):
        for sq in range(3):
            if (newGrid[sq][col] == newGrid[sq+1][col]):
                newGrid[sq][col] = 2*newGrid[sq][col]
                if (sq+1 == 3):
                    newGrid[sq+1][col] = 0
                else:
                    newGrid[sq+1][col] = newGrid[sq+2][col]
                    if (sq+2 == 3):
                        newGrid[sq+2][col] = 0
                    else:
                        newGrid[sq+2][col] = newGrid[sq+3][col]
                        newGrid[sq+3][col] = 0
    return newGrid
def swipeRight(Grid: List[List[int]]):
    # initialize
    newGrid = copy.deepcopy(Grid)
    # Condense first
    for row in range(4):
        for col in reversed(range(1,4)):
            colno = col - 1
            while (colno < 3 and newGrid[row][colno+1] == 0):
                newGrid[row][colno+1] = newGrid[row][colno]
                newGrid[row][colno] = 0
                colno = colno + 1
    # Now combine
    for row in range(4):
        for sq in reversed(range(1,4)):
            if (newGrid[row][sq] == newGrid[row][sq-1]):
                newGrid[row][sq] = 2*newGrid[row][sq]
                if (sq-1 == 0):
                    newGrid[row][sq-1] = 0
                else:
                    newGrid[row][sq-1] = newGrid[row][sq-2]
                    if (sq-2 == 0):
                        newGrid[row][sq-2] = 0
                    else:
                        newGrid[row][sq-2] = newGrid[row][sq-3]
                        newGrid[row][sq-3] = 0
    return newGrid
def swipeDown(Grid: List[List[int]]):
    # initialize
    newGrid = copy.deepcopy(Grid)
    # Condense first
    for col in range(4):
        for row in reversed(range(1,4)):
            rowno = row - 1
            while (rowno < 3 and newGrid[rowno+1][col] == 0):
                newGrid[rowno+1][col] = newGrid[rowno][col]
                newGrid[rowno][col] = 0
                rowno = rowno + 1
    # Now combine
    for col in range(4):
        for sq in reversed(range(1,4)):
            if (newGrid[sq][col] == newGrid[sq-1][col]):
                newGrid[sq][col] = 2*newGrid[sq][col]
                if (sq-1 == 0):
                    newGrid[sq-1][col] = 0
                else:
                    newGrid[sq-1][col] = newGrid[sq-2][col]
                    if (sq-2 == 0):
                        newGrid[sq-2][col] = 0
                    else:
                        newGrid[sq-2][col] = newGrid[sq-3][col]
                        newGrid[sq-3][col] = 0
    return newGrid

def checkHelper(Grid: List[List[int]]) -> bool:
    # Return true if nothing can be combined and game is over
    over = True
    for i in range(4):
        for j in range(4):
            if (Grid[i][j] == 0):
                over = False
    return over

def checkIfGameOver(Grid: List[List[int]]) -> bool:
    over = True
    for i in range(4):
        for j in range(4):
            if (Grid[i][j] == 0):
                over = False
    doublecheck = True
    if (over == True):
        if not (checkHelper(swipeLeft(Grid))):
            doublecheck = False
        if not (checkHelper(swipeUp(Grid))):
            doublecheck = False
        if not (checkHelper(swipeRight(Grid))):
            doublecheck = False
        if not (checkHelper(swipeDown(Grid))):
            doublecheck = False
    if (over == False):
        doublecheck = False
    return doublecheck

def getZeros(Grid: List[List[int]]):
    numZeros = 0
    for i in range(4):
        for j in range(4):
            if (Grid[i][j] == 0):
                numZeros += 1
    return numZeros

def getScore(Grid: List[List[int]]):
    numZeros = 0
    highest = 0
    bonus, bonus2, bonus3, bonus4, bonus5, bonus6, bonus7, bonus8, bonus9 = 0,0,0,0,0,0,0,0,0
    weightMatrix = [0,1,3,9,27,81,243,729,2187,6561,19683] # 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    score = 0
    for i in range(4):
        for j in range(4):
            for num in range(len(weightMatrix)):
                if (Grid[i][j] == 2**(num+1)):
                    score += weightMatrix[num]
            if (Grid[i][j] > highest):
                highest = Grid[i][j]
    # Check if highest is in a corner
    if (Grid[0][0] == highest):
        # Monotonicity
        bonus = 100
        if (Grid[0][2] <= Grid[0][1] and Grid[0][3] <= Grid[0][2]):
            bonus2 = 100
        if (Grid[2][0] <= Grid[1][0] and Grid[3][0] <= Grid[2][0]):
            bonus3 = 100
        
        if (Grid[1][1] <= Grid[1][0] and Grid[1][2] <= Grid[1][1] and Grid[1][3] <= Grid[1][2]):
            bonus4 = 80
        if (Grid[1][1] <= Grid[0][1] and Grid[2][1] <= Grid[1][1] and Grid[3][1] <= Grid[2][1]):
            bonus5 = 80
        
        if (Grid[2][1] <= Grid[2][0] and Grid[2][2] <= Grid[2][1] and Grid[2][3] <= Grid[2][2]):
            bonus6 = 30
        if (Grid[1][2] <= Grid[0][2] and Grid[2][2] <= Grid[1][2] and Grid[3][2] <= Grid[2][2]):
            bonus7 = 30

        '''if (Grid[3][1] <= Grid[3][0] and Grid[3][2] <= Grid[3][1] and Grid[3][3] <= Grid[3][2]):
            bonus8 = 5
        if (Grid[1][3] <= Grid[0][3] and Grid[2][3] <= Grid[1][3] and Grid[3][3] <= Grid[2][3]):
            bonus9 = 5'''
    elif (Grid[3][0] == highest):
        # Monotonicity
        bonus = 100
        if (Grid[1][0] <= Grid[2][0] and Grid[0][0] <= Grid[1][0]):
            bonus2 = 100
        if (Grid[3][2] <= Grid[3][1] and Grid[3][3] <= Grid[3][2]):
            bonus3 = 100
        
        if (Grid[2][1] <= Grid[3][1] and Grid[1][1] <= Grid[2][1] and Grid[0][1] <= Grid[1][1]):
            bonus4 = 80
        if (Grid[2][1] <= Grid[2][0] and Grid[2][2] <= Grid[2][1] and Grid[2][3] <= Grid[2][2]):
            bonus5 = 80

        if (Grid[2][2] <= Grid[3][2] and Grid[1][2] <= Grid[2][2] and Grid[0][2] <= Grid[1][2]):
            bonus6 = 30
        if (Grid[1][1] <= Grid[1][0] and Grid[1][2] <= Grid[1][1] and Grid[1][3] <= Grid[1][2]):
            bonus7 = 30

        '''if (Grid[2][3] <= Grid[3][3] and Grid[1][3] <= Grid[2][3] and Grid[0][3] <= Grid[1][3]):
            bonus8 = 5
        if (Grid[0][1] <= Grid[0][0] and Grid[0][2] <= Grid[0][1] and Grid[0][3] <= Grid[0][2]):
            bonus9 = 5'''
    elif (Grid[0][3] == highest):
        # Monotonicity
        bonus = 100
        if (Grid[0][1] <= Grid[0][2] and Grid[0][0] <= Grid[0][1]):
            bonus2 = 100
        if (Grid[2][3] <= Grid[1][3] and Grid[3][3] <= Grid[2][3]):
            bonus3 = 100

        if (Grid[1][2] <= Grid[1][3] and Grid[1][1] <= Grid[1][2] and Grid[1][0] <= Grid[1][1]):
            bonus4 = 80
        if (Grid[1][2] <= Grid[0][2] and Grid[2][2] <= Grid[1][2] and Grid[3][2] <= Grid[2][2]):
            bonus5 = 80

        if (Grid[2][2] <= Grid[2][3] and Grid[2][1] <= Grid[2][2] and Grid[2][0] <= Grid[2][1]):
            bonus6 = 30
        if (Grid[1][1] <= Grid[0][1] and Grid[2][1] <= Grid[1][1] and Grid[3][1] <= Grid[2][1]):
            bonus7 = 30

        '''if (Grid[3][2] <= Grid[3][3] and Grid[3][1] <= Grid[3][2] and Grid[3][0] <= Grid[3][1]):
            bonus8 = 5
        if (Grid[1][0] <= Grid[0][0] and Grid[2][0] <= Grid[1][0] and Grid[3][0] <= Grid[2][0]):
            bonus9 = 5'''
    elif (Grid[3][3] == highest):
        # Monotonicity
        bonus = 100
        if (Grid[1][3] <= Grid[2][3] and Grid[0][3] <= Grid[1][3]):
            bonus2 = 100
        if (Grid[3][1] <= Grid[3][2] and Grid[3][0] <= Grid[3][1]):
            bonus3 = 100

        if (Grid[2][2] <= Grid[2][3] and Grid[2][1] <= Grid[2][2] and Grid[2][0] <= Grid[2][1]):
            bonus4 = 80
        if (Grid[2][2] <= Grid[3][2] and Grid[1][2] <= Grid[2][2] and Grid[0][2] <= Grid[1][2]):
            bonus5 = 80

        if (Grid[1][2] <= Grid[1][3] and Grid[1][1] <= Grid[1][2] and Grid[1][0] <= Grid[1][1]):
            bonus6 = 30
        if (Grid[2][1] <= Grid[3][1] and Grid[1][1] <= Grid[2][1] and Grid[0][1] <= Grid[1][1]):
            bonus7 = 30
        
        '''if (Grid[0][2] <= Grid[0][3] and Grid[0][1] <= Grid[0][2] and Grid[0][0] <= Grid[0][1]):
            bonus8 = 5
        if (Grid[2][0] <= Grid[3][0] and Grid[1][0] <= Grid[2][0] and Grid[0][0] <= Grid[1][0]):
            bonus9 = 5'''
    return (bonus + bonus2 + bonus3 + bonus4 + bonus5 + bonus6 + bonus7 + bonus8 + bonus9 + 2*score)
                
def newBoard(Grid: List[List[int]], numzeros):
    children = []
    weights = []
    for row in range(4):
        for col in range(4):
            if (Grid[row][col] == 0):
                # copy grid to 2 new children
                newGrid1 = copy.deepcopy(Grid)
                newGrid2 = copy.deepcopy(Grid)

                # New possibility of this blank tile becoming a 2
                newGrid1[row][col] = 2
                children.append(newGrid1)
                weights.append(0.9 * 1/numzeros)
                # New possibility of this blank tile becoming a 4
                newGrid2[row][col] = 4
                children.append(newGrid2)
                weights.append(0.1 * 1/numzeros)
    return children, weights

def calcMoves(Grid: List[List[int]]):
    workwith = copy.deepcopy(Grid)
    children2vals = []
    children2Boards = []
    # 5 possible moves for this new board
    h0 = swipeUp(workwith)
    g0 = getScore(h0) # Up
    h1 = swipeDown(workwith)
    g1 = getScore(h1) # Down
    h2 = swipeLeft(workwith)
    g2 = getScore(h2) # Left
    h3 = swipeRight(workwith)
    g3 = getScore(h3) # Right
    h4 = workwith
    g4 = getScore(workwith) # Nothing
    
    children2vals.extend([g0,g1,g2,g3,g4])
    children2Boards.extend([h0,h1,h2,h3,h4])
    return children2vals, children2Boards

def afterBestMove(Grid: List[List[int]]):
    up = getScore(swipeUp(Grid))
    down = getScore(swipeDown(Grid))
    left = getScore(swipeLeft(Grid))
    right = getScore(swipeRight(Grid))
    nothing = getScore(Grid)
    best = max(up, down, left, right, nothing)
    if (up == best):
        return up
    if (down == best):
        return down
    if (left == best):
        return left
    if (right == best):
        return right
    else: return nothing


def goDeep(Grid: List[List[int]], depth):
    # Want to return all possible boards and their associated probabilities
    # For first time, initialize
    (possBoards, oldWeights) = newBoard(Grid, getZeros(Grid))
    bestMoves = []
    for board in possBoards:
        bestMoves.append(afterBestMove(board))
    # Now have a list of boards - bestMoves, and list of their probabilities - possWeights
    '''for i in range(depth-1):
        newBBs = []
        newWs = []
        for b in range(len(bestMoves)):
            (newBoards, newWeights) = newBoard(bestMoves[b], getZeros(bestMoves[b]))
            for b1 in range(len(newBoards)):
                proposedMove = afterBestMove(newBoards[b1])
                if (proposedMove not in newBBs):
                    newBBs.append(proposedMove)
                    newWs.append(oldWeights[b]*newWeights[b1])
        # Now have big list of newBBs and newWs
        bestMoves = newBBs
        oldWeights = newWs'''
    # Outside of for loop now have big list of newBBs and newWs
    return bestMoves, oldWeights

def heuristic(Grid: List[List[int]]):
    start = copy.deepcopy(Grid)
    depth = 1
    (allStates, allProbs) = goDeep(start,depth)
    finalValue = 0
    for i in range(len(allStates)):
        finalValue += allStates[i] * allProbs[i]
    return finalValue


def NextMove(Grid: List[List[int]], Step: int) -> int:
    if (checkIfGameOver(Grid) == False):
        # Game is not over
        print("--------------------")
        print("Current Board")
        if (Step > 900): printGrid(Grid)
        if (Step > 2000) : print("Step: " + str(Step))
        upScore = heuristic(swipeUp(Grid))
        print("Up: " + str(upScore))
        downScore = heuristic(swipeDown(Grid))
        print("Down: " + str(downScore))
        leftScore = heuristic(swipeLeft(Grid))
        print("Left: " + str(leftScore))
        rightScore = heuristic(swipeRight(Grid))
        print("Right: " + str(rightScore))
        doNothing = heuristic(Grid)
        print("Do nothing: " + str(doNothing))

        best = max(leftScore, upScore, rightScore, downScore, doNothing)
        if (upScore == best):
            return 0
        if (downScore == best):
            return 1
        if (leftScore == best):
            return 2
        if (rightScore == best):
            return 3
        return 5 # do nothing
        
    else: return 4



if (__name__ == '__main__'):
    grid = [[4,2,4,0], [4,2,8,0], [0,2,0,8], [0,2,4,16]]
    NextMove(grid,0)