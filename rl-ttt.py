"""
Code description:
it's 3-by-3 tic tac toe provided by Prof.Mate.
It include reinforcement learning.
https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/TicTacToe/ticTacToe.py
"""

import numpy as np
import pickle # data converter from binary to readable data

BOARD_ROWS = 4 # number of row
BOARD_COLS = 4 # number of column 

class State:
    def __init__(self, p1, p2):#initialzie the gameboard
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))# make a board with row by column and fill 0 into it.
        self.p1 = p1 # computer 1
        self.p2 = p2 # computer 2
        self.isEnd = False # flag judging end
        self.boardHash = None # 
        self.playerSymbol = 1 # p1 plays first

        #self.board: example
        # [[0],[0],[-1],[0]],
        # [[-1],[0],[0],[0]],
        # [[0],[0],[0],[0]],
        # [[0],[1],[1],[0]]
            # 0 => no piece
            # 1 => p1 piece
            # -1 => p2 piece
    
    def getHash(self):# get status of current board state
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        # example: self.boardHash = "[-1.  0.  0.  1.  1. -1.  1. -1.  1. 1. -1. -1. 0. 0. 0. 0.]"

        aa = {"aa":10}
        aa.get("aa") => return: 10

        return self.boardHash

    #check if player matches win condition and decide winner player(if winner exits)
    # return 1 => p1 win
    # return -1 => p2 win
    # return 0 => tie
    def winner(self):
        # use sum to check win
        # Examples:
        # [[0],[0],[0],[-1]],
        # [[0],[-1],[-1],[0]],
        # [[0],[-1],[0],[0]],
        # [[1],[1],[1],[1]]
            # sum of row4 is 4(number of column) => p1 winner
        # [[0],[-1],[0],[0]],
        # [[1],[-1],[0],[0]],
        # [[0],[-1],[1],[0]],
        # [[1],[-1],[0],[1]]
            # sum of col2 is -4(number of row) => p2 winner

        # check line on row: example
        for i in range(BOARD_COLS):
            if sum(self.board[i, :]) == BOARD_COLS:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -BOARD_COLS:
                self.isEnd = True
                return -1
        
        # check line in col
        for i in range(BOARD_ROWS):
            if sum(self.board[:, i]) == BOARD_ROWS:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -BOARD_ROWS:
                self.isEnd = True
                return -1
        
        # check line in diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(BOARD_COLS)])
        diag_sum2 = sum([self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        diag_length = min(BOARD_COLS,BOARD_ROWS); # just in case Row and 
        if diag_sum == diag_length:
            self.isEnd = True
            if diag_sum1 == diag_length or diag_sum2 == diag_length:
                return 1 # winner: p1
            else:
                return -1 # winner: p2

        if len(self.availablePositions()) == 0:# no available positions: tie
            self.isEnd = True # game end
            return 0 # tie

        self.isEnd = False # not end
        return None # not end

    def availablePositions(self): # returns positions available(not placed)
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0: # if not placed
                    positions.append((i, j))  # add it into array
        return positions
    
    def updateState(self, position):# get status of current board state
        self.board[position] = self.playerSymbol # put a piece into specifed place
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1 # switch to another player

    # give reward (reinforcement task)
    def giveReward(self):
        result = self.winner()
        # backpropagate reward
        if result == 1: #winner:p1 => give 1 to p1, 0 to p2
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1: #winner:p2 => give 1 to p2, 0 to p1
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:              #tie => give 0.1 to p1, 0.5 to p2
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def reset(self): # reset every state
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS)) # init board and fill 0
        self.boardHash = None # none
        self.isEnd = False # not end
        self.playerSymbol = 1 # p1's turn


    def play(self, rounds=50000): # training function: computer vs computer
        """ don't care about these codes
        color_dic = {"green":"\033[32m",
                     "yellow":"\033[33m",
                     "blue":"\033[34m",
                     "magenta":"\033[35m",
                     "cyan":"\033[36m"}
        dic_key = list(color_dic.values())
        k = 0"""
        for i in range(rounds):#playing for rounds times
            if i % 1000 == 0: # every 1000 times training, printing color changes.
                print("Rounds {}".format(i))
                """
                print(dic_key[k]+"Rounds {}".format(i))
                k+=1
                if(k == len(dic_key)):
                    k = 0;"""
            while not self.isEnd: # when geme is bor over, keep playing.
                # Player 1
                positions = self.availablePositions() # get positions available
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol) # p1 action
                self.updateState(p1_action) # update state
                board_hash = self.getHash() # get current board
                self.p1.addState(board_hash) # store it into p1.states(where stores training data before putting into policy files)
                win = self.winner() # check board status if it is end
                if win is not None: # not None => game is over: ended with p1 either win or draw
                    self.giveReward() # give reward
                    self.p1.reset() # init p1's status
                    self.p2.reset() # init p2's status
                    self.reset() # init baord status
                    break

                else:
                    # Player 2 same process as Player1
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None: # ended with p2 either win or draw
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
    
    def play2(self):# play with human, almost same proces as play1
        while not self.isEnd: 
            # Player 1
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            # take action and upate board state
            self.updateState(p1_action)
            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break

    def showBoard(self): # show current game board
        # p1: x  p2: o
        for i in range(0, BOARD_ROWS):
            for p in range(BOARD_COLS):
                print('----',end="")
            print("-")
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        for p in range(BOARD_COLS):
            print('----',end="")
        print("-")


class Player:
    def __init__(self, name,exp_rate=0.3):
        self.name = name # player name
        self.states = []  # record all positions taken
        self.lr = 0.2 # influence of total reward value (large influence)
        self.exp_rate = exp_rate #random rate
        self.decay_gamma = 0.9 # influence of specified reward value(small influence)
        self.states_value = {}  # use board hash and get its value.

    def getHash(self, board):# same thing
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    def chooseAction(self, positions, current_board, symbol): # choose action
        if np.random.uniform(0, 1) <= self.exp_rate: # set random rate to train widely
            # take random action
            idx = np.random.choice(len(positions)) # random choice
            action = positions[idx] # put it into array
        else:# select action based on "value" of training
            #print(len(self.states_value.keys()))
            value_max = -9999999999999999
            for p in positions: # pick up each position available.
                next_board = current_board.copy() # get copy 
                next_board[p] = symbol # put player symbol into a position available
                next_boardHash = self.getHash(next_board) # get board hash
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash) # set value of the position
                #print("value", value)
                if value >= value_max: # compare the value, and find the position with maximum value
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    # append a hash state
    def addState(self, state):# add states
        self.states.append(state)

    def feedReward(self, reward):# at the end of game, backpropagate and update states value
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st]) # calculate real reward, formula: backpropagate method
            reward = self.states_value[st] # real reward

    def reset(self): # reset
        self.states = []

    def savePolicy(self): # put training data into policy file, policy, note : file name is 'policy_ + self.name(player name)'
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file): # load policy file
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()


class HumanPlayer: # human
    def __init__(self, name):
        self.name = name # human name

    def chooseAction(self, positions): # choose action
        while True:
            row = int(input("Input your action row:"))# choose row of place 
            col = int(input("Input your action col:"))# choose col of place
            action = (row, col)
            if action in positions: # check if the palce is available
                return action

    def addState(self, state): # no states of human
        pass

    def feedReward(self, reward): # no reward about human
        pass

    def reset(self): # no reset about human
        pass

if __name__ == "__main__":
    p1 = Player("p1_4by4_100000") # set player1's name
    p2 = Player("p2") # set player2's name
    st = State(p1, p2) # create an game board

    print("training...")
    st.play(100000)# training: default 50000
    p1.savePolicy() # save training data after training
    print("\033[0m"+"---------Finished training-------")


    policy_name = p1.name
    p1 = Player("computer",exp_rate=0)
    p1.loadPolicy("policy_"+policy_name) # read training data file: policy_ + p1.name
    #p1.loadPolicy("policy_p1")
    p2 = HumanPlayer("human") # create human

    st = State(p1, p2) 
    st.play2() # human vs computer
