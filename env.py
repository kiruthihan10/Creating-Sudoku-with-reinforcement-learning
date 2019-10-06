class sudoku:

    import numpy as np

    def __init__(self):
        import time
        self.reset()
        self.t = time.time()


    def reset(self):
        print("Board Reseted!!")
        import numpy as np
        self.board = np.zeros((9,9,9))  ## Create board with all blank boxes in a shape of (9,9,9)
                                        ## For 9 Row, 9 Column, and 9 available numbers
                                        ## Created an array for numbers instead inputing number value there. Because it may give network a feel like there is a connection between each numbers, which is not.
                                        ## If no number selected n = [0,0,0,0,0,0,0,0,0]
                                        ## If any number selected n[x] = 1 where x is the selected number
        self.done = False
        self.info = None
        return self.board

    def visualizer(self):
        import numpy as np
        visual_board = np.zeros((9,9))  ## Create a temprary board with a shape of (9,9)
                                        ## For 9 row and 9 column
        for r in range(0,9):
            for c in range(0,9):
                if 1 in self.board[r,c]: ## Check whether there is a number in the box or not
                    visual_board[r,c] = np.where(self.board[r,c]==1)[0][0]+1 ## If so it will put the number in the board
                else:
                    visual_board[r,c] = 0 ## Else it will put 0 if it's blank
        print(visual_board)             ## Print the board


    def step(self,inp):
        ## Args:
            ## inp : 1 dim array of 3 values. which indicates the row, column, number
        self.reward = 0
        import time
        self.reward -= 1    ## -1 rewards for each action
        #self.visualizer()
        import numpy as np
        row = inp[0]
        col = inp[1]
        num = inp[2]
        changable = True
        end = False
        if row<0 or row>8 or col<0 or col>8 or num<1 or num>9:
            print("Wrong Range")
            ## Check the row and column values are permited
            self.reward-=200    ##If wrong input then reward will be reduced by 200
            changable = False
            end = False
        else:
            selected_number = np.zeros(9)
            selected_number[num-1] = 1
            r_set = int(row/3)
            c_set = int(col/3)
            ## Check the Row
            row_check = (selected_number == self.board[row]).all(1).any()
            col_check = (selected_number == self.board[:,col]).all(1).any()
            if row_check or col_check:
                ## If there is a same number in row or column reward reduced by 100
                print("Selected Number in ROW OR COLUMN")
                self.reward -= 100
                changable = False
            ## Check the Box
            ## print("Selected Number")
            ## print(selected_number)
            box = self.board[r_set*3:r_set*3+3,c_set*3:c_set*3+3]
            ## Select the mini box the number box in it
            box_check = False
            for ROW in box:
                for NUM in ROW:
                    ##print(NUM)
                    if list(NUM) == list(selected_number):
                        box_check = True
                        break
                break
            if box_check:
                print("Selected Number in Box")
                ## If the number already existed in the mini box the reward will be reducted by 50
                self.reward -= 50
                changable = False
            if changable == True:
                print("Changing...")
                self.board[row,col]=np.zeros(9)
                self.board[row,col,num-1]=1
                self.visualizer()
                ##print([row,col,num])
            if np.zeros(9) not in self.board:
                end = True
            if time.time()-self.t > 5*60:
                ## If game time existed 5 min the game counts as loss. This prevent infinite game if the system never add or change a specific box of number
                end = True
                ## If that happens the reward will be reduced by 1000
                self.reward -= 1000
            if end == True:
                ##If the game won the reward increased by 1000
                self.done = True
                self.reward += 1000

        return (self.board,self.reward,self.done,self.info)
