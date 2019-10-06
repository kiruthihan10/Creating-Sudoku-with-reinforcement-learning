# Creating-Sudoku-with-reinforcement-learning
Creating a sudoku board is a challenging one.
It may even consider as the hardest level of sudoku game.
Because at one point it has to come across a standard sudoku game.
Moreover it has a possibility of failing at an unsolvalble sudoku.

# The environment
I created a basic environment for the game to train.
The observation of the environment is a (9,9,9) numpy array.
The dimesions are row, column, and the new value of the selected cell.
Instead of adding number in the selected cell, Ive created an 1 dim array to store the value.
By doing so I Indicated that each number should not be considered as number but only as a symbol.
When an action is inputed it will check whether that number can be inputed in the specific cell.
Because there shouldn't be another cell in the same row or column or in the mini box.
According to these the rewardsare given.

# The Rewards

|Action|Reward|Reason|
|---|---|---|
|Each Action|-1|To reduce the game size|
|Wrong Range|-200|To give input in the sudoku board range|
|Selected Number in Row or Column|-100|To let the systen learn that there shouldn't be two cells with same number in same row or colums |
|Selected Number in miniBox|-50|To let the systen learn that there shouldn't be two cells with same number in same minibox|
|Time out|-1000|To prevent the system run forever|
|Win|1000|To encourage the system to win|

# The Network
Used Tensorflow as Deeplearning library
Created in Google Colab
The q network is structured as follows.

|Input Shape|Layer|Activation|Output Shape|
|---|---|---|---|
|9,9,9|Convolution(Kernel size=(9,1),filters=9)|Relu|1,9,9|
|1,9,9|Convolution(Kernel size=(1,9),filters=9)|Relu|1,1,9|
|1,1,9|Convolution(Kernel size=(9,1),filters=9)|Relu|1,1,9|
|1,1,9|Reshape|None|9|
|9|Fully Connected(Batch Normalized)|Relu|81|
|81|Fully Connected(Batch Normalized)|Relu|729|
|729|Outputs|None|729|

Learning rate = 0.01
Optimizer     = Adam Optimizer
batch size    = 50

Used Epsilon Greedy Alogrithm to try various parts of the sudoku board
Epsilon range(0.05-1)

The output is 1dim with 729 length because each gives a value and id of a cell.
[0,0,0],[0,0,1],......[0,0,9],[0,1,0],....[0,9,9],[1,0,0],....[9,9,8],[9,9,9]
