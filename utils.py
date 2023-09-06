import numpy as np

################ TIC TAC TOE RELATED FUNCTIONS ################

# Defining a function to check the win state of the 3x3 board 
# This will include a lot of manual conditions

def is_win(board):
    
    if board[0,0] != 0:
        if board[0,0] == board[0,1] and board[0,1] == board[0,2]:
            return board[0,0]
        elif board[0,0] == board[1,1] and board[1,1] == board[2,2]:
            return board[0,0]
        elif board[0,0] == board[1,0] and board[1,0] == board[2,0]:
            return board[0,0]
            
    if board[2,2] != 0:
        if board[0,2] == board[1,2] and board[1,2] == board[2,2]:
            return board[0,2]
        elif board[2,0] == board[2,1] and board[2,1] == board[2,2]:
            return board[2,2]
    
    if board[0,2] != 0:
        if board[0,2] == board[1,1] and board[1,1] == board[2,0]:
            return board[0,2]

    if board[1,1] != 0:
        if board[1,0] == board[1,1] and board[1,1] == board[1,2]:
            return board[1,1]
        if board[0,1] == board[1,1] and board[1,1] == board[2,1]:
            return board[1,1]

    return 0
        
# This function just takes the backend board of integers and converts it to a slightly more traditional board
# It's a bit janky right now but will update later

def print_board(board):
    printed_board = np.array([["", "", ""], ["","", ""], ["", "", ""]], dtype = str)
    symbols = ['O', ' ', 'X']

    for i in range(3):
        for j in range(3):
            printed_board[i,j] = symbols[int(board[i,j] + 1)]
    return printed_board

# This function updates the board position according to the player number

def update_board(board, pos, player):
    pos = pos-1
    moves = np.array([1,-1])
    board[int((pos - pos%3)/3), pos%3] = moves[(player + 1)%2]
    player = (player % 2) + 1
    return board, player

# This function takes in a players input when required. 
# We also make sure to check that the position is within the board
# As well as a check for whether the position on the board is already filled in

def player_input(board, player):
    if player == 1:
        # Player 1 places Xs
        pos = int(input('Where would you like to put an X \n'))
        while pos > 9 or pos <1:
            # Check if the position is within the board
            pos = int(input('That is not a valid board position. Enter a board position for X \n'))
        pos = pos-1
        while board[int((pos - pos%3)/3), pos%3] != 0:
            # Check if the position is already filled in
            pos = int(input('That is not an empty square. Enter a board position for X \n'))
            pos=pos-1
        pos = pos+1
    else:
        # Player 2 places Os
        pos = int(input('Where would you like to put an O \n'))
        while pos > 9 or pos <1:
            # Check if the position is within the board
            pos = int(input('That is not a valid board position. Enter a board position for O \n'))
        pos = pos-1
        while board[int((pos - pos%3)/3), pos%3] != 0:
            # Check if the position is already filled in
            pos = int(input('That is not an empty square. Enter a board position for O \n'))
            pos=pos-1
        pos = pos+1
    
    return pos

################ REINFORCEMENT LEARNING RELATED FUNCTIONS ################

# Defining a function to add new scores for unseen states to the chart as well as the list of states themselves

def add_score(states, scores, board):

    possible_scores = [0, 0.5, 1]
    append_board = board.reshape((1,3,3))
    states = np.vstack((states, append_board))
    scores = np.append(scores, possible_scores[int(is_win(board)+1)])
    return states, scores

# Defining a function to lookup scores. To-do: Implement a cleaner, list-comprehension version of this

def lookup_score(states, scores, board):
    for i in range(states.shape[0]):
        if (states[i] == board).all():
            return scores[i]

# Defining a function to update scores of a previous state given the current state after a move has been made

def update_score(prev_board, cur_board, states, scores, learn_rate):
    for prev_index in range(states.shape[0]): 
        if (states[prev_index] == prev_board).all():
            for cur_index in range(states.shape[0]):
                if (states[cur_index] == cur_board).all():
                    scores[prev_index] = scores[prev_index] + learn_rate*(scores[cur_index] - scores[prev_index])
                    return scores
                
# Defining a function to update final score of a previous state 

def update_final_score(prev_board, states, scores, learn_rate, win_value):
    for prev_index in range(states.shape[0]): 
        if (states[prev_index] == prev_board).all():
            scores[prev_index] = scores[prev_index] + learn_rate*(win_value - scores[prev_index])
            return scores
        
# Inefficient function to see if we have already seen a board state
# Need to fix this with either list comprehension or hash table later

def is_in(board, states):
    for known_state in states[:]:
        if (board == known_state).all():
            return True
    return False


# We need to define a rule for generating a new move for the training phase of the model
# We will pick moves based on the current state of the board and
# the possible board states we can move to from there

def train_move_gen(board, states, scores, player, explore_rate):

    valid_pos = np.array([], dtype= int)

    max_score = -1
    max_pos = 1

    min_score = 2
    min_pos = 1

    for i in range(3):
        for j in range(3):

            cur_pos = int( i*3 + j + 1)

            copy_board = board.copy() 

            # Checking if the position is already filled in
            if copy_board[i,j] != 0:
                continue

            # Checking if the position is empty
            if copy_board[i,j] == 0:
                valid_pos = np.append(valid_pos, cur_pos)

            copy_board = update_board(copy_board, cur_pos, player)[0]

            # Adding the new board to the list of states if it is not already there
            if not is_in(copy_board, states):
                states, scores = add_score(states, scores, copy_board)

            cur_score = lookup_score(states, scores, copy_board)
            
            if cur_score >= max_score:
                # Updating the max score and position
                max_pos = cur_pos
                max_score = cur_score

            if cur_score <= min_score:
                # Updating the min score and position
                min_pos = cur_pos
                min_score = cur_score
    
    # Random variable for exploration
    test = np.random.uniform(0,1)

    if test >= explore_rate and valid_pos.size > 1:
        pos = valid_pos[np.random.random_integers(valid_pos.size-1)]
        return states, scores, pos
    
    if player == 1:
        return states, scores, max_pos
    else:
        return states, scores, min_pos


# Define a function to generate a move when playing an actual game
# Most of this is the same as the training move gen function but without the 
# exploration rate. We can also just combine these two functions later

def move_gen(states, scores, board, computer): 

    max_score = -1
    min_score = +2
    min_pos = 1
    max_pos = 1

    for i in range(3):
        for j in range(3):

            cur_pos = int( i*3 + j + 1)

            copy_board = board.copy()

            if copy_board[i,j] != 0:
                continue

            copy_board = update_board(copy_board, cur_pos, computer)[0]

            if not is_in(copy_board, states):
                states, scores = add_score(states, scores, copy_board)

            cur_score = lookup_score(states, scores, copy_board)
            
            if cur_score >= max_score:
                max_pos = cur_pos
                max_score = cur_score

            if cur_score <= min_score:
                min_pos = cur_pos
                min_score = cur_score

    if computer == 1:
        return states, scores, max_pos
    else:
        return states, scores, min_pos
    
    # Define a function to see if the game has ended during the training phase

def is_ended_train(board):

    state = is_win(board)
    
    if state == 0:
        if not(0 in board):
            return True
        return False
    else:
        return True
    

# Define a function to see if the game has ended during an actual game

def is_ended(board):

    state = is_win(board)

    if state == 0:
        if not(0 in board):
            print ("Game over! The final board is \n", print_board(board) , "\n and it's a tie")
            return True
        return False
    else:
        print ("Game over! The final board is \n", print_board(board), "\n and the winner is ")
        if state == 1:
            print ('Player 1')
        else:
            print ('Player 2')
        return True
        
