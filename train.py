import numpy as np
from utils import *


# Run this function to just play one training game to test all the training functions
# in the utils.py file
def train_example():

    #### HYPERPARAMETERS ####

    learn_rate = 0.3
    explore_rate = 0.3

    #### Initialize states and scores ####

    states_p1 = np.zeros((1,3,3))
    scores_p1 = np.array([0.5])

    for i in range(9):
        board = np.zeros((3,3))
        board[int(i/3), i%3] = 1
        states_p1, scores_p1 = add_score(states_p1, scores_p1, board)


    states_p2 = np.zeros((1,3,3))
    scores_p2 = np.array([0.5])

    #### Game conditions ####

    board = np.zeros((3,3))
    player = 1
    game_over = False
    cur_board = board.copy()
    iteration = 0

    #### BEGIN TEST ####

    while not game_over:

        iteration += 1
        
        # Updte the board and player

        board = cur_board.copy()
        cur_player = player

        # Generate a move and store the current state for future score updates before making the move

        if cur_player == 1:
            states_p1, scores_p1, pos = train_move_gen(board, states_p1, scores_p1, cur_player, explore_rate)
            prev_board_p2 = board.copy()
        else:
            states_p2, scores_p2, pos = train_move_gen(board, states_p2, scores_p2, cur_player, explore_rate)
            prev_board_p1 = board.copy()
        
        # Player makes the move and updates the board (and player number)

        cur_board, player = update_board(board, pos, cur_player)

        # Update the scores of the state that led to the current state

        if cur_player == 1 and iteration > 1:
            scores_p1 = update_score(prev_board_p1, cur_board, states_p1, scores_p1, learn_rate)
        elif cur_player == 2:
            scores_p2 = update_score(prev_board_p2, cur_board, states_p2, scores_p2, learn_rate)

        # Check if the game has ended
        
        game_over = is_ended_train(cur_board)

        # Adding the end state to the states and scores of the losing player ####
        
        if game_over:
            if is_win(cur_board) == 1:
                print ('Player 1 wins!')
                scores_p2 = update_final_score(prev_board_p2, states_p2, scores_p2, learn_rate, 1)
            elif is_win(cur_board) == -1:
                print ('Player 2 wins!')
                scores_p1 = update_final_score(prev_board_p1, states_p1, scores_p1, learn_rate, 0)
            else:
                print ('Tie!')

                if cur_player == 1:
                    scores_p2 = update_final_score(prev_board_p2, states_p2, scores_p2, learn_rate, 0.5)
                else:
                    scores_p1 = update_final_score(prev_board_p1, states_p1, scores_p1, learn_rate, 0.5)
        
        # Print the current board

        print(print_board(cur_board))

# Function for training the reinforcement learning agent
# Run this function to train the agent over 100000 games

def train():

    #### HYPERPARAMETERS ####

    learn_rate = 0.6
    explore_rate = 0.6
    num_games = 10000

    #### Initialize states and scores ####

    states_p1 = np.zeros((1,3,3))
    scores_p1 = np.array([0.5])

    for i in range(9):
        board = np.zeros((3,3))
        board[int(i/3), i%3] = 1
        states_p1, scores_p1 = add_score(states_p1, scores_p1, board)

    states_p2 = np.zeros((1,3,3))
    scores_p2 = np.array([0.5])

    #### BEGIN TRAINING ####

    for i in range(num_games):

        if i%200 == 0:
            print("Game ", i)

        # Reset the game conditions for each game

        board = np.zeros((3,3))
        player = 1
        game_over = False
        cur_board = board.copy()
        iteration = 0

        while not game_over:

            iteration += 1
            
            # Updte the board and player

            board = cur_board.copy()
            cur_player = player

            # Generate a move and store the current state for future score updates before making the move

            if cur_player == 1:
                states_p1, scores_p1, pos = train_move_gen(board, states_p1, scores_p1, cur_player, explore_rate)
                prev_board_p2 = board.copy()
            else:
                states_p2, scores_p2, pos = train_move_gen(board, states_p2, scores_p2, cur_player, explore_rate)
                prev_board_p1 = board.copy()
            
            # Player makes the move and updates the board (and player number)

            cur_board, player = update_board(board, pos, cur_player)

            # Update the scores of the state that led to the current state

            if cur_player == 1 and iteration > 1:
                scores_p1 = update_score(prev_board_p1, cur_board, states_p1, scores_p1, learn_rate)
            elif cur_player == 2:
                scores_p2 = update_score(prev_board_p2, cur_board, states_p2, scores_p2, learn_rate)

            # Check if the game has ended
            
            game_over = is_ended_train(cur_board)

            # Adding the end state to the states and scores of the losing player ####
            
            if game_over:
                if is_win(cur_board) == 1:
                    scores_p2 = update_final_score(prev_board_p2, states_p2, scores_p2, learn_rate, 1)
                elif is_win(cur_board) == -1:
                    scores_p1 = update_final_score(prev_board_p1, states_p1, scores_p1, learn_rate, 0)
                else:
                    if cur_player == 1:
                        scores_p2 = update_final_score(prev_board_p2, states_p2, scores_p2, learn_rate, 0.5)
                    else:
                        scores_p1 = update_final_score(prev_board_p1, states_p1, scores_p1, learn_rate, 0.5)

    print("Training done!")

    # Save the states and scores
    np.save('states_p1.npy', states_p1)
    np.save('scores_p1.npy', scores_p1)
    np.save('states_p2.npy', states_p2)
    np.save('scores_p2.npy', scores_p2)


if __name__ == '__main__':
    # train_example()
    train()