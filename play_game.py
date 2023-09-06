import numpy as np
from utils import *

# Import the models

def import_models():
    states_p1 = np.load('states_p1.npy')
    scores_p1 = np.load('scores_p1.npy')
    states_p2 = np.load('states_p2.npy')
    scores_p2 = np.load('scores_p2.npy')

    return states_p1, scores_p1, states_p2, scores_p2

# Player vs Computer
def player_vs_computer():

    states_p1, scores_p1, states_p2, scores_p2 = import_models()

    game_over = False
    player = int(input("Do you want to be player 1 or 2? \n"))
    computer = 3 - player

    if computer == 1:
        states = states_p1
        scores = scores_p1
    else:
        states = states_p2
        scores = scores_p2

    board = np.zeros((3,3))

    cur_move = 1

    position_vec = np.array([1,2,3,4,5,6,7,8,9])
    print("The positions are \n", position_vec.reshape(3,3))

    while not game_over:

        print("The current board is \n", print_board(board))
        print("\n")

        if player == cur_move:
            pos = player_input(board, player)

        if computer == cur_move:
            states, scores, pos = move_gen(states, scores, board, computer)

        board, cur_move = update_board(board, pos, cur_move)
        
        game_over = is_ended(board) 

        if game_over:
            if is_win(board) == 1:

                print ('Player 1 wins!')

            elif is_win(board) == -1:
                print ('Player 2 wins!')
            else:
                print ('Tie!')

            print(print_board(board))
        
# Player vs Player
def player_vs_player():

    game_over = False
    player = 1

    board = np.zeros((3,3))

    while not game_over:

        print("The current board is \n", print_board(board))

        pos = player_input(board, player)
        board, player = update_board(board, pos, player)
        
        state = is_win(board)

        if state == 0:
            if not(0 in board):
                print ("Game over! The final board is \n", print_board(board) , "\n and it's a tie")
                game_over = True
            continue
        else:
            print ("Game over! The final board is \n", print_board(board), "\n and the winner is ")
            game_over = True
            if state == 1:
                print ('Player 1')
            else:
                print ('Player 2')
        

if __name__ == '__main__':

    player_vs_computer()
    #player_vs_player()