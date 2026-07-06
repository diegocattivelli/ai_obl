from agent import Agent
from board import Board
import math


class MinimaxAgent(Agent):

    def __init__(self, player=1, depth=3):
        super().__init__(player)
        self.depth = depth
        self._nodes_explored = 0

    def next_action(self, obs: Board):
        self._nodes_explored = 0
        action, _ = self._minimax(obs, self.player, self.depth)
        return action

    def heuristic_utility(self, board: Board) -> float:
        opponent = 3 - self.player

        my_moves = len(board.get_possible_actions(self.player))
        opp_moves = len(board.get_possible_actions(opponent))
        h1 = my_moves - opp_moves

        my_pos = board.find_player_position(self.player)
        center = (board.board_size[0] // 2, board.board_size[1] // 2)
        h2 = -(abs(my_pos[0] - center[0]) + abs(my_pos[1] - center[1]))

        opp_pos = board.find_player_position(opponent)
        h3 = -(abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1]))

        return h1 + h2 + h3

    def _minimax(self, board: Board, current_player: int, depth: int):
        self._nodes_explored += 1

        is_end, winner = board.is_end(current_player)

        if is_end:
            if winner == self.player:
                return None, math.inf
            elif winner == 3 - self.player:
                return None, -math.inf
            else:
                return None, 0

        if depth == 0:
            return None, self.heuristic_utility(board)

        actions = board.get_possible_actions(current_player)
        next_player = 3 - current_player
        best_action = None

        if current_player == self.player:

            if not actions:
                return None, -math.inf

            best_value = -math.inf

            for action in actions:
                child = board.clone()
                child.play(action, current_player)

                _, value = self._minimax(
                    child,
                    next_player,
                    depth - 1,
                )

                if value > best_value:
                    best_value = value
                    best_action = action

            return best_action, best_value

        else:

            if not actions:
                return None, math.inf

            best_value = math.inf

            for action in actions:
                child = board.clone()
                child.play(action, current_player)

                _, value = self._minimax(
                    child,
                    next_player,
                    depth - 1,
                )

                if value < best_value:
                    best_value = value
                    best_action = action

            return best_action, best_value