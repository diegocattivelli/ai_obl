from agent import Agent
from board import Board
import math


class MinimaxAgentAlphaBetaPruning(Agent):

    def __init__(self, player=1, depth=3, eval_fn=None):
        super().__init__(player)
        self.depth = depth
        self._nodes_explored = 0
        # Si no se pasa eval_fn, importa el baseline para no romper el código existente
        if eval_fn is None:
            from heuristics import eval_baseline
            self._eval_fn = eval_baseline
        else:
            self._eval_fn = eval_fn

    def next_action(self, obs: Board):
        self._nodes_explored = 0
        action, _ = self._minimax(obs, self.player, self.depth, -math.inf, math.inf)
        return action

    def heuristic_utility(self, board: Board) -> float:
        return self._eval_fn(board, self.player)

    def _minimax(self, board, current_player, depth, alpha, beta):
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
        best_action = None
        next_player = 3 - current_player

        if current_player == self.player:
            if not actions:
                return None, -math.inf
            best_value = -math.inf
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, value = self._minimax(child, next_player, depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if best_value >= beta:
                    break
            return best_action, best_value
        else:
            if not actions:
                return None, math.inf
            best_value = math.inf
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, value = self._minimax(child, next_player, depth - 1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if best_value <= alpha:
                    break
            return best_action, best_value