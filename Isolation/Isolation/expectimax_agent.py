from agent import Agent
from board import Board
import math


class ExpectimaxAgent(Agent):

    def __init__(self, player=1, depth=3, eval_fn=None):
        super().__init__(player)
        self.depth = depth
        if eval_fn is None:
            from heuristics import eval_baseline
            self._eval_fn = eval_baseline
        else:
            self._eval_fn = eval_fn

    def next_action(self, obs: Board):
        action, _ = self._expectimax(obs, self.player, self.depth)
        return action

    def heuristic_utility(self, board: Board) -> float:
        return self._eval_fn(board, self.player)

    def _expectimax(self, board, current_player, depth):
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

        if current_player == self.player:
            best_value  = -math.inf
            best_action = None
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, value = self._expectimax(child, next_player, depth - 1)
                if value > best_value:
                    best_value  = value
                    best_action = action
            return best_action, best_value
        else:
            if not actions:
                return None, 0
            prob  = 1.0 / len(actions)
            total = 0.0
            for action in actions:
                child = board.clone()
                child.play(action, current_player)
                _, value = self._expectimax(child, next_player, depth - 1)
                total += prob * value
            return actions[0], total