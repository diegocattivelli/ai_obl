
# Funciones de evaluación para el juego Isolation.

from board import Board

    # Diferencia de movimientos disponibles: mis acciones menos las del oponente.
def h_mobility(board: Board, player: int) -> float:
    opponent = 3 - player
    my_moves  = len(board.get_possible_actions(player))
    opp_moves = len(board.get_possible_actions(opponent))
    return float(my_moves - opp_moves)

    # Distancia Manhattan al centro (negativa). Prefiere estar cerca del centro.
def h_center(board: Board, player: int) -> float:
    pos = board.find_player_position(player)
    cx = (board.board_size[0] - 1) / 2.0
    cy = (board.board_size[1] - 1) / 2.0
    return -(abs(pos[0] - cx) + abs(pos[1] - cy))

    # Distancia Manhattan al oponente (negativa). Prefiere estar cerca del rival.
def h_distance_to_opponent(board: Board, player: int) -> float:
    opponent = 3 - player
    my_pos  = board.find_player_position(player)
    opp_pos = board.find_player_position(opponent)
    return -(abs(my_pos[0] - opp_pos[0]) + abs(my_pos[1] - opp_pos[1]))

    # Diferencia de celdas alcanzables por BFS desde cada jugador.
    # Más sofisticado que mobility: considera el espacio total accesible,
    # no solo los movimientos inmediatos.
def h_open_space(board: Board, player: int) -> float:
    def reachable(start):
        visited = set()
        queue = [start]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            r, c = cur
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < board.board_size[0] and
                            0 <= nc < board.board_size[1] and
                            board.grid[nr, nc] == 0 and
                            (nr, nc) not in visited):
                        queue.append((nr, nc))
        return len(visited)

    opponent = 3 - player
    my_pos  = board.find_player_position(player)
    opp_pos = board.find_player_position(opponent)
    return float(reachable(my_pos) - reachable(opp_pos))


    # Diferencia de celdas destruidas en la vecindad de cada jugador.
    # Positivo cuando el oponente tiene más celdas destruidas alrededor (más acorralado).
def h_blocked_neighbors(board: Board, player: int) -> float:
    opponent = 3 - player
    my_pos  = board.find_player_position(player)
    opp_pos = board.find_player_position(opponent)

    def blocked(pos):
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = pos[0] + dr, pos[1] + dc
                if 0 <= r < board.board_size[0] and 0 <= c < board.board_size[1]:
                    if board.grid[r, c] == 3:
                        count += 1
        return count

    return float(blocked(opp_pos) - blocked(my_pos))





    # Como mobility pero penaliza el doble los movimientos del oponente.
    # Estrategia más agresiva: prioriza reducir las opciones del rival.
def h_aggressive_mobility(board: Board, player: int) -> float:
    opponent = 3 - player
    my_moves  = len(board.get_possible_actions(player))
    opp_moves = len(board.get_possible_actions(opponent))
    return float(my_moves - 2 * opp_moves)



    #Crea una función de evaluación como combinación lineal de heurísticas.
def make_eval(components):
    def eval_fn(board: Board, player: int) -> float:
        return sum(w * fn(board, player) for fn, w in components)
    return eval_fn


# Combinaciones predefinidas listas para usar
# La que ya tenían: las tres heurísticas originales con peso 1
eval_baseline = make_eval([
    (h_mobility,             1.0),
    (h_center,               1.0),
    (h_distance_to_opponent, 1.0),
])

# Solo movilidad
eval_pure_mobility = make_eval([
    (h_mobility, 1.0),
])

# Movilidad con más peso que las demás
eval_mobility_heavy = make_eval([
    (h_mobility, 3.0),
    (h_center,   1.0),
])

# Movilidad agresiva: penaliza el doble las opciones del rival
eval_aggressive = make_eval([
    (h_aggressive_mobility, 2.0),
    (h_blocked_neighbors,   1.0),
])

# Control de espacio mediante BFS
eval_space_control = make_eval([
    (h_open_space, 3.0),
    (h_center,     1.0),
])

# Combinación balanceada con espacio BFS
eval_balanced = make_eval([
    (h_mobility,         2.0),
    (h_open_space,       2.0),
    (h_center,           1.0),
    (h_blocked_neighbors, 1.0),
])

# Solo espacio BFS
eval_pure_space = make_eval([
    (h_open_space, 1.0),
])