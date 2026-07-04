"""Mide y visualiza el ALIASING de la discretizacion: cuando el mismo par
(bin de posicion, bin de velocidad, accion) lleva a estados siguientes DISTINTOS,
el entorno discretizado deja de ser deterministico. Ese es el supuesto que Dyna-Q
necesita y que acá se rompe.

Corre una politica aleatoria, registra para cada (estado, accion) cuantos estados
siguientes distintos aparecen, y grafica un mapa del grado de no-determinismo.

    poetry run python run_aliasing.py
"""
from collections import defaultdict, Counter
import numpy as np
import gymnasium as gym
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import uniform_pos, nonuniform_vel, make_actions, make_get_state

# discretizacion BASE (cambiar por nonuniform_pos_left() para medir la final)
XS = uniform_pos(20)
VS = nonuniform_vel()
GET_STATE = make_get_state(XS, VS)
ACTIONS = make_actions(3)
EPISODES = 400
MIN_VISITS = 5            # solo contamos aliasing en pares vistos varias veces

trans = defaultdict(Counter)   # (s, a) -> Counter de estados siguientes
env = gym.make("MountainCarContinuous-v0")
np.random.seed(0)
for ep in range(EPISODES):
    obs, _ = env.reset(seed=ep)
    s = GET_STATE(obs)
    done = False
    while not done:
        a = np.random.randint(len(ACTIONS))
        obs, _, term, trunc, _ = env.step(np.array([ACTIONS[a]]))
        done = term or trunc
        s2 = GET_STATE(obs)
        trans[(s, a)][s2] += 1
        s = s2
env.close()

# aliasing por (s,a): cantidad de destinos distintos entre los pares bien visitados
n_pos, n_vel = len(XS) + 1, len(VS) + 1
grid = np.full((n_pos, n_vel), np.nan)
tot = alias = 0
for (s, a), c in trans.items():
    if sum(c.values()) < MIN_VISITS:
        continue
    distintos = len(c)
    tot += 1
    if distintos > 1:
        alias += 1
    px, vx = s
    if 0 <= px < n_pos and 0 <= vx < n_vel:
        grid[px, vx] = np.nanmax([grid[px, vx] if not np.isnan(grid[px, vx]) else 0, distintos])

rate = 100 * alias / tot if tot else 0
print("Pares (estado,accion) con >=%d visitas: %d" % (MIN_VISITS, tot))
print("De esos, con mas de un destino (aliaseados): %d  (%.0f%%)" % (alias, rate))

g = grid[1:len(XS), 1:len(VS)].T
cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("white")
fig, ax = plt.subplots(figsize=(9, 6))
m = ax.pcolormesh(XS, VS, np.ma.masked_invalid(g), cmap=cmap, vmin=1,
                  edgecolors="white", linewidth=0.2)
fig.colorbar(m, ax=ax, label="Destinos distintos por (bin, accion)  (1 = deterministico)")
ax.set_xlabel("Posicion"); ax.set_ylabel("Velocidad")
ax.set_title("No-determinismo del entorno discretizado (base)\n"
             "%.0f%% de los pares (estado, accion) son aliaseados" % rate)
fig.tight_layout()
fig.savefig("checkpoints_dyna_fino/figs/aliasing_base.png", dpi=120, bbox_inches="tight")
print("Guardado: checkpoints_dyna_fino/figs/aliasing_base.png")
