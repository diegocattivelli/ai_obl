"""Trayectoria en el espacio de fases (el "espiral") de la configuracion final,
dibujada ENCIMA del heatmap de valor aprendido (el fondo de colores).

El fondo muestra cuan bueno considera el agente cada estado (max Q), y la linea
blanca es el camino real del auto en un episodio greedy. Se ve como recorre la
zona de mayor valor balanceandose hasta llegar a la meta.

Necesita gymnasium (corre un episodio), por eso va en tu maquina.

    poetry run python plot_spiral.py
"""

import pickle

import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from q_learning_agent import QLearningAgent
from configs import nonuniform_pos_left, nonuniform_vel, make_actions, make_get_state

PKL = "checkpoints_duel_fast/pos_nouniforme_izq_epsdecay_fast_seed0_ep20000.pkl"
RESET_SEED = 0
FONDO = "valor"   # "valor" (max Q) o "politica" (accion greedy)

# ---- cargar agente entrenado ----
agent, _ = QLearningAgent.load(PKL)
xs_edges = nonuniform_pos_left()      # 20 bordes de posicion
vs_edges = nonuniform_vel()           # 18 bordes de velocidad
get_state = make_get_state(xs_edges, vs_edges)
actions = make_actions(3)
q = agent.q

# ---- fondo: heatmap del espacio de estados ----
visited = np.any(q != 0, axis=2)
if FONDO == "valor":
    bg = q.max(axis=2).astype(float)
    cmap = plt.cm.viridis.copy()
    cbar_label = "Valor del estado (max Q)"
else:  # politica
    bg = np.argmax(q, axis=2).astype(float)
    cmap = plt.cm.coolwarm.copy()
    cbar_label = "Accion greedy (0=izq, 1=nada, 2=der)"
bg[~visited] = np.nan
cmap.set_bad("white")
grid = bg[1:len(xs_edges), 1:len(vs_edges)].T   # bins interiores (vel, pos)

# ---- correr un episodio greedy registrando (posicion, velocidad) ----
env = gym.make("MountainCarContinuous-v0")
obs, _ = env.reset(seed=RESET_SEED)
px, pv = [obs[0]], [obs[1]]
done = False
while not done:
    a = agent.next_action(get_state(obs))
    obs, _, terminated, truncated, _ = env.step(np.array([actions[a]]))
    px.append(obs[0]); pv.append(obs[1])
    done = terminated or truncated
env.close()
px, pv = np.array(px), np.array(pv)
print(f"Pasos: {len(px)-1}   llego a la meta: {terminated}")

# ---- dibujar fondo + trayectoria ----
fig, ax = plt.subplots(figsize=(9, 6))
mesh = ax.pcolormesh(xs_edges, vs_edges, np.ma.masked_invalid(grid),
                     cmap=cmap, edgecolors="white", linewidth=0.2)
# trayectoria: linea blanca con borde oscuro para que resalte sobre el fondo
ax.plot(px, pv, color="black", lw=3.5, zorder=4, solid_capstyle="round")
ax.plot(px, pv, color="white", lw=2.0, zorder=5, solid_capstyle="round")
ax.scatter(px[0], pv[0], c="white", edgecolors="black", s=70, zorder=6, label="inicio")
ax.scatter(px[-1], pv[-1], marker="*", c="red", edgecolors="black", s=220, zorder=6, label="meta")
ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.6)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label(cbar_label)
ax.set_xlabel("Posicion")
ax.set_ylabel("Velocidad")
ax.set_title("Trayectoria sobre el valor aprendido — configuracion final\n"
             "(fondo = valor del estado, linea blanca = episodio greedy)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()

out = "checkpoints_duel_fast/figs/espiral_valor_final.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
