"""Heatmap de la POLÍTICA aprendida por la configuración final.

Para cada estado (posición x velocidad) muestra la acción greedy (argmax de Q):
empujar a la izquierda (-1), no empujar (0) o empujar a la derecha (+1).
Los estados nunca visitados (Q = 0) se dejan en gris.

Usa los bordes no uniformes de la config final, así las celdas tienen el ancho
real de cada bin (se ve la mayor resolución concentrada a la izquierda).

No reentrena: lee el .pkl ya guardado.

    poetry run python plot_policy.py
"""

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from configs import nonuniform_pos_left, nonuniform_vel, make_actions

PKL = "checkpoints_duel_fast/pos_nouniforme_izq_epsdecay_fast_seed0_ep20000.pkl"

d = pickle.load(open(PKL, "rb"))
q = d["q"]                      # (n_pos, n_vel, n_actions)
xs = nonuniform_pos_left()      # 20 bordes de posición
vs = nonuniform_vel()           # 18 bordes de velocidad
actions = make_actions(3)       # [-1, 0, 1]

# acción greedy por estado y máscara de "no visitado" (Q todo cero)
greedy = np.argmax(q, axis=2).astype(float)     # 0,1,2  -> -1,0,+1
visited = np.any(q != 0, axis=2)
greedy[~visited] = np.nan

# np.digitize: el bin i (1..len-1) cae entre xs[i-1] y xs[i]; los bins 0 y len
# son colas abiertas (fuera del rango del entorno), se descartan para el ploteo.
pol = greedy[1:len(xs), 1:len(vs)]              # (19, 17) bins interiores
pol = pol.T                                     # (vel, pos) para pcolormesh

cmap = ListedColormap(["#3b6fb0", "#dddddd", "#c0392b"])   # -1 izq, 0 nada, +1 der
cmap.set_bad("white")
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

fig, ax = plt.subplots(figsize=(9, 6))
mesh = ax.pcolormesh(xs, vs, np.ma.masked_invalid(pol), cmap=cmap, norm=norm,
                     edgecolors="white", linewidth=0.3)
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)   # v = 0 (cambio de dirección)

cbar = fig.colorbar(mesh, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(["empuja izq. (-1)", "no empuja (0)", "empuja der. (+1)"])

ax.set_xlabel("Posición")
ax.set_ylabel("Velocidad")
ax.set_title("Política aprendida — config final (izq + decay rápido)\n"
             "celdas blancas = estados nunca visitados")
fig.tight_layout()

out = "checkpoints_duel_fast/figs/politica_final.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
