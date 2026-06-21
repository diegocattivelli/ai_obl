"""Heatmap del VALOR aprendido por la configuración final.

Para cada estado (posición x velocidad) muestra max_a Q(s, a), es decir, cuán
bueno considera el agente ese estado. Es más suave que el mapa de política y
muestra el "paisaje de valor" (estados mejor valorados = más cerca de poder
llegar a la meta). Los estados nunca visitados (Q = 0) se dejan en blanco.

Usa los bordes no uniformes de la config final, así las celdas tienen el ancho
real de cada bin. No reentrena: lee el .pkl ya guardado.

    poetry run python plot_value.py
"""

import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs import nonuniform_pos_left, nonuniform_vel

PKL = "checkpoints_duel_fast/pos_nouniforme_izq_epsdecay_fast_seed0_ep20000.pkl"

d = pickle.load(open(PKL, "rb"))
q = d["q"]                      # (n_pos, n_vel, n_actions)
xs = nonuniform_pos_left()      # 20 bordes de posición
vs = nonuniform_vel()           # 18 bordes de velocidad

value = q.max(axis=2).astype(float)       # max_a Q(s,a)
visited = np.any(q != 0, axis=2)
value[~visited] = np.nan

# bins interiores (se descartan las colas abiertas 0 y len de np.digitize)
val = value[1:len(xs), 1:len(vs)].T        # (vel, pos)

cmap = plt.cm.viridis.copy()
cmap.set_bad("white")

fig, ax = plt.subplots(figsize=(9, 6))
mesh = ax.pcolormesh(xs, vs, np.ma.masked_invalid(val), cmap=cmap,
                     edgecolors="white", linewidth=0.3)
ax.axhline(0, color="w", lw=0.8, ls="--", alpha=0.7)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Valor del estado  (max Q)")

ax.set_xlabel("Posición")
ax.set_ylabel("Velocidad")
ax.set_title("Valor aprendido — config final (izq + decay rápido)\n"
             "celdas blancas = estados nunca visitados")
fig.tight_layout()

out = "checkpoints_duel_fast/figs/valor_final.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
