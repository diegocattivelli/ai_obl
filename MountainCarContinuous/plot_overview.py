"""Heatmap panorama: todas las configuraciones (filas) x episodios (columnas),
color = recompensa media de test (entre semillas). Compacta en una sola figura
el comportamiento de todos los experimentos: se ve de un vistazo cuáles fallan
(franja amarilla en 0), cuáles aprenden rápido o tarde, y cuáles llegan alto.

No reentrena: lee los *_multiseed.json ya guardados.

    poetry run python plot_overview.py
"""

import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "checkpoints_seeds"

# nombres legibles (orden temático)
LABELS = {
    "gamma_0.9": "γ = 0,9",
    "pos100_epsdecay_slow": "pos100 + decay lento",
    "pos100_epsdecay_fast": "pos100 + decay rápido",
    "alpha_0.2": "α = 0,2",
    "pos_nouniforme": "pos. no unif. centro",
    "acciones_5": "5 acciones",
    "epsilon_decay_slow": "ε decay lento",
    "pos_uniforme_10": "pos. 10 bins",
    "pos_uniforme_100": "pos. 100 bins",
    "vel_uniforme_20": "vel. uniforme",
    "acciones_2": "2 acciones",
    "alpha_0.05": "α = 0,05",
    "epsilon_decay_0.9995": "ε decay rápido",
    "pos_uniforme_30": "pos. 30 bins",
    "pos_nouniforme_izq": "pos. no unif. izq.",
    "pos_nouniforme_izq_epsdecay_fast": "pos. izq. + decay rápido (final)",
    "base_posU20_velNU_3acc": "base",
}

rows, labels = [], []
episodes = None
data = {}
for f in glob.glob(os.path.join(OUTDIR, "*_multiseed.json")):
    d = json.load(open(f))
    name = d["config_name"]
    if name not in LABELS:
        continue
    h = d["history"]
    episodes = [x["trained_episodes"] for x in h]
    data[name] = [x["mean"] for x in h]

# ordenar por recompensa final (peores arriba, mejores abajo)
order = sorted(data.keys(), key=lambda n: data[n][-1])
mat = np.array([data[n] for n in order])
labels = [LABELS[n] for n in order]

fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-100, vmax=100)
ax.set_xticks(range(len(episodes)))
ax.set_xticklabels([f"{e//1000}k" if e else "0" for e in episodes])
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Episodios de entrenamiento")
ax.set_title("Panorama de experimentos — recompensa de test (media entre semillas)")

# anotar el valor en cada celda
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                fontsize=7, color="black")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Recompensa de test")
fig.tight_layout()

out = os.path.join(OUTDIR, "figs", "panorama_experimentos.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
