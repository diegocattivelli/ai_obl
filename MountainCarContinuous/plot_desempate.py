"""Figura de desempate (entrenamiento extendido a 20.000 ep) con 3 curvas:
base, posición no uniforme izquierda (ε fijo) y posición izquierda + decay rápido.
Se quitó pos100, que ya se descarta en Exploración por lenta/cara.

No reentrena: combina los historiales multi-semilla ya guardados.

    poetry run python plot_desempate.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting

DUEL = "checkpoints_duel_fast"

# (archivo, etiqueta para la leyenda)
CONFIGS = [
    ("base_posU20_velNU_3acc_extendido_multiseed.json",
     "base (pos U20, vel NU, 3 acc)"),
    ("pos_nouniforme_izq_extendido_multiseed.json",
     "pos. no uniforme izq. (ε fijo)"),
    ("pos_nouniforme_izq_epsdecay_fast_extendido_multiseed.json",
     "pos. no uniforme izq. + ε decay rápido"),
]

results = []
for fname, label in CONFIGS:
    path = os.path.join(DUEL, fname)
    with open(path) as f:
        d = json.load(f)
    results.append({"config_name": label, "history": d["history"]})

fig, ax = plt.subplots(figsize=(11, 6))
plotting.plot_comparison_multiseed(
    results,
    ax=ax,
    title="Desempate — entrenamiento extendido (banda entre semillas)",
    show_seeds=False,
)

figs_dir = os.path.join(DUEL, "figs")
os.makedirs(figs_dir, exist_ok=True)
out = os.path.join(figs_dir, "desempate_3curvas.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
