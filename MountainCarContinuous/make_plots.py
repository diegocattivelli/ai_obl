"""Genera las gráficas comparativas POR GRUPO de decisión a partir de los
historiales ya guardados (checkpoints/<config>_history.json).

No depende de gymnasium, así que puede correrse sin entrenar, para regenerar
figuras cuando ya existen los historiales:

    python make_plots.py                 # todos los grupos con historial disponible
    python make_plots.py --band minmax   # cambiar el tipo de banda
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting
from configs import GROUPS, configs_in_group


def load_history(outdir, name):
    path = os.path.join(outdir, f"{name}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["history"]


def make_group_plots(outdir="checkpoints", band="std"):
    figs_dir = os.path.join(outdir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    generadas = []

    for group_key, title in GROUPS.items():
        items = []
        for cfg in configs_in_group(group_key):
            hist = load_history(outdir, cfg["name"])
            if hist:
                label = cfg["name"]
                if "base" in cfg["name"]:
                    label += " (ganadora)"
                items.append((label, hist))

        if len(items) < 2:
            print(f"[saltado] {group_key}: faltan historiales ({len(items)} disponibles)")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        plotting.plot_comparison(items, band=band, ax=ax,
                                 title=f"Decisión: {title}")
        out = os.path.join(figs_dir, f"grupo_{group_key}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        generadas.append(out)
        print("Guardado:", out)

    return generadas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="checkpoints")
    p.add_argument("--band", default="std", choices=["std", "minmax", "iqr"])
    args = p.parse_args()
    make_group_plots(args.outdir, args.band)
