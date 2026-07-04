"""Arma la comparativa de Dyna-Q desde los dyna_plan{K}_multiseed.json que existan.

No reentrena: sirve para plotear el progreso (o el resultado final) sin importar
en que nivel de planning vas. Ordena por cantidad de pasos de planificacion.

    poetry run python plot_dyna.py
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotting

OUTDIR = "checkpoints_dyna"

archivos = glob.glob(os.path.join(OUTDIR, "dyna_plan*_multiseed.json"))
def nivel(f):
    m = re.search(r"dyna_plan(\d+)_multiseed", f)
    return int(m.group(1)) if m else 999
archivos.sort(key=nivel)

if not archivos:
    raise SystemExit("No hay dyna_plan*_multiseed.json todavia en " + OUTDIR)

resultados = []
for f in archivos:
    d = json.load(open(f))
    k = d.get("n_planning_steps", nivel(f))
    label = f"planning = {k}" + (" (= Q-Learning)" if k == 0 else "")
    resultados.append({"config_name": label, "history": d["history"]})
    print(f"cargado planning={k:>3}  (rec final ~{d['history'][-1]['mean']:.0f}, "
          f"exito {d['history'][-1]['success_rate']*100:.0f}%)")

os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 6))
plotting.plot_comparison_multiseed(
    resultados, ax=ax,
    title="Dyna-Q — efecto de los pasos de planificacion (banda entre semillas)")
out = os.path.join(OUTDIR, "figs", "dyna_comparacion.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
