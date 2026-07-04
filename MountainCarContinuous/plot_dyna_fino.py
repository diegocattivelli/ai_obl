"""Figura fina de Dyna-Q en los primeros 2.000 episodios, con todos los niveles
que existan en checkpoints_dyna_fino. No reentrena.

    poetry run python plot_dyna_fino.py
"""
import glob, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotting

OUTDIR = "checkpoints_dyna_fino"
files = sorted(glob.glob(os.path.join(OUTDIR, "dynafino_plan*_multiseed.json")),
               key=lambda f: int(re.search(r"plan(\d+)", f).group(1)))
res = []
for f in files:
    d = json.load(open(f)); k = d["n_planning_steps"]
    lab = "planning = %d%s" % (k, " (= Q-Learning)" if k == 0 else "")
    res.append({"config_name": lab, "history": d["history"]})
    print("cargado planning=%d" % k)

fig, ax = plt.subplots(figsize=(12, 6))
plotting.plot_comparison_multiseed(res, ax=ax,
    title="Dyna-Q - eficiencia de muestras (primeros 2.000 episodios)")
ax.axhline(90, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_xlim(0, 2000); ax.set_xticks(range(0, 2001, 200))
os.makedirs(os.path.join(OUTDIR, "figs"), exist_ok=True)
out = os.path.join(OUTDIR, "figs", "dyna_fino_2000.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print("Guardado:", out)
