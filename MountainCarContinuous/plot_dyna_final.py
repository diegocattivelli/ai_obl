"""Figura de Dyna-Q sobre la config final, todos los niveles presentes, 0-2000,
solo medias (sin bandas)."""
import glob, json, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="checkpoints_dyna_final"
files=sorted(glob.glob(os.path.join(OUT,"dynafinal_plan*_multiseed.json")),
             key=lambda f:int(re.search(r"plan(\d+)",f).group(1)))
fig,ax=plt.subplots(figsize=(11,6))
for f in files:
    d=json.load(open(f)); h=d["history"]; k=d["n_planning_steps"]
    x=[p["trained_episodes"] for p in h]; y=[p["mean"] for p in h]
    ax.plot(x,y,marker="o",ms=4,lw=1.8,label="planning = %d%s"%(k," (= Q-Learning)" if k==0 else ""))
ax.axhline(90,color="gray",ls="--",lw=1,alpha=0.6)
ax.set_xlim(0,2000); ax.set_xticks(range(0,2001,200))
ax.set_xlabel("Episodios de entrenamiento"); ax.set_ylabel("Recompensa de test (media entre semillas)")
ax.set_title("Dyna-Q sobre la config FINAL (izq + epsilon decay), sin bandas")
ax.grid(True,alpha=0.3); ax.legend()
out=os.path.join(OUT,"figs","dyna_final_2000.png"); fig.savefig(out,dpi=120,bbox_inches="tight")
print("Guardado:",out)
