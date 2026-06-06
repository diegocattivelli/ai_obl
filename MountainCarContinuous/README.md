# Mountain Car Continuous — Q-Learning (Proyecto LOST)

Agente Q-Learning tabular para `MountainCarContinuous-v0` (Gymnasium), con
discretización de estados y acciones, exploración de hiperparámetros y
metodología de **entrenamiento/evaluación iterativa** (curvas de aprendizaje).

> Estado actual: **v2 cerrada**. Próximo: v2.1 (multi-semilla) → init de tabla /
> más episodios → Dyna-Q. Ver "Roadmap" al final.

---

## Metodología (lo importante)

En vez de entrenar cada configuración entera y testear una sola vez al final, se
**intercala** entrenamiento y evaluación:

```
[test base] → train 1000 → test 100 → train 1000 → test 100 → … (×10 = 10 000 episodios)
```

En cada checkpoint se registran, sobre los episodios de test (greedy, sin
explorar): media, desvío, min, max, percentiles 25/50/75, **tasa de éxito** y
pasos promedio. Con eso se grafican **curvas de aprendizaje con bandas de rango**,
que revelan inestabilidad / mesetas / varianza que el número final esconde.

La tabla Q se guarda en `.pkl` en cada checkpoint (modelo computado + reanudable).

## Configuraciones (un factor por vez)

Se varía un solo factor por vez alrededor de la config ganadora ("base"), de modo
que la comparativa de cada grupo justifica una decisión por sí sola.

| Grupo | Configs | Conclusión |
|---|---|---|
| Bins de posición  | 10, **20**, 30, 100 | 20–30 es el punto dulce; 10 inestable, 100 aprende pero lentísimo |
| Discretización velocidad | uniforme vs **no uniforme** | no uniforme (bins cerca de v≈0) aprende antes y más estable |
| Acciones          | 2, **3**, 5 | menos acciones = más rápido/estable; 5 muy inestable; 3 es el balance |
| Gamma             | 0.9 vs **0.999** | con γ=0.9 **no aprende** (recompensa diferida); 0.999 obligatorio |
| Alpha             | 0.05, **0.1**, 0.2 | 0.1 equilibra: 0.05 lento, 0.2 oscila |
| Epsilon           | **0.9 fijo** vs decay | recompensa final equivalente; el decay converge más estable |

**Config ganadora (base):** posición uniforme 20, velocidad no uniforme (~20),
3 acciones `[-1,0,1]`, α=0.1, γ=0.999, ε=0.9 fijo.

**Config final v2** (`final_v2_pos30_velNU_3acc_epsdecay_slow`): combina los dos
hallazgos de mayor estabilidad → posición **30** + velocidad no uniforme + 3
acciones + α=0.1 + γ=0.999 + **ε con decay lento** (1.0 → 0.1, decay 0.9998).
Aprende a los ~2000 episodios y se mantiene estable al 100% de éxito.

> **Hallazgo (interacción posición × exploración).** El primer intento naive
> (`final_v2_pos30_velNU_3acc_epsdecay`, decay 0.9995, ε_min 0.05) **NO aprende**:
> la tabla más grande de pos30 necesita más exploración, pero ese decay —calibrado
> para pos20— la recorta antes de descubrir la meta, y el agente queda clavado en
> recompensa 0. Se corrige estirando el decay (0.9998, ε_min 0.1). Muestra que los
> óptimos de cada eje por separado no necesariamente se combinan. Ambas configs se
> conservan en `configs.py` (`build_final_config_naive` y `build_final_config`).

## v2.1 — Robustez (multi-semilla) y desempate (entrenamiento extendido)

Se reentrenó cada config con **3 semillas** (0,1,2); las curvas pasan a mostrar la
**banda ENTRE semillas** (qué tanto cambia el resultado según el azar). Hallazgos:

- Las conclusiones por eje **se sostienen** entre semillas (γ=0.9 no aprende en las
  3; vel no uniforme > uniforme; acciones 5 y pos 10 frágiles; α=0.2 se degrada; etc.).
- El pozo de la base a ~7000 episodios era **en buena parte cosa de seed 0**:
  promediando 3 semillas la base termina robusta. La multi-semilla **matiza** la v2.
- `base`, `pos_30`, `epsilon_decay` y `final_slow` **convergen a un desempeño
  equivalente** (~93, 100% éxito); las diferencias estaban dentro del ruido de semilla.

Para desempatar se hizo un **entrenamiento extendido** (`--duel`, hasta 20 000
episodios) de `base` vs `final_slow` vs `pos_100`, con su gráfica aparte
(`checkpoints_duel/figs/duel_extendido.png`). No hay un único ganador: es un
**frente de Pareto** entre tres ejes.

| Config | Velocidad de aprendizaje | Estabilidad a largo plazo | Costo (tamaño tabla) |
|---|---|---|---|
| base (pos20, ε fijo) | rápida (~2000) | la peor (bandazo ancho a ~19k) | la más barata (~1200 estados) |
| **final_slow (pos30, decay)** | **rápida (~2000)** | **buena** | media (~1800) |
| pos_100 (ε fijo) | lenta (~7000) | la mejor (banda finísima) | la más cara (~5800) |

**Modelo final declarado: `final_v2_pos30_velNU_3acc_epsdecay_slow`** — no por ser
"la más estable" (ese título es de pos_100, que paga 3,5× más episodios y 5× más
tabla), sino por ser el **mejor compromiso**: aprende rápido, es estable y tiene
costo razonable. La estabilidad asintótica superior de pos_100 y el bandazo tardío
de la base (ε fijo nunca deja de perturbar; tabla gruesa amplifica los bandazos) se
documentan como parte del análisis.

> Caveat: con 3 semillas los pozos tardíos son en parte eventos de una sola semilla
> (la banda se abre, no caen las 3). El desempate es sugerente, no definitivo.

## Archivos

| Archivo | Rol |
|---|---|
| `q_learning_agent.py` | Agente: entrenamiento **incremental** (Q y ε persisten), `evaluate` con tasa de éxito, `save`/`load` (.pkl). |
| `train_eval_loop.py` | Harness del loop train/test: `run_train_eval`, `resume_train_eval`. Genera `.pkl`, `_metrics.csv`, `_history.json`. |
| `configs.py` | Discretizaciones + 11 configs agrupadas + config final (`build_final_config`). |
| `plotting.py` | Curva de aprendizaje (banda std/minmax/iqr), tasa de éxito, recompensa de train, comparación. |
| `make_plots.py` | Comparativa **por grupo** desde los `_history.json` (sin gymnasium; regenera figuras sin reentrenar). |
| `run_experiments.py` | Runner de terminal. |
| `continuous_mountain_car.ipynb` | Notebook orquestador. |
| `README_metodologia.md` | Detalle extendido de la metodología. |

## Cómo correr

```bash
poetry run python run_experiments.py                 # 11 configs de grupos + 2 finales
poetry run python run_experiments.py --group acciones
poetry run python run_experiments.py --only final_v2_pos30_velNU_3acc_epsdecay_slow  # final
poetry run python run_experiments.py --only final_v2_pos30_velNU_3acc_epsdecay        # naive (no aprende)
poetry run python run_experiments.py --cycles 5 --train 200 --eval 20   # prueba rápida
poetry run python make_plots.py                      # regenerar comparativas por grupo
poetry run python run_experiments.py --resume checkpoints/<config>_ep10000.pkl --cycles 5

# v2.1 — multi-semilla (banda entre semillas):
poetry run python run_experiments.py --seeds 0,1,2 --outdir checkpoints_v21      # todas, 3 semillas
poetry run python run_experiments.py --group gamma --seeds 0,1,2 --outdir checkpoints_v21

# v2.1 — duelo / entrenamiento extendido (desempate por estabilidad asintótica):
poetry run python run_experiments.py \
  --duel base_posU20_velNU_3acc,final_v2_pos30_velNU_3acc_epsdecay_slow,pos_uniforme_100 \
  --seeds 0,1,2 --resume-dir checkpoints_v21 --add-cycles 10 --outdir checkpoints_duel
```

Salidas en `checkpoints/`: `<config>_ep<N>.pkl`, `<config>_metrics.csv`,
`<config>_history.json`, `figs/<config>_reporte.png`, `figs/grupo_<grupo>.png`.

> Nota: cada config tarda varios minutos (`pos_100` es la más lenta). Se puede
> correr grupo por grupo en distintas sesiones; `make_plots.py` arma las
> comparativas con los historiales que haya.

## Decisiones de diseño

- **Evaluación con semilla fija** (`eval_seed=0`): mismos estados iniciales en cada
  checkpoint → curvas comparables.
- **Éxito** = el episodio termina en la meta (`terminated`), no por límite de pasos.
- La recompensa de **train** se ve baja porque se entrena con ε-greedy (mucha
  exploración); el rendimiento real se mide aparte en modo greedy (test).
- Entrenamiento reproducible con `seed` fijo. La v2 usó seed=0; la **v2.1**
  promedia 3 semillas (0,1,2) y muestra la banda entre semillas.

---

## Roadmap

- [x] **v2** — loop train/test, 11 configs agrupadas, comparativas por grupo, config final.
- [x] **v2.1 — Multi-semilla + desempate.** 3 semillas con banda entre semillas;
  conclusiones por eje robustas; final declarado `final_slow` por mejor compromiso
  (Pareto vs base y pos_100). "Más episodios" cubierto por el duelo (hasta 20k).
- [ ] **Init de tabla Q (optimista).** Único pendiente menor antes de Dyna-Q; opcional.
- [ ] **Dyna-Q** (obligatorio, punto 4). Reusa este harness sobre la versión final
  del Q-learning normal; variar nº de pasos de planning (0, 5, 50) como grupo.

## Punto de retorno (para retomar)

Si volvés a este commit: v2 y v2.1 están completas y reproducibles (multi-semilla
+ duelo extendido), con `final_slow` declarado como modelo final. Lo que queda:
inicialización optimista de la tabla Q (opcional) y **Dyna-Q** (obligatorio,
punto 4), que reusa este mismo harness sobre la versión final. El ejercicio 2
(MATE / Isolation) ya está hecho aparte.
