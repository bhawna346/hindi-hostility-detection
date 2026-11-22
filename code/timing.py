"""End-to-end inference timing for the classical models (TF-IDF transform of the test posts + classifier),
median of 5 repeats, run alone on the machine. Coarse-task models with the tuned hyperparameters."""
import sys, time
import numpy as np, pandas as pd
from scipy.sparse import hstack
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "raw"]
exec(open("/home/claude/exp/classical_v2.py").read().split("configs = [")[0])  # reuse data loading, vecs(), make()
D, OUT = sys.argv[1], sys.argv[2]
rows = []
for fk in ("word", "char", "word+char"):
    vs = vecs(fk)
    Xtr = hstack([v.fit_transform(X["tr"]) for v in vs]).tocsr()
    for r in prev[prev.features == fk].itertuples():
        m = make(r.model, r.params)
        if isinstance(m, XGBClassifier): m.set_params(scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum())
        m.fit(Xtr, ytr)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            Xte = hstack([v.transform(X["te"]) for v in vs]).tocsr(); m.predict(Xte)
            ts.append(time.perf_counter() - t0)
        ms = float(np.median(ts)) / len(te) * 1e6
        rows.append(dict(features=fk, model=r.model, infer_total_ms_per_1k=ms)); print(fk, r.model, round(ms, 1), flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/classical_timing_end_to_end.csv", index=False)
