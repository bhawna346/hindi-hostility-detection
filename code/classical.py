"""Classical ML benchmark on CONSTRAINT-2021 Hindi hostility detection.
Coarse task: hostile vs non-hostile. Fine task: 4 binary labels (fake, hate, offensive, defamation).
Hyperparameters tuned on the official validation split; final numbers on the official test split.
"""
import sys, time, pickle, json, re
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

D = sys.argv[1]; OUT = sys.argv[2]
SEED = 42
tr, va, te = (pd.read_csv(f"{D}/{s}.csv") for s in ["train", "valid", "test"])
FINE = ["fake", "hate", "offensive", "defamation"]

def labels(df):
    sets = df["Labels Set"].str.split(",").apply(lambda l: {x.strip() for x in l})
    y_coarse = (~sets.apply(lambda s: "non-hostile" in s)).astype(int).values
    y_fine = np.stack([sets.apply(lambda s, l=l: int(l in s)).values for l in FINE], 1)
    return y_coarse, y_fine

ytr_c, ytr_f = labels(tr); yva_c, yva_f = labels(va); yte_c, yte_f = labels(te)
Xtxt = {k: d["Post"].fillna("").astype(str).values for k, d in [("tr", tr), ("va", va), ("te", te)]}

def feats(kind):
    vs = []
    if kind in ("word", "word+char"):
        vs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\S+"))
    if kind in ("char", "word+char"):
        vs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000))
    t0 = time.perf_counter()
    mats = {k: [] for k in Xtxt}
    for v in vs:
        mats["tr"].append(v.fit_transform(Xtxt["tr"]))
        for k in ("va", "te"):
            mats[k].append(v.transform(Xtxt[k]))
    fit_t = time.perf_counter() - t0
    return {k: hstack(m).tocsr() for k, m in mats.items()}, vs, fit_t

def models():
    return {
        "LR": [lambda C=C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED) for C in (0.3, 1, 3, 10)],
        "LinearSVM": [lambda C=C: LinearSVC(C=C, class_weight="balanced", random_state=SEED) for C in (0.03, 0.1, 0.3, 1)],
        "ComplementNB": [lambda a=a: ComplementNB(alpha=a) for a in (0.1, 0.3, 1.0)],
        "RandomForest": [lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)],
        "XGBoost": [lambda: XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="logloss")],
    }

def fit_binary(make, X, y):
    m = make()
    if isinstance(m, XGBClassifier):
        m.set_params(scale_pos_weight=(len(y) - y.sum()) / max(y.sum(), 1))
    m.fit(X, y); return m

def boot_ci(y, p, fn, n=1000):
    rng = np.random.default_rng(SEED); idx = np.arange(len(y)); s = []
    for _ in range(n):
        b = rng.choice(idx, len(idx)); s.append(fn(y[b], p[b]))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

rows, preds = [], {}
for fk in ("word", "char", "word+char"):
    X, vecs, vec_t = feats(fk)
    print(fk, X["tr"].shape, flush=True)
    for mname, grid in models().items():
        # tune on validation (coarse macro-F1)
        best = None
        for make in grid:
            m = fit_binary(make, X["tr"], ytr_c)
            s = f1_score(yva_c, m.predict(X["va"]), average="macro")
            if best is None or s > best[0]: best = (s, make)
        make = best[1]
        # coarse
        t0 = time.perf_counter(); mc = fit_binary(make, X["tr"], ytr_c); t_train = time.perf_counter() - t0
        t0 = time.perf_counter(); pc = mc.predict(X["te"]); t_inf = time.perf_counter() - t0
        # fine: one binary classifier per label
        t0 = time.perf_counter(); fine_models = [fit_binary(make, X["tr"], ytr_f[:, j]) for j in range(4)]; t_train_f = time.perf_counter() - t0
        pf = np.stack([fm.predict(X["te"]) for fm in fine_models], 1)
        per = [f1_score(yte_f[:, j], pf[:, j]) for j in range(4)]
        sup = yte_f.sum(0); wfine = float(np.dot(per, sup) / sup.sum())
        size_mb = (len(pickle.dumps(mc)) + sum(len(pickle.dumps(v)) for v in vecs)) / 1e6
        lo, hi = boot_ci(yte_c, pc, lambda a, b: f1_score(a, b, average="macro"))
        r = dict(features=fk, model=mname, params=str(make().get_params().get("C", make().get_params().get("alpha", ""))),
                 val_macroF1=round(best[0], 4),
                 coarse_acc=accuracy_score(yte_c, pc), coarse_P_hostile=precision_score(yte_c, pc), coarse_R_hostile=recall_score(yte_c, pc),
                 coarse_F1_hostile=f1_score(yte_c, pc), coarse_macroF1=f1_score(yte_c, pc, average="macro"),
                 coarse_weightedF1=f1_score(yte_c, pc, average="weighted"), coarse_macroF1_CI=f"{lo:.3f}-{hi:.3f}",
                 F1_fake=per[0], F1_hate=per[1], F1_offensive=per[2], F1_defamation=per[3], fine_weightedF1=wfine,
                 vec_time_s=vec_t, train_time_coarse_s=t_train, train_time_fine_s=t_train_f,
                 infer_ms_per_1k=t_inf / len(pc) * 1000 * 1000, model_size_MB=size_mb)
        rows.append(r); preds[f"{fk}|{mname}"] = (pc.tolist(), pf.tolist())
        pd.DataFrame(rows).to_csv(f"{OUT}/classical_results.csv", index=False); json.dump(preds, open(f"{OUT}/classical_preds.json", "w"))
        print(mname, round(r["coarse_macroF1"], 4), round(wfine, 4), f"train {t_train:.1f}s", flush=True)

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/classical_results.csv", index=False)
json.dump(preds, open(f"{OUT}/classical_preds.json", "w"))
json.dump({"train": len(tr), "valid": len(va), "test": len(te),
           "test_hostile": int(yte_c.sum()), "test_fine_support": dict(zip(FINE, yte_f.sum(0).tolist())),
           "train_hostile": int(ytr_c.sum()), "train_fine_support": dict(zip(FINE, ytr_f.sum(0).tolist()))},
          open(f"{OUT}/data_stats.json", "w"), indent=1)
print(df.sort_values("coarse_macroF1", ascending=False)[["features", "model", "coarse_macroF1", "fine_weightedF1", "train_time_coarse_s", "infer_ms_per_1k"]].to_string())
