"""Fairness fixes for the classical models.
(1) Fine-grained: per-label decision thresholds tuned on validation (same protocol as the transformers).
(2) Inference time: end-to-end, TF-IDF transform of the test posts + classifier.
Hyperparameters reused from the first run (classical_results.csv). Raw text unless --variant given."""
import sys, time, re, json
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

D, OUT, VARIANT, ONLY = sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else None)
SEED = 42; FINE = ["fake", "hate", "offensive", "defamation"]
tr, va, te = (pd.read_csv(f"{D}/{s}.csv") for s in ["train", "valid", "test"])
def labels(df):
    sets = df["Labels Set"].str.split(",").apply(lambda l: {x.strip() for x in l})
    return (~sets.apply(lambda s: "non-hostile" in s)).astype(int).values, np.stack([sets.apply(lambda s, l=l: int(l in s)).values for l in FINE], 1)
(ytr, ytrf), (yva, yvaf), (yte, ytef) = labels(tr), labels(va), labels(te)
URL = re.compile(r"https?://\S+|www\.\S+"); MEN = re.compile(r"@\w+"); HASH = re.compile(r"#\S+")
NONDEV = re.compile(r"[^ऀ-ॿ\s]")
fn = {"raw": lambda s: s, "no URLs": lambda s: URL.sub(" ", s),
      "no URLs/mentions/hashtags": lambda s: HASH.sub(" ", MEN.sub(" ", URL.sub(" ", s))),
      "Devanagari only": lambda s: re.sub(r"\s+", " ", NONDEV.sub(" ", HASH.sub(" ", MEN.sub(" ", URL.sub(" ", s)))))}[VARIANT]
X = {k: [fn(str(s)) for s in d["Post"].fillna("")] for k, d in [("tr", tr), ("va", va), ("te", te)]}
prev = pd.read_csv(f"{OUT}/classical_results.csv")
if VARIANT != "raw":  # ablation runs used tuned C per variant
    abl = {"LR": pd.read_csv(f"{OUT}/ablation_LR.csv"), "SVM": pd.read_csv(f"{OUT}/ablation_SVM.csv")}

def vecs(kind):
    vs = []
    if kind in ("word", "word+char"):
        vs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\S+"))
    if kind in ("char", "word+char"):
        vs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000))
    return vs

def make(mname, p):
    if mname == "LR": return LogisticRegression(C=float(p), max_iter=3000, class_weight="balanced", random_state=SEED)
    if mname == "LinearSVM": return LinearSVC(C=float(p), class_weight="balanced", random_state=SEED)
    if mname == "ComplementNB": return ComplementNB(alpha=float(p))
    if mname == "RandomForest": return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=SEED)
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, tree_method="hist", n_jobs=2, random_state=SEED, eval_metric="logloss")

def score_fn(m, Xm):
    return m.decision_function(Xm) if isinstance(m, LinearSVC) else m.predict_proba(Xm)[:, 1]

def grid(m):
    return np.linspace(-1.5, 1.5, 61) if isinstance(m, LinearSVC) else np.arange(0.05, 0.96, 0.05)

configs = [(r.features, r.model, r.params) for r in prev.itertuples()]
if VARIANT != "raw":
    configs = [("word+char", "LR", abl["LR"].set_index("variant").loc[VARIANT, "C"]),
               ("char", "LinearSVM", abl["SVM"].set_index("variant").loc[VARIANT, "C"])]
if ONLY: configs = [c for c in configs if c[1] in ONLY.split(",")]
rows = []
for fk in dict.fromkeys(c[0] for c in configs):
    vs = vecs(fk)
    Xtr = hstack([v.fit_transform(X["tr"]) for v in vs]).tocsr(); Xva = hstack([v.transform(X["va"]) for v in vs]).tocsr()
    t0 = time.perf_counter(); Xte = hstack([v.transform(X["te"]) for v in vs]).tocsr(); t_vec = time.perf_counter() - t0
    for f, mname, p in configs:
        if f != fk: continue
        mc = make(mname, p)
        if isinstance(mc, XGBClassifier): mc.set_params(scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum())
        mc.fit(Xtr, ytr)
        t0 = time.perf_counter(); mc.predict(Xte); t_clf = time.perf_counter() - t0
        per_def, per_tuned, ths = [], [], []
        for j in range(4):
            m = make(mname, p)
            if isinstance(m, XGBClassifier): m.set_params(scale_pos_weight=(len(ytrf) - ytrf[:, j].sum()) / ytrf[:, j].sum())
            m.fit(Xtr, ytrf[:, j])
            sva, ste = score_fn(m, Xva), score_fn(m, Xte)
            th = max(grid(m), key=lambda t: f1_score(yvaf[:, j], (sva > t).astype(int)))
            ths.append(round(float(th), 2))
            per_def.append(f1_score(ytef[:, j], m.predict(Xte)))
            per_tuned.append(f1_score(ytef[:, j], (ste > th).astype(int)))
        sup = ytef.sum(0)
        r = dict(variant=VARIANT, features=fk, model=mname, params=p, thresholds=str(ths),
                 fine_weightedF1_default=float(np.dot(per_def, sup) / sup.sum()), fine_weightedF1_tuned=float(np.dot(per_tuned, sup) / sup.sum()),
                 **{f"F1_{l}_tuned": v for l, v in zip(FINE, per_tuned)},
                 vec_test_ms_per_1k=t_vec / len(te) * 1e6, clf_ms_per_1k=t_clf / len(te) * 1e6,
                 infer_total_ms_per_1k=(t_vec + t_clf) / len(te) * 1e6)
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        tag = VARIANT.replace("/", "_").replace(" ", "_")
        pd.DataFrame(rows).to_csv(f"{OUT}/classical_v2_{tag}{'_' + ONLY.replace(',', '_') if ONLY else ''}.csv", index=False)
