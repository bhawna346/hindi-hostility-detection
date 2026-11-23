"""Sensitivity vs shortcut: is the effect of a link directional?
For every test post WITHOUT a link we append a random t.co link (or a same-length random string as control) and record
(a) the mean change in the model's hostile score and (b) flips in each direction:
    hostile->non-hostile  (direction predicted by the training-data correlation)
    non-hostile->hostile  (opposite direction; generic sensitivity would produce these too).
Scores: LR = P(hostile); SVM = signed margin (positive = hostile)."""
import sys, re, string
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

D, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
y = lambda d: (~d["Labels Set"].str.contains("non-hostile")).astype(int).values
ytr, yte = y(tr), y(te)
Xtr_txt = tr["Post"].fillna("").astype(str).tolist(); Xte_txt = te["Post"].fillna("").astype(str).tolist()
URL = re.compile(r"https?://\S+|www\.\S+")
AL = list(string.ascii_letters + string.digits)
link = lambda: "https://t.co/" + "".join(rng.choice(AL, 10))
rstr = lambda: "".join(rng.choice(list(string.ascii_lowercase), 23))

def vecs(kind):
    vs = []
    if kind in ("word", "word+char"): vs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\S+"))
    if kind in ("char", "word+char"): vs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000))
    return vs
models = {"LR, word + char": ("word+char", LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=42)),
          "SVM, char": ("char", LinearSVC(C=1.0, class_weight="balanced", random_state=42))}
nolink = np.array([not URL.search(t) for t in Xte_txt])
rows = []
for name, (kind, m) in models.items():
    vs = vecs(kind); Xtr = hstack([v.fit_transform(Xtr_txt) for v in vs]).tocsr(); m.fit(Xtr, ytr)
    T = lambda texts: hstack([v.transform(texts) for v in vs]).tocsr()
    sc = (lambda X: m.predict_proba(X)[:, 1]) if hasattr(m, "predict_proba") else (lambda X: m.decision_function(X))
    idx = np.where(nolink)[0]; base_txt = [Xte_txt[i] for i in idx]
    s0, p0 = sc(T(base_txt)), m.predict(T(base_txt))
    for cond, fn in [("link", link), ("random string", rstr)]:
        s1 = sc(T([t + " " + fn() for t in base_txt])); p1 = m.predict(T([t + " " + fn() for t in base_txt])) if False else (s1 > (0.5 if hasattr(m, "predict_proba") else 0)).astype(int)
        g = yte[idx]
        rows.append(dict(model=name, appended=cond, n_posts=len(idx),
                         mean_score_change_gold_hostile=float((s1 - s0)[g == 1].mean()),
                         mean_score_change_gold_nonhostile=float((s1 - s0)[g == 0].mean()),
                         share_posts_score_decreased=100 * float((s1 < s0).mean()),
                         flips_hostile_to_non_pct=100 * float(((p0 == 1) & (p1 == 0)).sum() / max((p0 == 1).sum(), 1)),
                         flips_non_to_hostile_pct=100 * float(((p0 == 0) & (p1 == 1)).sum() / max((p0 == 0).sum(), 1)),
                         n_pred_hostile=int((p0 == 1).sum()), n_pred_non=int((p0 == 0).sum())))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/url_direction_classical.csv", index=False)
