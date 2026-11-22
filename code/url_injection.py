"""Counterfactual URL test for the classical models (trained on raw text, as in Table 2).
(A) Add: append a fresh random t.co link to correctly-detected hostile test posts that have no URL -> % now called non-hostile.
(B) Remove: delete URLs from correctly-detected non-hostile test posts that have a URL -> % now called hostile.
Control for (A): append a random 23-character Latin string of the same length that is not a link."""
import sys, re, string, json
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

D, OUT = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(42)
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
y = lambda d: (~d["Labels Set"].str.contains("non-hostile")).astype(int).values
ytr, yte = y(tr), y(te)
Xtr_txt = tr["Post"].fillna("").astype(str).tolist(); Xte_txt = te["Post"].fillna("").astype(str).tolist()
URL = re.compile(r"https?://\S+|www\.\S+")
alnum = string.ascii_letters + string.digits
def fake_url(): return "https://t.co/" + "".join(rng.choice(list(alnum), 10))
def fake_str(): return "".join(rng.choice(list(string.ascii_lowercase), 23))

def build(kind):
    vs = []
    if kind in ("word", "word+char"): vs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\S+"))
    if kind in ("char", "word+char"): vs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000))
    return vs

models = {"LR, word + char": ("word+char", LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=42)),
          "SVM, char": ("char", LinearSVC(C=1.0, class_weight="balanced", random_state=42)),
          "LR, word": ("word", LogisticRegression(C=10, max_iter=3000, class_weight="balanced", random_state=42))}
rows = []
for name, (kind, m) in models.items():
    vs = build(kind); Xtr = hstack([v.fit_transform(Xtr_txt) for v in vs]).tocsr(); m.fit(Xtr, ytr)
    pred = lambda texts: m.predict(hstack([v.transform(texts) for v in vs]).tocsr())
    p0 = pred(Xte_txt)
    has_url = np.array([bool(URL.search(t)) for t in Xte_txt])
    A = np.where((yte == 1) & (p0 == 1) & ~has_url)[0]
    B = np.where((yte == 0) & (p0 == 0) & has_url)[0]
    pa = pred([Xte_txt[i] + " " + fake_url() for i in A])
    pc = pred([Xte_txt[i] + " " + fake_str() for i in A])
    pb = pred([URL.sub(" ", Xte_txt[i]) for i in B])
    r = dict(model=name, n_add=len(A), add_url_flip_pct=100 * float((pa == 0).mean()), add_control_flip_pct=100 * float((pc == 0).mean()),
             n_remove=len(B), remove_url_flip_pct=100 * float((pb == 1).mean()))
    rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/url_injection_classical.csv", index=False)
