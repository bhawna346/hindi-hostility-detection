"""Second-dataset check (e.g. HASOC 2019 Hindi, files from the organisers).
Usage: python second_dataset.py TRAIN TEST TEXT_COL LABEL_COL POSITIVE_LABEL(S, comma-sep) OUTDIR [VALID]
Reports: surface cues per class, LR word+char / SVM char / LR word on raw and cue-removed text, URL add/remove test."""
import sys, re, string
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

TR, TE, TXT, LAB, POS, OUT = sys.argv[1:7]
POS = set(POS.split(","))
rd = lambda f: pd.read_csv(f, sep="\t" if f.endswith(".tsv") else ",")
tr, te = rd(TR), rd(TE)
for d in (tr, te): d["y"] = d[LAB].astype(str).str.strip().isin(POS).astype(int); d["t"] = d[TXT].fillna("").astype(str)
if len(sys.argv) > 7:   # official validation file given
    va = rd(sys.argv[7]); va["y"] = va[LAB].astype(str).str.strip().isin(POS).astype(int); va["t"] = va[TXT].fillna("").astype(str)
else:
    tr, va = train_test_split(tr, test_size=0.15, random_state=42, stratify=tr["y"])
URL = re.compile(r"https?://\S+|www\.\S+"); MEN = re.compile(r"@\w+"); HASH = re.compile(r"#\S+"); LAT = re.compile(r"[A-Za-z]+")
allp = pd.concat([tr, va, te])
stats = [dict(label=("positive" if c else "negative"), n=int((allp.y == c).sum()),
              **{f"pct_{k}": 100 * allp[allp.y == c].t.str.contains(p).mean() for k, p in [("url", URL), ("mention", MEN), ("hashtag", HASH), ("latin", LAT)]})
         for c in (0, 1)]
pd.DataFrame(stats).to_csv(f"{OUT}/d2_surface_stats.csv", index=False); print(pd.DataFrame(stats).round(1))
clean = lambda s: HASH.sub(" ", MEN.sub(" ", URL.sub(" ", s)))

def vecs(kind):
    vs = []
    if kind in ("word", "word+char"): vs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\S+"))
    if kind in ("char", "word+char"): vs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000))
    return vs
specs = {"LR, word + char": ("word+char", lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced"), (0.3, 1, 3, 10)),
         "SVM, char": ("char", lambda C: LinearSVC(C=C, class_weight="balanced"), (0.03, 0.1, 0.3, 1)),
         "LR, word": ("word", lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced"), (0.3, 1, 3, 10))}
rng = np.random.default_rng(42); al = string.ascii_letters + string.digits
rows = []
for variant, fn in [("raw", lambda s: s), ("cleaned", clean)]:
    X = {k: [fn(s) for s in d.t] for k, d in [("tr", tr), ("va", va), ("te", te)]}
    for name, (kind, mk, grid) in specs.items():
        vs = vecs(kind); Xtr = hstack([v.fit_transform(X["tr"]) for v in vs]).tocsr()
        T = lambda texts: hstack([v.transform(texts) for v in vs]).tocsr()
        C = max(grid, key=lambda C: f1_score(va.y, mk(C).fit(Xtr, tr.y).predict(T(X["va"])), average="macro"))
        m = mk(C).fit(Xtr, tr.y); p = m.predict(T(X["te"]))
        r = dict(variant=variant, model=name, C=C, macroF1=f1_score(te.y, p, average="macro"))
        if variant == "raw":
            hu = np.array([bool(URL.search(s)) for s in X["te"]])
            A = np.where((te.y.values == 1) & (p == 1) & ~hu)[0]; B = np.where((te.y.values == 0) & (p == 0) & hu)[0]
            if len(A): r["add_url_flip_pct"] = 100 * float((m.predict(T([X["te"][i] + " https://t.co/" + "".join(rng.choice(list(al), 10)) for i in A])) == 0).mean())
            if len(B): r["remove_url_flip_pct"] = 100 * float((m.predict(T([URL.sub(" ", X["te"][i]) for i in B])) == 1).mean())
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/d2_results.csv", index=False)
