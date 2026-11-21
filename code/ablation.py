"""Surface-cue ablation: does char n-gram performance depend on URLs, mentions, hashtags or Latin script?
Also: per-class surface statistics and length-binned error analysis for the best classical model."""
import sys, re, json
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

D, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
FINE = ["fake", "hate", "offensive", "defamation"]
tr, va, te = (pd.read_csv(f"{D}/{s}.csv") for s in ["train", "valid", "test"])
def labels(df):
    sets = df["Labels Set"].str.split(",").apply(lambda l: {x.strip() for x in l})
    return (~sets.apply(lambda s: "non-hostile" in s)).astype(int).values, np.stack([sets.apply(lambda s, l=l: int(l in s)).values for l in FINE], 1)
(ytr, ytrf), (yva, yvaf), (yte, ytef) = labels(tr), labels(va), labels(te)

URL = re.compile(r"https?://\S+|www\.\S+")
MEN = re.compile(r"@\w+"); HASH = re.compile(r"#\S+")
LATIN = re.compile(r"[A-Za-z]+"); DIG = re.compile(r"[0-9०-९]+")
NONDEV = re.compile(r"[^ऀ-ॿ\s]")
variants = {
    "raw": lambda s: s,
    "no URLs": lambda s: URL.sub(" ", s),
    "no URLs/mentions/hashtags": lambda s: HASH.sub(" ", MEN.sub(" ", URL.sub(" ", s))),
    "Devanagari only": lambda s: re.sub(r"\s+", " ", NONDEV.sub(" ", HASH.sub(" ", MEN.sub(" ", URL.sub(" ", s))))),
}
def run(fn):
    X = {k: [fn(str(s)) for s in d["Post"].fillna("")] for k, d in [("tr", tr), ("va", va), ("te", te)]}
    vs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)]
    if MODEL == "LR": vs.append(TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\S+"))
    from scipy.sparse import hstack
    Xtr = hstack([v.fit_transform(X["tr"]) for v in vs]).tocsr(); Xva = hstack([v.transform(X["va"]) for v in vs]).tocsr(); Xte = hstack([v.transform(X["te"]) for v in vs]).tocsr()
    from sklearn.linear_model import LogisticRegression
    mk = (lambda C: LinearSVC(C=C, class_weight="balanced", random_state=42)) if MODEL == "SVM" else (lambda C: LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=42))
    grid = (0.03, 0.1, 0.3, 1) if MODEL == "SVM" else (0.3, 1, 3, 10)
    best = max(grid, key=lambda C: f1_score(yva, mk(C).fit(Xtr, ytr).predict(Xva), average="macro"))
    pc = mk(best).fit(Xtr, ytr).predict(Xte)
    pf = np.stack([mk(best).fit(Xtr, ytrf[:, j]).predict(Xte) for j in range(4)], 1)
    per = [f1_score(ytef[:, j], pf[:, j]) for j in range(4)]; sup = ytef.sum(0)
    return dict(C=best, coarse_macroF1=f1_score(yte, pc, average="macro"), coarse_weightedF1=f1_score(yte, pc, average="weighted"),
                **{f"F1_{l}": p for l, p in zip(FINE, per)}, fine_weightedF1=float(np.dot(per, sup) / sup.sum())), pc

rows = []; preds = {}
for name, fn in variants.items():
    r, pc = run(fn); r = {"variant": name, **r}; rows.append(r); preds[name] = pc; print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/ablation_{MODEL}.csv", index=False)

# surface statistics by class (train+valid+test combined)
al = pd.concat([tr, va, te]); post = al["Post"].fillna("").astype(str)
sets = al["Labels Set"].str.split(",").apply(lambda l: {x.strip() for x in l})
stat = []
for lab in ["non-hostile"] + FINE:
    m = sets.apply(lambda s: lab in s).values; p = post[m]
    stat.append(dict(label=lab, n=int(m.sum()), pct_with_url=100 * p.str.contains(URL).mean(), pct_with_mention=100 * p.str.contains(MEN).mean(),
                     pct_with_hashtag=100 * p.str.contains(HASH).mean(), pct_with_latin=100 * p.str.contains(LATIN).mean(),
                     median_words=float(p.str.split().str.len().median())))
pd.DataFrame(stat).to_csv(f"{OUT}/surface_stats.csv", index=False); print(pd.DataFrame(stat).round(1).to_string())

# error analysis by length for raw best model
L = te["Post"].fillna("").astype(str).str.split().str.len().values
bins = [(0, 15), (16, 30), (31, 60), (61, 10**6)]; ea = []
for lo, hi in bins:
    m = (L >= lo) & (L <= hi)
    ea.append(dict(words=f"{lo}-{hi if hi < 10**6 else '+'}", n=int(m.sum()), macroF1=f1_score(yte[m], preds["raw"][m], average="macro"),
                   error_rate=float((yte[m] != preds["raw"][m]).mean())))
pd.DataFrame(ea).to_csv(f"{OUT}/length_errors_{MODEL}.csv", index=False); print(pd.DataFrame(ea).round(3).to_string())
