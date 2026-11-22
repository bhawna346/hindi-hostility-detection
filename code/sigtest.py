"""Significance tests: transformer (seeds 13, 77) vs best classical model on the same test posts.
Raw text: LR word+char (Table 2). Cleaned text: char SVM retrained on cleaned text (ablation setting).
Exact McNemar test on coarse predictions + paired bootstrap 95% CI of the macro-F1 difference."""
import sys, re, json, glob
import numpy as np, pandas as pd
from scipy.sparse import hstack
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

D, PRED, OUT = sys.argv[1:4]
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
y = lambda d: (~d["Labels Set"].str.contains("non-hostile")).astype(int).values
ytr, yte = y(tr), y(te)
URL = re.compile(r"https?://\S+|www\.\S+"); MEN = re.compile(r"@\w+"); HASH = re.compile(r"#\S+")
clean = lambda s: HASH.sub(" ", MEN.sub(" ", URL.sub(" ", s)))
classical = {"raw": np.array(json.load(open(f"{OUT}/classical_preds.json"))["word+char|LR"][0])}
C = pd.read_csv(f"{OUT}/ablation_SVM.csv").set_index("variant").loc["no URLs/mentions/hashtags", "C"]
v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
Xtr = v.fit_transform([clean(str(s)) for s in tr["Post"].fillna("")])
classical["clean"] = LinearSVC(C=C, class_weight="balanced", random_state=42).fit(Xtr, ytr).predict(v.transform([clean(str(s)) for s in te["Post"].fillna("")]))
cname = {"raw": "LR, word + char", "clean": "SVM, char"}
rng = np.random.default_rng(42); n = len(yte); B = 2000
idx = rng.integers(0, n, (B, n))
rows = []
for text in ("raw", "clean"):
    pc = classical[text]
    for model in ("mBERT", "MuRIL"):
        for seed in (13, 77):
            pt = np.load(f"{PRED}/{model}_{text}_coarse_{seed}.npy")
            ct, cc = pt == yte, pc == yte
            b, c = int((ct & ~cc).sum()), int((~ct & cc).sum())   # transformer right only / classical right only
            p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
            diffs = [f1_score(yte[i], pt[i], average="macro") - f1_score(yte[i], pc[i], average="macro") for i in idx]
            r = dict(text=text, transformer=model, seed=seed, classical=cname[text],
                     f1_transformer=f1_score(yte, pt, average="macro"), f1_classical=f1_score(yte, pc, average="macro"),
                     diff=f1_score(yte, pt, average="macro") - f1_score(yte, pc, average="macro"),
                     ci_low=float(np.percentile(diffs, 2.5)), ci_high=float(np.percentile(diffs, 97.5)),
                     only_transformer_right=b, only_classical_right=c, mcnemar_p=p)
            rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/significance_tests.csv", index=False)
