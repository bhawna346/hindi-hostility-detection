import sys, re, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
D=sys.argv[1]
tr,va,te=(pd.read_csv(f"{D}/{s}.csv") for s in ["train","valid","test"])
y=lambda d:(~d["Labels Set"].str.contains("non-hostile")).astype(int).values
URL=re.compile(r"https?://\S+|www\.\S+");MEN=re.compile(r"@\w+");HASH=re.compile(r"#\S+")
cl=lambda s:HASH.sub(" ",MEN.sub(" ",URL.sub(" ",s)))
for name,fn in [("raw",lambda s:s),("no URLs",lambda s:URL.sub(" ",s)),("no URLs/mentions/hashtags",cl)]:
    X={k:[fn(str(s)) for s in d["Post"].fillna("")] for k,d in [("tr",tr),("va",va),("te",te)]}
    v=TfidfVectorizer(analyzer="word",ngram_range=(1,2),min_df=2,sublinear_tf=True,token_pattern=r"(?u)\S+")
    a=v.fit_transform(X["tr"]);b=v.transform(X["va"]);c=v.transform(X["te"])
    C=max((0.3,1,3,10),key=lambda C:f1_score(y(va),LogisticRegression(C=C,max_iter=3000,class_weight="balanced").fit(a,y(tr)).predict(b),average="macro"))
    print(name,C,round(f1_score(y(te),LogisticRegression(C=C,max_iter=3000,class_weight="balanced").fit(a,y(tr)).predict(c),average="macro"),4))
