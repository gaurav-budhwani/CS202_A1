import pandas as pd
import os
import numpy as np
from sklearn.utils import resample

df = pd.read_csv("diff_rectified.csv")

sample_size = 50
sample_df = df.sample(n=sample_size, random_state=42)

ann_file = "annotations.csv"

if not os.path.exists(ann_file):
    # Create a fresh template
    template = sample_df[["hash", "filename", "message", "LLM Inference (fix type)", "Rectified Message"]].copy()
    template["dev_score"] = ""        # to be filled manually (0/1/2)
    template["llm_score"] = ""
    template["rectified_score"] = ""
    template["notes"] = ""
    template.to_csv(ann_file, index=False)
    print(f"Annotation template saved as {ann_file}. Open it in Excel/Sheets and score commits (0/1/2).")
    exit()

ann = pd.read_csv(ann_file)

def hit_rate(scores):
    return (scores == 2).mean()

def partial_rate(scores):
    return (scores == 1).mean()

def ci_bootstrap(scores, n=2000, alpha=0.05):
    arr = np.array(scores == 2, dtype=float)
    boots = [resample(arr, replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

for col, label in [("dev_score","Developer"),
                   ("llm_score","LLM"),
                   ("rectified_score","Rectifier")]:
    if col not in ann.columns:
        continue
    vals = pd.to_numeric(ann[col], errors="coerce").dropna().astype(int)
    if len(vals) == 0:
        continue
    hr = hit_rate(vals)
    pr = partial_rate(vals)
    lo, hi = ci_bootstrap(vals)
    print(f"{label}: hit rate={hr:.2%} (95% CI {lo:.2%}–{hi:.2%}), partial={pr:.2%}")

