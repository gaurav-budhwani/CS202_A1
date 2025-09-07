import argparse
import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
import sacrebleu
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze as raw_analyze
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x)

def compute_radon_metrics(code):
    try:
        if not code or code.strip() == "":
            return (np.nan, np.nan, np.nan)
        mi_score = mi_visit(code, True)
        cc_blocks = cc_visit(code)
        cc_sum = sum([b.complexity for b in cc_blocks]) if cc_blocks else 0.0
        raw = raw_analyze(code)
        sloc = raw.sloc
        return (float(mi_score), float(cc_sum), int(sloc))
    except Exception:
        return (np.nan, np.nan, np.nan)

def bleu_score(reference, hypothesis):
    try:
        if safe_str(reference).strip() == "" or safe_str(hypothesis).strip() == "":
            return np.nan
        sc = sacrebleu.sentence_bleu(hypothesis, [reference])
        return float(sc.score) / 100.0
    except Exception:
        return np.nan

class CodeBERTBatchEmbedder:
    def __init__(self, model_name="microsoft/codebert-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CodeBERT] device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pool(self, last_hidden_state, attention_mask):
        attn = attention_mask.unsqueeze(-1)
        summed = (last_hidden_state * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        return (summed / counts).cpu().numpy()

    def embed_batch(self, texts, max_length=512, batch_size=32):
        """
        texts: list[str]
        returns: list[np.array] (None for empty strings)
        """
        out = [None] * len(texts)
        # indices with non-empty text
        nonempty = [i for i,t in enumerate(texts) if t and t.strip() != ""]
        for i in range(0, len(nonempty), batch_size):
            batch_idx = nonempty[i:i+batch_size]
            batch_texts = [texts[j] for j in batch_idx]
            toks = self.tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            input_ids = toks["input_ids"].to(self.device)
            attn = toks["attention_mask"].to(self.device)
            with torch.no_grad():
                res = self.model(input_ids=input_ids, attention_mask=attn)
            last_hidden = res.last_hidden_state
            pooled = self._mean_pool(last_hidden, attn)  # shape (B, hidden)
            for k, idx in enumerate(batch_idx):
                out[idx] = pooled[k]
        return out

def cosine_sim(a, b):
    if a is None or b is None:
        return np.nan
    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def prepare_columns(df):
    float_cols = ["MI_Before","MI_After","CC_Before","CC_After","LOC_Before","LOC_After",
                  "MI_Change","CC_Change","LOC_Change","Semantic_Similarity","Token_Similarity"]
    obj_cols = ["Semantic_Class","Token_Class","Classes_Agree"]
    for c in float_cols:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = df[c].astype(float)
    for c in obj_cols:
        if c not in df.columns:
            df[c] = pd.Series([None]*len(df), dtype=object)
        else:
            df[c] = df[c].astype(object)
    return df

def main(args):
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    df = prepare_columns(df)
    print("\n=== Baseline Descriptive Statistics ===")
    total_commits = df["hash"].nunique() if "hash" in df.columns else np.nan
    total_files = df["filename"].nunique() if "filename" in df.columns else np.nan
    avg_files_per_commit = (
        df.groupby("hash")["filename"].nunique().mean()
        if "hash" in df.columns and "filename" in df.columns else np.nan
    )

    print("Total commits:", total_commits)
    print("Total files:", total_files)
    print("Avg. modified files per commit:", avg_files_per_commit)

    if "LLM Inference (fix type)" in df.columns:
        print("\nFix type distribution:")
        print(df["LLM Inference (fix type)"].value_counts().head(10))  # top 10 only

    if "filename" in df.columns:
        print("\nTop filenames:")
        print(df["filename"].value_counts().head(10))

        df["extension"] = df["filename"].astype(str).apply(
            lambda x: x.split(".")[-1] if "." in x else ""
        )
        print("\nTop file extensions:")
        print(df["extension"].value_counts().head(10))
    print("=" * 35)
    if args.resume and os.path.exists(args.output):
        print("Resume mode: loading existing output and skipping completed rows.")
        prev = pd.read_csv(args.output)
        if "Semantic_Similarity" in prev.columns:
            df = df.copy()
            prev_map = prev.set_index("hash") if "hash" in prev.columns else None
            # simpler: mark rows that have non-null Semantic_Similarity in prev by matching an index column if available
            # we'll do per-row check by merging on all identifying columns (hash+filename+message) if present
            # but easiest approach: if prev has same length -> assume done
            if len(prev) == len(df):
                print("Output has same number of rows as input — assuming processing already finished. Exiting.")
                return
            else:
                print("Partial output detected but different shape; proceeding to recompute (no complex resume).")

    total_commits = df["hash"].nunique() if "hash" in df.columns else np.nan
    total_files = df["filename"].nunique() if "filename" in df.columns else np.nan
    avg_files_per_commit = df.groupby("hash")["filename"].nunique().mean() if "hash" in df.columns and "filename" in df.columns else np.nan
    print("Total commits:", total_commits)
    print("Total files:", total_files)
    print("Avg files/commit:", avg_files_per_commit)

    # compute radon metrics
    use_radon_mask = df["filename"].astype(str).str.endswith(".py") | args.force_radon
    if args.force_radon:
        print("Force radon for non-.py files enabled.")

    # compute radon metrics for rows
    for idx in tqdm(df.index, desc="Radon (structural)"):
        if use_radon_mask.iloc[df.index.get_loc(idx)]:
            before = safe_str(df.at[idx, "source_code_before"]) if "source_code_before" in df.columns else ""
            after = safe_str(df.at[idx, "source_code_after"]) if "source_code_after" in df.columns else ""
            mi_b, cc_b, loc_b = compute_radon_metrics(before)
            mi_a, cc_a, loc_a = compute_radon_metrics(after)
        else:
            mi_b = mi_a = cc_b = cc_a = loc_b = loc_a = np.nan
        df.at[idx, "MI_Before"] = mi_b
        df.at[idx, "MI_After"]  = mi_a
        df.at[idx, "CC_Before"] = cc_b
        df.at[idx, "CC_After"]  = cc_a
        df.at[idx, "LOC_Before"] = loc_b
        df.at[idx, "LOC_After"]  = loc_a
        # changes
        df.at[idx, "MI_Change"]  = (mi_a - mi_b) if (not pd.isna(mi_a) and not pd.isna(mi_b)) else np.nan
        df.at[idx, "CC_Change"]  = (cc_a - cc_b) if (not pd.isna(cc_a) and not pd.isna(cc_b)) else np.nan
        df.at[idx, "LOC_Change"] = (loc_a - loc_b) if (not pd.isna(loc_a) and not pd.isna(loc_b)) else np.nan

    # Token similarity (BLEU)
    print("Computing token similarity (BLEU)...")
    for idx in tqdm(df.index, desc="BLEU"):
        before = safe_str(df.at[idx, "source_code_before"]) if "source_code_before" in df.columns else ""
        after  = safe_str(df.at[idx, "source_code_after"]) if "source_code_after" in df.columns else ""
        df.at[idx, "Token_Similarity"] = bleu_score(before, after)

    # Semantic similarity (batch embed)
    if not args.skip_semantic:
        print("Preparing CodeBERT embedder...")
        embedder = CodeBERTBatchEmbedder()
        # collect unique texts (to avoid repeated embedding)
        texts = []
        idx_map_before = []
        idx_map_after = []
        for idx in df.index:
            before = safe_str(df.at[idx, "source_code_before"]) if "source_code_before" in df.columns else ""
            after  = safe_str(df.at[idx, "source_code_after"]) if "source_code_after" in df.columns else ""
            texts.append(before)
            texts.append(after)
            idx_map_before.append(len(texts)-2)
            idx_map_after.append(len(texts)-1)
        # unique mapping
        unique_texts, inverse_idx = np.unique(np.array(texts, dtype=object), return_inverse=True)
        unique_texts = unique_texts.tolist()
        print(f"Unique code snippets to embed: {len(unique_texts)}")
        # embed in batches
        embeddings = embedder.embed_batch(unique_texts, batch_size=args.batch_size)
        # map back
        for row_i, idx in enumerate(df.index):
            emb_b = embeddings[inverse_idx[2*row_i]]
            emb_a = embeddings[inverse_idx[2*row_i + 1]]
            df.at[idx, "Semantic_Similarity"] = cosine_sim(emb_b, emb_a)

    else:
        print("Skipping semantic similarity (CodeBERT) as requested (--skip-semantic).")

    print("Classifying and marking agreement...")
    for idx in tqdm(df.index, desc="Classify"):
        sem_sim = df.at[idx, "Semantic_Similarity"]
        tok_sim = df.at[idx, "Token_Similarity"]
        sem_class = None
        tok_class = None
        if not pd.isna(sem_sim):
            sem_class = "Minor" if sem_sim >= args.semantic_threshold else "Major"
        if not pd.isna(tok_sim):
            tok_class = "Minor" if tok_sim >= args.token_threshold else "Major"
        df.at[idx, "Semantic_Class"] = sem_class
        df.at[idx, "Token_Class"] = tok_class
        if sem_class is not None and tok_class is not None:
            df.at[idx, "Classes_Agree"] = "YES" if sem_class == tok_class else "NO"
        else:
            df.at[idx, "Classes_Agree"] = None

    # save
    df.to_csv(args.output, index=False)
    print(f"Wrote output to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--skip-semantic", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--semantic-threshold", type=float, default=0.80)
    p.add_argument("--token-threshold", type=float, default=0.75)
    p.add_argument("--force-radon", action="store_true")
    p.add_argument("--resume", action="store_true", help="attempt to resume if output exists (simple check)")
    args = p.parse_args()
    main(args)
