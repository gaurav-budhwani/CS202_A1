#!/usr/bin/env python3
"""
CS202 Lab 4 — Exploration of different diff algorithms on Open-Source Repositories
Generates a consolidated CSV with git diffs (Myers vs Histogram), and a summary plot.

Usage examples:
  python lab4_diff_analysis.py --repos JakeWharton/butterknife psf/requests pallets/flask
  python lab4_diff_analysis.py --repos-file repos.txt --max-commits 300 --outdir outputs

This script:
  1) Clones the repos locally (if needed) to ./repos/<owner>__<name>
  2) Mines non-merge commits using PyDriller
  3) For each modified file, runs:
       git diff -w --ignore-blank-lines --unified=0 --diff-algorithm=myers     <parent> <commit> -- <path>
       git diff -w --ignore-blank-lines --unified=0 --diff-algorithm=histogram <parent> <commit> -- <path>
     and stores both patches in the CSV
  4) Adds a Discrepancy column = "Yes" if the two diffs differ after normalization, else "No"
  5) Classifies files into categories: Source, Test, README, LICENSE, Other
  6) Produces summary CSVs and bar plots under <outdir>/reports

Notes:
- Requires: git (CLI), Python 3.10+, and the packages in requirements.txt
- You can tune --max-commits per repo for performance.
"""
import argparse
import csv
import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
from tqdm import tqdm
from pydriller import Repository


def check_git_available() -> None:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except Exception as e:
        raise SystemExit("ERROR: git is not installed or not on PATH. Install Xcode Command Line Tools or Git.") from e

def repo_dir_name(owner_repo: str) -> str:
    owner_repo = owner_repo.strip().strip("/")
    owner, name = owner_repo.split("/", 1)
    return f"{owner}__{name}"

def clone_repo(owner_repo: str, base_dir: Path) -> Path:
    """
    Clone the repo if needed. Returns the local path.
    """
    local = base_dir / repo_dir_name(owner_repo)
    if local.exists() and (local / ".git").exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{owner_repo}.git"
    print(f"[clone] {owner_repo} -> {local}")
    # Full clone (history). You may replace with --depth to speed up, but you’ll miss older commits.
    subprocess.run(["git", "clone", "--no-tags", url, str(local)], check=True)
    return local

def run_git_diff(repo_path: Path, parent: str, commit: str, pathspec: str, algorithm: str) -> str:
    """
    Run git diff for a single file with given algorithm, ignoring whitespace/blank lines, zero context.
    Returns the raw patch text (may be empty).
    """
    cmd = [
        "git", "-C", str(repo_path),
        "diff",
        "-w", "--ignore-blank-lines",
        "--unified=0",
        f"--diff-algorithm={algorithm}",
        parent, commit, "--", pathspec
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    # We do not check returncode because git diff returns 1 when differences exist if used with certain flags.
    return res.stdout or ""

HEADER_STRIP_RE = re.compile(r"^(diff --git|index [0-9a-f]+\.\.[0-9a-f]+|--- |\+\+\+ |@@ )")

def normalize_patch(patch: str) -> str:
    """
    Normalize a patch by removing header lines and keeping only added/removed markers.
    We also strip trailing spaces and skip empty lines, to be conservative in 'match' judgment.
    """
    norm_lines = []
    for line in patch.splitlines():
        if HEADER_STRIP_RE.match(line):
            continue
        if line.startswith("+") or line.startswith("-"):
            s = line.rstrip()
            if s and s not in {"+", "-"}:
                norm_lines.append(s)
    return "\n".join(norm_lines).strip()

def classify_category(path: Optional[str]) -> str:
    """
    Classify file path into Source, Test, README, LICENSE, Other.
    """
    if not path:
        return "Other"
    pname = Path(path)
    name = pname.name.lower()

    if name.startswith("readme"):
        return "README"
    if name.startswith("license") or name in {"copying", "licence"}:
        return "LICENSE"
    if re.search(r"(^|/)(tests?|spec|__tests__|testdata)(/|$)", path.lower().replace("\\", "/")):
        return "Test"
    if re.match(r"(test_|_test\.)", name):
        return "Test"

    CODE_EXT = {
        ".py",".java",".c",".cc",".cpp",".h",".hpp",".cs",".go",".rs",".rb",".kt",".kts",
        ".swift",".m",".mm",".scala",".php",".js",".jsx",".ts",".tsx",".sh",".bash",".zsh",
        ".r",".m4",".pl",".lua",".sql",".css",".html",".xml",".yml",".yaml",".gradle",".kt",
    }
    if pname.suffix.lower() in CODE_EXT:
        return "Source"
    return "Other"

def safe_pathspec(mod_old: Optional[str], mod_new: Optional[str]) -> str:
    """
    Choose a pathspec that should work for 'git diff <parent> <commit> -- <pathspec>'.
    Prefer new path when available, otherwise old path.
    """
    return mod_new or mod_old or ""


def mine_repo(owner_repo: str, out_rows: List[Dict], repos_base: Path, max_commits: Optional[int]) -> None:
    local_path = clone_repo(owner_repo, repos_base)
    counted = 0
    for commit in Repository(str(local_path), only_no_merge=True).traverse_commits():
        parent_shas = commit.parents
        if not parent_shas:
            continue
        parent = parent_shas[0]
        for m in commit.modified_files:
            old_path = m.old_path
            new_path = m.new_path
            pathspec = safe_pathspec(old_path, new_path)
            if not pathspec:
                continue
            patch_myers = run_git_diff(local_path, parent, commit.hash, pathspec, "myers")
            patch_hist  = run_git_diff(local_path, parent, commit.hash, pathspec, "histogram")
            nm = normalize_patch(patch_myers)
            nh = normalize_patch(patch_hist)
            discrepancy = "Yes" if nm != nh else "No"
            category = classify_category(new_path or old_path)
            out_rows.append({
                "repo": owner_repo,
                "old_file_path": old_path or "",
                "new_file_path": new_path or "",
                "commit_sha": commit.hash,
                "parent_commit_sha": parent,
                #"commit_message": commit.msg.strip().replace("\n", " "),
                "commit_message": commit.msg.encode("utf-8", errors="replace").decode("utf-8"),
                "diff_myers": patch_myers,
                "diff_hist": patch_hist,
                "category": category,
                "Discrepancy": discrepancy,
            })
        counted += 1
        if max_commits and counted >= max_commits:
            break

def write_reports(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Final dataset CSV
    csv_path = outdir / "diff_dataset.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[saved] {csv_path}")

    # Discrepancy stats by category (only required buckets)
    required = ["Source", "Test", "README", "LICENSE"]
    sub = df[df["category"].isin(required)]
    mismatches = sub[sub["Discrepancy"] == "Yes"].groupby("category").size().reindex(required, fill_value=0)
    stats_df = mismatches.reset_index()
    stats_df.columns = ["category", "num_mismatches"]
    stats_csv = outdir / "mismatches_by_category.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[saved] {stats_csv}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        plt.bar(stats_df["category"], stats_df["num_mismatches"])
        plt.title("#Mismatches by File Category")
        plt.xlabel("Category")
        plt.ylabel("#Mismatches")
        plot_path = outdir / "mismatches_by_category.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        print(f"[saved] {plot_path}")
    except Exception as e:
        print(f"[warn] Could not generate plot: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description="CS202 Lab 4: Diff algorithms comparison")
    ap.add_argument("--repos", nargs="*", help="List of owner/repo identifiers (e.g., psf/requests)")
    ap.add_argument("--repos-file", type=str, help="Path to a text file with one owner/repo per line")
    ap.add_argument("--max-commits", type=int, default=200, help="Max commits per repo to analyze (default=200). Use 0 for all.")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory (will create reports/ under it)")
    return ap.parse_args()

def load_repos(args) -> List[str]:
    repos = []
    if args.repos:
        repos.extend(args.repos)
    if args.repos_file:
        with open(args.repos_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    repos.append(line)
    if not repos:
        raise SystemExit("Provide --repos or --repos-file")
    return repos

def main():
    args = parse_args()
    check_git_available()
    repos = load_repos(args)

    out_rows = []
    repos_base = Path("repos")
    for r in repos:
        print(f"=== Mining {r} ===")
        try:
            mine_repo(r, out_rows, repos_base, None if args.max_commits == 0 else args.max_commits)
        except subprocess.CalledProcessError as e:
            print(f"[error] Git error on {r}: {e}")
        except Exception as e:
            print(f"[error] Skipping {r} due to: {e}")

    if not out_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(out_rows)
    reports_dir = Path(args.outdir) / "reports"
    write_reports(df, reports_dir)

if __name__ == "__main__":
    main()
