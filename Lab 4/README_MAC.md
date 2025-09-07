# CS202 Lab 4 — Diff Algorithms (Myers vs Histogram) — Mac Setup & Run Guide

This bundle gives you **ready-to-run** code that clones three open‑source repositories, runs `git diff` with two algorithms (Myers and Histogram) while **ignoring whitespace and blank lines**, creates a consolidated **CSV dataset**, and produces the required **plots & summary tables**.

---

## 0) What you’ll need (macOS)
- macOS 12+ (Monterey or newer). Works on Intel and Apple Silicon.
- **Git** (via Xcode Command Line Tools or Homebrew)
- **Python 3.10+** (3.11 recommended). If your Mac already has **Python 3.13**, that’s fine too.

> If Git or Python are missing, follow the steps below.

### Install Git
```bash
xcode-select --install   # accepts GUI prompt; installs git
# or, via Homebrew:
# brew install git
```

### Install Python 3 (if needed)
```bash
# if Homebrew is installed:
brew install python@3.11
python3 --version
```

---

## 1) Create a project folder and virtual environment
```bash
# (a) move this folder somewhere convenient, e.g. ~/Desktop/cs202-lab4
cd ~/Desktop
mkdir -p cs202-lab4 && cd cs202-lab4

# (b) copy the files you downloaded into this folder (requirements.txt, lab4_diff_analysis.py, repos.txt)

# (c) create & activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

---

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## 3) Choose repositories (selection criteria you can put in the report)
We provide a starter list in `repos.txt`:
- `JakeWharton/butterknife`
- `psf/requests`
- `pallets/flask`

**Example selection criteria (justify in your report):**
- Stars ≥ 5,000 and Forks ≥ 1,000 (SEART GH Search Engine can help)
- Active in the last 12 months
- Has both `/src` (or code) and `/tests` directories
- Contains README and LICENSE

You can search via SEART: <https://seart-ghs.si.usi.ch/> and paste chosen `owner/repo` into `repos.txt` (one per line).

---

## 4) Run the script
Option A — use the sample list:
```bash
python lab4_diff_analysis.py --repos-file repos.txt --max-commits 200 --outdir outputs
```

Option B — specify repos inline:
```bash
python lab4_diff_analysis.py --repos psf/requests pallets/flask JakeWharton/butterknife --max-commits 200 --outdir outputs
```

- The script will clone repos into `./repos/<owner>__<name>` (only once).
- It mines **non-merge commits** and, for each modified file, runs:

  ```
  git diff -w --ignore-blank-lines --unified=0 --diff-algorithm=myers     <parent> <commit> -- <path>
  git diff -w --ignore-blank-lines --unified=0 --diff-algorithm=histogram <parent> <commit> -- <path>
  ```

- Output files go to `outputs/reports/`:
  - `diff_dataset.csv`  (final dataset with columns required by the assignment)
  - `mismatches_by_category.csv` (summary table)
  - `mismatches_by_category.png` (bar chart)

> Tip: Increase `--max-commits` to scan more history (or `0` for full history). For very large repos, start with 100–200 to keep run-time reasonable.

---

## 5) What’s inside the dataset?
Columns:
- `repo`
- `old_file_path`, `new_file_path`
- `commit_sha`, `parent_commit_sha`
- `commit_message`
- `diff_myers`, `diff_hist` (plain-text patches)
- `category` ∈ {Source, Test, README, LICENSE, Other}
- `Discrepancy` ∈ {Yes, No}  → **Yes means a mismatch** between the normalized patches.

Normalization removes diff headers and compares only `+`/`-` lines with zero context. Whitespace and blank lines are already ignored via `-w` and `--ignore-blank-lines` in `git diff`.

---

## 6) Plots & stats to paste in your report
- `mismatches_by_category.png` shows **#Mismatches** (Discrepancy == “Yes”) for:
  - Source Code files
  - Test Code files
  - README files
  - LICENSE files

Include the PNG and the `mismatches_by_category.csv` table in your submission.

---

## 7) Reproducibility notes
- The script skips **merge commits** (matches typical mining setups).
- Binary files are skipped.
- For renamed files, Git may produce empty or simple diffs when there’s no line change; we store paths and patch text as-is.
- You can rerun the script with different repos or commit limits; outputs are overwritten.

---

## 8) Troubleshooting (macOS)
**Q:** `git: command not found`  
**A:** Run `xcode-select --install` or `brew install git`.

**Q:** SSL or network errors when cloning  
**A:** Try again on a stable network; optionally `git config --global http.postBuffer 524288000` for large repos.

**Q:** Python packages fail to build on Apple Silicon  
**A:** Ensure Command Line Tools are installed; try Python 3.11 via Homebrew.

**Q:** “Permission denied” when running the script  
**A:** `chmod +x lab4_diff_analysis.py` or invoke with `python lab4_diff_analysis.py`.

---

## 9) How to explain part (f) in your report — “Which algorithm is better?”
You can automate an assessment as follows:
1. **Define quality signals** for a diff (e.g., fewer spurious hunks, better grouping of moved/duplicated lines).  
2. **Proxy ground truth:** approximate developer‑intent via tests or an AST‑based differ; compute how well each algorithm’s diff aligns with these signals.  
3. **Operationalize metrics:** e.g., (i) number of hunks, (ii) percentage of changed lines that are part of syntactic elements (via a lightweight parser), (iii) edit script length, (iv) stability across whitespace/blank‑line options.  
4. **Decide per file type:** rank algorithms per commit/file and aggregate. Prior work suggests `--histogram` often yields better code changes than default Myers for source code; confirm empirically on your sample.

Cite the Git docs and literature in your report.

---

**You’re set. Good luck!**
