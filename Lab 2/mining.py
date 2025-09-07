import pandas as pd
from pydriller import Repository
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# repo
repo = "https://github.com/airbnb/javascript"
keywords = [
    "fix", "bug", "issue", "resolve", "bugfix", "fixes", "fixed",
    "crash", "solves", "testcase", "fail", "except",
    "overflow", "break", "workaround", "stop", "failure",
    "crash", "freez", "problem", " problem", "avoid", "break",
    " break", "issue", " issue", "error", "test", "hang"
]

# manual rectifier
def my_rectifier(filename, diff, org_msg):
    fname = filename.lower()
    difft = str(diff).lower()

    if "test" in fname or "assert" in difft:
        return f"Update tests in {filename}"
    elif fname.endswith(".py"):
        if any(i in org_msg.lower() for i in ["fix", "bug"]):
            return f"Fix bug in {filename}"
        else:
            return f"Modify Python code in {filename}"
    elif fname.endswith(".md") or fname.endswith(".rst"):
        return f"Update documentation in {filename}"
    elif fname.endswith(".html") or fname.endswith(".css"):
        return f"Update UI layout in {filename}"
    else:
        return f"Update {filename}"

# commits and diffs
commits = []
diffs = []

for c in Repository(repo).traverse_commits():
    if any(i in c.msg.lower() for i in keywords):
        commits.append({
            "hash": c.hash,
            "message": c.msg,
            "hashes_of_parents": [h for h in c.parents],
            "is_merge_commit": len(c.parents) > 1,
            "list_of_modified_files": [m.filename for m in c.modified_files]
        })
        for m in c.modified_files:
            diffs.append({
                "hash": c.hash,
                "message": c.msg,
                "filename": m.filename,
                "source_code_before": m.source_code_before,
                "source_code_after": m.source_code,
                "diff": m.diff
            })

bugfix = pd.DataFrame(commits)
bugfix.to_csv("bugfix.csv", index=False)

diff = pd.DataFrame(diffs)
diff.to_csv("diff.csv", index=False)

print(f"Saved {len(bugfix)} bugfix commits and {len(diff)} diffs")

# LLM
print("Loading LLM model (this may take a while)...")
model_name = "mamiksik/CommitPredictorT5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def llm_predict(diff_text):
    if not diff_text or diff_text.strip() == "":
        return "No changes"
    inputs = tokenizer(diff_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# add to the database
df = pd.read_csv("diff.csv")

df["LLM Inference (fix type)"] = df["diff"].apply(lambda d: llm_predict(str(d)))
df["Rectified Message"] = df.apply(
    lambda row: my_rectifier(
        row.get("filename", ""),
        row.get("diff", ""),
        row.get("message", ""),
    ), axis=1
)

# save
df.to_csv("diff_rectified.csv", index=False)

print("Final dataset saved as diff_rectified.csv")
print("Columns: hash, message (developer), filename, source_code_before, source_code_after, diff, llm_message, rectified_message")
