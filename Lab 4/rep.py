import pandas as pd
from pathlib import Path
import csv


def write_reports(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "diff_dataset.csv"
    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"[saved] {csv_path}")

    required = ["Source", "Test", "README", "LICENSE"]
    sub = df[df["category"].isin(required)]
    mismatches = sub[sub["Discrepancy"] == "Yes"].groupby("category").size().reindex(required, fill_value=0)
    stats_df = mismatches.reset_index()
    stats_df.columns = ["category", "num_mismatches"]
    stats_csv = outdir / "mismatches_by_category.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"[saved] {stats_csv}")

    try:
        import matplotlib.pyplot as plt
        equal_counts = df["Discrepancy"].value_counts().reindex(["No","Yes"], fill_value=0)
        plt.figure(figsize=(5,5))
        plt.pie(equal_counts, labels=["Equal Diffs","Non-Equal Diffs"], autopct="%1.1f%%", startangle=140)
        plt.title("Equal vs Non-Equal Diffs (All Files)")
        pie_path = outdir / "equal_diffs_pie.png"
        plt.savefig(pie_path, dpi=160)
        plt.close()
        print(f"[saved] {pie_path}")
    except Exception as e:
        print(f"[warn] Could not generate equal diffs pie: {e}")

    df["is_code"] = df["category"].isin(["Source","Test"])
    summary = df.groupby(["is_code","Discrepancy"]).size().unstack(fill_value=0)
    summary_csv = outdir / "code_noncode_matches.csv"
    summary.to_csv(summary_csv)
    print(f"[saved] {summary_csv}")

    try:
        code_counts = summary.loc[True] if True in summary.index else pd.Series()
        noncode_counts = summary.loc[False] if False in summary.index else pd.Series()

        if not code_counts.empty:
            plt.figure(figsize=(5,5))
            plt.pie(code_counts, labels=["Equal Diffs","Non-Equal Diffs"], autopct="%1.1f%%")
            plt.title("Diffs for Code Artifacts")
            code_pie = outdir / "code_artifacts_pie.png"
            plt.savefig(code_pie, dpi=160)
            plt.close()
            print(f"[saved] {code_pie}")

        if not noncode_counts.empty:
            plt.figure(figsize=(5,5))
            plt.pie(noncode_counts, labels=["Equal Diffs","Non-Equal Diffs"], autopct="%1.1f%%")
            plt.title("Diffs for Non-Code Artifacts")
            noncode_pie = outdir / "noncode_artifacts_pie.png"
            plt.savefig(noncode_pie, dpi=160)
            plt.close()
            print(f"[saved] {noncode_pie}")
    except Exception as e:
        print(f"[warn] Could not generate code/non-code pies: {e}")

    try:
        def get_ext(row):
            path = row["new_file_path"] if pd.notna(row["new_file_path"]) and row["new_file_path"] else row["old_file_path"]
            if pd.isna(path) or not path:
                return "(no_ext)"
            path = str(path)
            if "." in path:
                return Path(path).suffix.lower()
            return "(no_ext)"

        df["file_ext"] = df.apply(get_ext, axis=1)

        ext_counts = df["file_ext"].value_counts().head(20)  # top 20 extensions
        ext_csv = outdir / "file_extension_distribution.csv"
        ext_counts.to_csv(ext_csv, header=["count"])
        print(f"[saved] {ext_csv}")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,6))
        ext_counts.plot(kind="bar")
        plt.title("Top 20 File Extensions in Dataset")
        plt.xlabel("Extension")
        plt.ylabel("Number of Files Changed")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        ext_plot = outdir / "file_extension_distribution.png"
        plt.savefig(ext_plot, dpi=160)
        plt.close()
        print(f"[saved] {ext_plot}")
    except Exception as e:
        print(f"[warn] Could not generate file extension plot: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate extra reports/plots from diff_dataset.csv")
    parser.add_argument("csv", help="Path to diff_dataset.csv")
    parser.add_argument("--outdir", default=None, help="Output directory for plots (defaults to same folder as csv/reports)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        # default: same folder as csv /reports (keep your current structure)
        outdir = csv_path.parent

    df = pd.read_csv(csv_path)
    write_reports(df, outdir)
    print(f"Reports written to {outdir}")

