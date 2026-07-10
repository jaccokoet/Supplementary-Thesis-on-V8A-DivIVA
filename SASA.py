from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# -----------------------------
# Settings
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input_files"
GRAPH_DIR = BASE_DIR / "graphs"
OUTPUT_DIR = BASE_DIR / "output_files"

# Your SASA time-series .xvg files say Time (ps), so usually this should be True.
# If your x-axis is already ns, change this to False.
CONVERT_PS_TO_NS = True

# Optional: ignore early simulation time for summary statistics.
# Example: set to 200 if you only want stats after 200 ns.
EQUILIBRATION_CUTOFF_NS = 50

# Residue plots can get crowded; this marks mutated residue V8 if present.
MUTATION_RESIDUE = 8

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.titlesize": 35,
    "axes.labelsize": 35,
    #"axes.titlesize": 25,
    #"axes.labelsize": 25,
    "xtick.labelsize": 20,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 20,
})


# -----------------------------
# Custom colors
# -----------------------------
light_pink = "#ff98cb"
pink = "#ff69b3"
dark_pink = "#ec3499"
darkest_pink = "#cb1378"
light_blue = "#65a9ed"
blue = "#65a9ed"
dark_blue = "#1546c7"
darkest_blue = "#000081"

VARIANT_COLORS = {
    "wt": dark_blue,
    "v8a": dark_pink,
}

VARIANT_FILL_COLORS = {
    "wt": light_blue,
    "v8a": light_pink,
}

REPLICATE_COLORS = {
    ("wt", "repl1"): light_blue,
    ("wt", "repl2"): blue,
    ("wt", "repl3"): darkest_blue,
    ("v8a", "repl1"): light_pink,
    ("v8a", "repl2"): pink,
    ("v8a", "repl3"): darkest_pink,
}


# -----------------------------
# Reading and parsing
# -----------------------------
def make_dirs() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(filename: str) -> dict:
    """
    Parse filenames like:
        mem_dim_v8a_repl1_chain_a_sasa.xvg
        pro_dim_wt_repl2_protein_sasa.xvg
        pro_mon_v8a_repl3_protein_sasa_residue.xvg
    """
    stem = filename
    stem = stem.replace("_sasa_residue.xvg", "")
    stem = stem.replace("_sasa.xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem|pro)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)_"
        r"(?P<selection>protein|chain_a|chain_b)$"
    )

    match = pattern.match(stem)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    data = match.groupdict()
    data["system"] = f"{data['environment']}_{data['oligomer']}"
    data["comparison_group"] = f"{data['system']}_{data['selection']}"
    data["condition"] = f"{data['system']}_{data['variant']}_{data['selection']}"
    return data


def read_sasa_time_xvg(filepath: Path) -> pd.DataFrame:
    """
    Read a GROMACS gmx sasa time-series .xvg file.
    Uses the first numeric column as time and the second as total SASA.

    Example columns from your files:
        time(ps)  Total  chain_a/protein
    """
    rows = []

    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                time = float(parts[0])
                sasa = float(parts[1])
                rows.append((time, sasa))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric SASA time-series data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "sasa_nm2"])
    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]
    return df


def read_sasa_residue_xvg(filepath: Path) -> pd.DataFrame:
    """
    Read a GROMACS gmx sasa residue .xvg file.
    Uses:
        column 1 = residue number
        column 2 = average SASA per residue
        column 3 = SD SASA per residue

    Some files contain duplicated extra columns; these are ignored.
    """
    rows = []

    with filepath.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                residue = int(float(parts[0]))
                mean_sasa = float(parts[1])
                sd_sasa = float(parts[2])
                rows.append((residue, mean_sasa, sd_sasa))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric residue SASA data found in {filepath}")

    return pd.DataFrame(rows, columns=["residue", "mean_sasa_nm2", "sd_sasa_nm2"])


def add_metadata(df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    meta = parse_filename(filepath.name)
    for key, value in meta.items():
        df[key] = value
    df["source_file"] = filepath.name
    return df


def load_all_time_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_sasa.xvg"))
    files = [f for f in files if not f.name.endswith("_sasa_residue.xvg")]

    if not files:
        raise FileNotFoundError(f"No SASA time-series .xvg files found in {INPUT_DIR}")

    all_frames = []
    for filepath in files:
        df = read_sasa_time_xvg(filepath)
        df = add_metadata(df, filepath)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["replicate_number"] = combined["replicate"].str.replace("repl", "", regex=False).astype(int)
    combined = combined.sort_values(
        ["system", "selection", "variant", "replicate_number", "time_ns"]
    ).reset_index(drop=True)
    return combined


def load_all_residue_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_sasa_residue.xvg"))

    if not files:
        print(f"No residue SASA files found in {INPUT_DIR}; skipping residue analysis.")
        return pd.DataFrame()

    all_frames = []
    for filepath in files:
        df = read_sasa_residue_xvg(filepath)
        df = add_metadata(df, filepath)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["replicate_number"] = combined["replicate"].str.replace("repl", "", regex=False).astype(int)
    combined = combined.sort_values(
        ["system", "selection", "variant", "replicate_number", "residue"]
    ).reset_index(drop=True)
    return combined


# -----------------------------
# Summaries: time-series SASA
# -----------------------------
def calculate_replicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df[df["time_ns"] >= EQUILIBRATION_CUTOFF_NS].copy()

    summary = (
        stats_df.groupby(
            [
                "system",
                "environment",
                "oligomer",
                "selection",
                "comparison_group",
                "variant",
                "replicate",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            n_points=("sasa_nm2", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_sasa_nm2=("sasa_nm2", "mean"),
            median_sasa_nm2=("sasa_nm2", "median"),
            std_sasa_nm2=("sasa_nm2", "std"),
            min_sasa_nm2=("sasa_nm2", "min"),
            max_sasa_nm2=("sasa_nm2", "max"),
        )
        .sort_values(["system", "selection", "variant", "replicate"])
    )
    return summary


def calculate_wt_vs_v8a_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    condition_summary = (
        replicate_summary.groupby(
            ["system", "selection", "comparison_group", "variant"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_sasa_nm2=("mean_sasa_nm2", "mean"),
            sd_mean_sasa_nm2=("mean_sasa_nm2", "std"),
            mean_max_sasa_nm2=("max_sasa_nm2", "mean"),
            sd_max_sasa_nm2=("max_sasa_nm2", "std"),
        )
        .sort_values(["system", "selection", "variant"])
    )
    return condition_summary


def calculate_v8a_minus_wt_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Positive difference means V8A has higher SASA than WT.
    Only matching system/selection/replicate comparisons are included.
    """
    pivot = replicate_summary.pivot_table(
        index=["system", "selection", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_sasa_nm2", "max_sasa_nm2"],
    )

    rows = []
    for index, row in pivot.iterrows():
        system, selection, comparison_group, replicate = index
        if pd.isna(row.get(("mean_sasa_nm2", "wt"))) or pd.isna(row.get(("mean_sasa_nm2", "v8a"))):
            continue

        rows.append({
            "system": system,
            "selection": selection,
            "comparison_group": comparison_group,
            "replicate": replicate,
            "mean_sasa_difference_v8a_minus_wt_nm2": row[("mean_sasa_nm2", "v8a")] - row[("mean_sasa_nm2", "wt")],
            "max_sasa_difference_v8a_minus_wt_nm2": row[("max_sasa_nm2", "v8a")] - row[("max_sasa_nm2", "wt")],
        })

    return pd.DataFrame(rows)


def calculate_t_tests(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        wt = sub[sub["variant"] == "wt"]
        v8a = sub[sub["variant"] == "v8a"]

        #if len(wt) < 2 or len(v8a) < 2:
        #    continue

        mean_t, mean_p = stats.ttest_ind(
            wt["mean_sasa_nm2"],
            v8a["mean_sasa_nm2"],
            equal_var=False,
        )

        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_sasa_nm2": wt["mean_sasa_nm2"].mean(),
            "wt_sd_sasa_nm2": wt["mean_sasa_nm2"].std(),
            "v8a_mean_sasa_nm2": v8a["mean_sasa_nm2"].mean(),
            "v8a_sd_sasa_nm2": v8a["mean_sasa_nm2"].std(),
            "mean_sasa_t_statistic": mean_t,
            "mean_sasa_p_value": mean_p,
            "mean_sasa_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Summaries: residue SASA
# -----------------------------
def calculate_residue_condition_summary(residue_df: pd.DataFrame) -> pd.DataFrame:
    if residue_df.empty:
        return pd.DataFrame()

    summary = (
        residue_df.groupby(["system", "selection", "comparison_group", "variant", "residue"], as_index=False)
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_residue_sasa_nm2=("mean_sasa_nm2", "mean"),
            sd_between_replicates_nm2=("mean_sasa_nm2", "std"),
            mean_within_replicate_sd_nm2=("sd_sasa_nm2", "mean"),
        )
        .sort_values(["system", "selection", "variant", "residue"])
    )
    return summary


def calculate_residue_v8a_minus_wt(residue_summary: pd.DataFrame) -> pd.DataFrame:
    if residue_summary.empty:
        return pd.DataFrame()

    pivot = residue_summary.pivot_table(
        index=["system", "selection", "comparison_group", "residue"],
        columns="variant",
        values="mean_residue_sasa_nm2",
    ).reset_index()

    if "wt" not in pivot.columns or "v8a" not in pivot.columns:
        return pd.DataFrame()

    pivot = pivot.dropna(subset=["wt", "v8a"]).copy()
    pivot["residue_sasa_difference_v8a_minus_wt_nm2"] = pivot["v8a"] - pivot["wt"]
    return pivot.rename(columns={"wt": "wt_mean_residue_sasa_nm2", "v8a": "v8a_mean_residue_sasa_nm2"})


def save_tables(
    df: pd.DataFrame,
    residue_df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
    residue_summary: pd.DataFrame,
    residue_differences: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_sasa_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "sasa_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_sasa_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_sasa_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_sasa_t_tests.csv", index=False)

    if not residue_df.empty:
        residue_df.to_csv(OUTPUT_DIR / "all_residue_sasa_long_format.csv", index=False)
        residue_summary.to_csv(OUTPUT_DIR / "residue_sasa_summary_by_group.csv", index=False)
        residue_differences.to_csv(OUTPUT_DIR / "residue_sasa_v8a_minus_wt_by_group.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wt_vs_v8a_replicate_pairs_full_protein(df: pd.DataFrame) -> None:
    """
    Makes WT vs V8A SASA plots per replicate, only for full protein selections.
    """
    plot_df = df[df["selection"] == "protein"].copy()

    for (system, replicate), sub in plot_df.groupby(["system", "replicate"]):
        variants = set(sub["variant"])
        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")
            plt.plot(
                variant_df["time_ns"],
                variant_df["sasa_nm2"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("SASA (nm²)")
        plt.xlim(0, 300)
        plt.ylim(45,95)

        legend = plt.legend(title=f"{replicate.upper()}", loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{system}_{replicate}_wt_vs_v8a_full_protein_sasa.png")


def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(mean_sasa_nm2=("sasa_nm2", "mean"), sd_sasa_nm2=("sasa_nm2", "std"))
    )

    grouped["lower"] = grouped["mean_sasa_nm2"] - grouped["sd_sasa_nm2"].fillna(0)
    grouped["upper"] = grouped["mean_sasa_nm2"] + grouped["sd_sasa_nm2"].fillna(0)

    for comparison_group, sub in grouped.groupby("comparison_group"):
        variants = set(sub["variant"])
        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")
            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_sasa_nm2"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )
            plt.fill_between(
                variant_df["time_ns"],
                variant_df["lower"],
                variant_df["upper"],
                color=VARIANT_FILL_COLORS[variant],
                alpha=0.20,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("SASA (nm²)")
        plt.xlim(0, 300)
        plt.ylim(45,95)


        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"02_{comparison_group}_wt_vs_v8a_mean_sd_sasa.png")


def plot_wt_vs_v8a_mean_sasa_bars(replicate_summary: pd.DataFrame) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        variants = set(sub["variant"])
        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_sasa_nm2",
            errorbar="sd",
            capsize=0.15,
            order=["wt", "v8a"],
            palette=[dark_blue, dark_pink],
            err_kws={"color": "black", "linewidth": 2},
        )
        sns.stripplot(
            data=sub,
            x="variant",
            y="mean_sasa_nm2",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        plt.xlabel("Variant")
        plt.ylabel(f"Mean SASA after {EQUILIBRATION_CUTOFF_NS} ns (nm²)")
        save_plot(f"03_{comparison_group}_mean_sasa_bar.png")


def plot_all_mean_sasa_comparisons_one_figure(replicate_summary: pd.DataFrame) -> None:
    comparison_order = [
        "pro_mon_protein",
        "mem_mon_protein",
        "pro_dim_protein",
        "mem_dim_protein",
    ]

    plot_df = replicate_summary[replicate_summary["selection"] == "protein"].copy()
    plot_df["comparison_group"] = pd.Categorical(
        plot_df["comparison_group"],
        categories=comparison_order,
        ordered=True,
    )
    plot_df = plot_df.dropna(subset=["comparison_group"]).sort_values("comparison_group")

    if plot_df.empty:
        return

    plt.figure(figsize=(16, 10))

    sns.barplot(
        data=plot_df,
        x="comparison_group",
        y="mean_sasa_nm2",
        hue="variant",
        errorbar="sd",
        capsize=0.15,
        hue_order=["wt", "v8a"],
        palette={"wt": dark_blue, "v8a": dark_pink},
        err_kws={"color": "black", "linewidth": 2},
    )

    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(
        handles=handles,
        labels=["WT", "V8A"],
        loc="upper left",
        title=None
    )
    plt.ylim(45, 95)
    plt.xlabel("Comparison group")
    plt.ylabel(f"Mean SASA (nm²)")

    custom_labels = [
        "Soluble Monomer",
        "Membrane Monomer",
        "Soluble Dimer",
        "Membrane Dimer",
    ]
    plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=25, ha="right")
    save_plot("04_all_groups_wt_vs_v8a_mean_sasa.png")





def make_all_plots(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    residue_summary: pd.DataFrame,
    residue_differences: pd.DataFrame,
) -> None:
    plot_wt_vs_v8a_replicate_pairs_full_protein(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_sasa_bars(replicate_summary)
    plot_all_mean_sasa_comparisons_one_figure(replicate_summary)


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    make_dirs()

    print(f"Reading SASA XVG files from: {INPUT_DIR}")
    df = load_all_time_data()
    residue_df = load_all_residue_data()

    print(f"Loaded {df['source_file'].nunique()} SASA time-series files")
    print(f"Total time-series data points: {len(df):,}")

    if not residue_df.empty:
        print(f"Loaded {residue_df['source_file'].nunique()} residue SASA files")
        print(f"Total residue rows: {len(residue_df):,}")

    replicate_summary = calculate_replicate_summary(df)
    condition_summary = calculate_wt_vs_v8a_summary(replicate_summary)
    differences = calculate_v8a_minus_wt_differences(replicate_summary)
    t_tests = calculate_t_tests(replicate_summary)

    residue_summary = calculate_residue_condition_summary(residue_df)
    residue_differences = calculate_residue_v8a_minus_wt(residue_summary)

    save_tables(
        df,
        residue_df,
        replicate_summary,
        condition_summary,
        differences,
        t_tests,
        residue_summary,
        residue_differences,
    )
    make_all_plots(df, replicate_summary, residue_summary, residue_differences)

    print("\nDone!")
    print(f"Graphs saved to: {GRAPH_DIR}")
    print(f"Tables saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
