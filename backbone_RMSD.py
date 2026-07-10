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

# GROMACS .xvg RMSD files often use ps on the x-axis.
# If your x-axis is already ns, change this to False.
CONVERT_PS_TO_NS = False

# Optional: ignore early simulation time for summary statistics.
# Example: set to 50 if you only want stats after 50 ns.
EQUILIBRATION_CUTOFF_NS = 50

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


# WT = blue palette
# V8A = pink palette
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
        mem_dim_v8a_repl1_chain_a_backbone_rmsd.xvg
        pro_dim_wt_repl2_protein_backbone_rmsd.xvg
        pro_mon_v8a_repl3_protein_backbone_rmsd.xvg
    """
    stem = filename.replace("_backbone_rmsd.xvg", "")

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


def read_xvg(filepath: Path) -> pd.DataFrame:
    """
    Read a GROMACS .xvg file while skipping metadata lines starting with # or @.
    Uses the first two numeric columns as time and RMSD.
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
                rmsd = float(parts[1])
                rows.append((time, rmsd))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric RMSD data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "rmsd_nm"])
    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]
    return df


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_backbone_rmsd.xvg"))

    if not files:
        raise FileNotFoundError(f"No .xvg files found in {INPUT_DIR}")

    all_frames = []

    for filepath in files:
        meta = parse_filename(filepath.name)
        df = read_xvg(filepath)

        for key, value in meta.items():
            df[key] = value

        df["source_file"] = filepath.name
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined["replicate_number"] = combined["replicate"].str.replace("repl", "", regex=False).astype(int)

    combined = combined.sort_values(
        ["system", "selection", "variant", "replicate_number", "time_ns"]
    ).reset_index(drop=True)

    return combined


# -----------------------------
# Summaries
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
            n_points=("rmsd_nm", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_rmsd_nm=("rmsd_nm", "mean"),
            median_rmsd_nm=("rmsd_nm", "median"),
            std_rmsd_nm=("rmsd_nm", "std"),
            min_rmsd_nm=("rmsd_nm", "min"),
            max_rmsd_nm=("rmsd_nm", "max"),
        )
        .sort_values(["system", "selection", "variant", "replicate"])
    )

    return summary


def calculate_wt_vs_v8a_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Makes one row per comparison group and variant.
    Example groups:
        pro_mon_protein
        pro_dim_chain_a
        mem_dim_chain_b
    """
    condition_summary = (
        replicate_summary.groupby(
            ["system", "selection", "comparison_group", "variant"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_rmsd_nm=("mean_rmsd_nm", "mean"),
            sd_mean_rmsd_nm=("mean_rmsd_nm", "std"),
            mean_max_rmsd_nm=("max_rmsd_nm", "mean"),
            sd_max_rmsd_nm=("max_rmsd_nm", "std"),
        )
        .sort_values(["system", "selection", "variant"])
    )

    return condition_summary


def calculate_wt_minus_v8a_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates V8A - WT differences for matching groups.
    Positive difference means V8A has higher RMSD than WT.
    """
    pivot = replicate_summary.pivot_table(
        index=["system", "selection", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_rmsd_nm", "max_rmsd_nm"],
    )

    rows = []
    for index, row in pivot.iterrows():
        system, selection, comparison_group, replicate = index

        if ("mean_rmsd_nm", "wt") not in row or ("mean_rmsd_nm", "v8a") not in row:
            continue

        try:
            rows.append(
                {
                    "system": system,
                    "selection": selection,
                    "comparison_group": comparison_group,
                    "replicate": replicate,
                    "mean_rmsd_difference_v8a_minus_wt_nm": row[("mean_rmsd_nm", "v8a")] - row[("mean_rmsd_nm", "wt")],
                    "max_rmsd_difference_v8a_minus_wt_nm": row[("max_rmsd_nm", "v8a")] - row[("max_rmsd_nm", "wt")],
                }
            )
        except KeyError:
            continue

    return pd.DataFrame(rows)


def calculate_t_tests(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        wt = sub[sub["variant"] == "wt"]
        v8a = sub[sub["variant"] == "v8a"]

        if wt.empty or v8a.empty:
            continue

        mean_t, mean_p = stats.ttest_ind(
            wt["mean_rmsd_nm"],
            v8a["mean_rmsd_nm"],
            equal_var=False,
        )


        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_rmsd_nm": wt["mean_rmsd_nm"].mean(),
            "wt_sd_rmsd_nm": wt["mean_rmsd_nm"].std(),
            "v8a_mean_rmsd_nm": v8a["mean_rmsd_nm"].mean(),
            "v8a_sd_rmsd_nm": v8a["mean_rmsd_nm"].std(),
            "mean_rmsd_t_statistic": mean_t,
            "mean_rmsd_p_value": mean_p,
            "mean_rmsd_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)

def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_backbone_rmsd_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "rmsd_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_t_tests.csv", index=False)

# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


TITLE_MAP = {
    "pro_mon_protein": "Backbone RMSD of WT and V8A DivIVA Monomers",
    "pro_dim_protein": "Backbone RMSD of WT and V8A DivIVA Dimers",
    "pro_dim_chain_a": "Backbone RMSD of WT and V8A DivIVA Dimers (Chain A)",
    "pro_dim_chain_b": "Backbone RMSD of WT and V8A DivIVA Dimers (Chain B)",
    "mem_mon_protein": "Backbone RMSD of Membrane_Associated WT and V8A DIvIVA Monomers",
    "mem_dim_protein": "Backbone RMSD of Membrane-Associated WT and V8A DivIVA Dimers",
    "mem_dim_chain_a": "Backbone RMSD of Membrane-Associated WT and V8A DivIVA Dimers (Chain A)",
    "mem_dim_chain_b": "Backbone RMSD of Membrane-Associated WT and V8A DivIVA Dimers (Chain B)",
}


def pretty_title(text: str) -> str:
    return TITLE_MAP.get(text, text.replace("_", " ").upper())

def plot_wt_vs_v8a_replicate_pairs_full_protein(df: pd.DataFrame) -> None:
    """
    Makes WT vs V8A RMSD plots per replicate, only for full protein selections.
    Expected output:
        4 systems × 3 replicates = 12 graphs
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
                variant_df["rmsd_nm"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("Backbone RMSD (nm)")
        plt.xlim(0, 300)
        plt.ylim(0, 1.6)

        legend = plt.legend(title=f"{replicate.upper()}", loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{system}_{replicate}_wt_vs_v8a_full_protein.png")

def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:
    """
    For each matching group, plot the replicate average ± SD for WT and V8A.
    """
    grouped = (
        df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(mean_rmsd_nm=("rmsd_nm", "mean"), sd_rmsd_nm=("rmsd_nm", "std"))
    )

    grouped["lower"] = grouped["mean_rmsd_nm"] - grouped["sd_rmsd_nm"].fillna(0)
    grouped["upper"] = grouped["mean_rmsd_nm"] + grouped["sd_rmsd_nm"].fillna(0)

    for comparison_group, sub in grouped.groupby("comparison_group"):
        variants = set(sub["variant"])
        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant]

            if variant_df.empty:
                continue

            variant_df = variant_df.sort_values("time_ns")
            label = variant.upper()

            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_rmsd_nm"],
                label=label,
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

        #plt.title(f"{pretty_title(comparison_group)}")
        plt.xlabel("Time (ns)")
        plt.ylabel("Backbone RMSD (nm)")
        plt.xlim(0, 300)
        plt.ylim(0,1.6)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"02_{comparison_group}_wt_vs_v8a_mean_sd.png")


def plot_wt_vs_v8a_mean_rmsd_bars(replicate_summary: pd.DataFrame) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        variants = set(sub["variant"])
        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_rmsd_nm",
            errorbar="sd",
            capsize=0.15,
            order=["wt", "v8a"],
            palette=[dark_blue, dark_pink],
            err_kws={
                "color": "black",
                "linewidth": 2,
            },
        )
        sns.stripplot(
            data=sub,
            x="variant",
            y="mean_rmsd_nm",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        #plt.title(f"{pretty_title(comparison_group)}")
        plt.xlabel("Variant")
        plt.ylim(0,1.25)
        plt.ylabel(f"Mean backbone RMSD after {EQUILIBRATION_CUTOFF_NS} ns (nm)")
        save_plot(f"03_{comparison_group}_mean_rmsd_bar.png")


def plot_all_mean_rmsd_comparisons_one_figure(replicate_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(16, 10))

    comparison_order = [
        "pro_mon_protein",
        "mem_mon_protein",
        "pro_dim_protein",
        "mem_dim_protein",
    ]

    plot_df = replicate_summary.copy()
    plot_df["comparison_group"] = pd.Categorical(
        plot_df["comparison_group"],
        categories=comparison_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values("comparison_group")

    sns.barplot(
        data=plot_df,
        x="comparison_group",
        y="mean_rmsd_nm",
        hue="variant",
        errorbar="sd",
        capsize=0.15,
        hue_order=["wt", "v8a"],
        palette={"wt": dark_blue, "v8a": dark_pink},
        err_kws={
            "color": "black",
            "linewidth": 2,
        },
    )

    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(
        handles=handles,
        labels=["WT", "V8A"],
        loc="upper left",
        title=None
    )

#    for line in legend.get_lines():
#        line.set_linewidth(4)
    #plt.title("WT vs V8A mean backbone RMSD across all groups")
    plt.xlabel("Comparison group")
    plt.ylim(0, 1.25)
    plt.ylabel(f"Mean backbone RMSD (nm)")
    custom_labels = [
        "Soluble Monomer",
        "Membrane Monomer",
        "Soluble Dimer",
        "Membrane Dimer",
    ]

    plt.xticks(ticks=range(len(custom_labels)), labels=custom_labels, rotation=25, ha="right")
    save_plot("04_all_groups_wt_vs_v8a_mean_rmsd.png")



def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs_full_protein(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_rmsd_bars(replicate_summary)
    plot_all_mean_rmsd_comparisons_one_figure(replicate_summary)


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    make_dirs()

    print(f"Reading XVG files from: {INPUT_DIR}")
    df = load_all_data()

    print(f"Loaded {df['source_file'].nunique()} files")
    print(f"Total data points: {len(df):,}")

    replicate_summary = calculate_replicate_summary(df)
    condition_summary = calculate_wt_vs_v8a_summary(replicate_summary)
    differences = calculate_wt_minus_v8a_differences(replicate_summary)
    t_tests = calculate_t_tests(replicate_summary)

    save_tables(df, replicate_summary, condition_summary, differences, t_tests)
    make_all_plots(df, replicate_summary)

    print("\nDone!")

if __name__ == "__main__":
    main()