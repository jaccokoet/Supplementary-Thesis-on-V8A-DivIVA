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

# GROMACS gyrate .xvg files usually use ps on the x-axis.
# If your x-axis is already ns, keep this as False.
CONVERT_PS_TO_NS = True

# Optional: ignore early simulation time for summary statistics.
# Example: set to 100 if you only want stats after 100 ns.
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
        mem_dim_v8a_repl1_chain_a_rg.xvg
        pro_dim_wt_repl2_protein_rg.xvg
        pro_mon_v8a_repl3_protein_rg.xvg

    This also supports your future mem_mon and mem_dim files as long as they
    use the same naming pattern.
    """
    stem = filename.replace("_rg.xvg", "")

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
    Uses the first two numeric columns as time and radius of gyration.
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
                rg = float(parts[1])
                rows.append((time, rg))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric radius of gyration data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "rg_nm"])
    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]
    return df


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_rg.xvg"))

    if not files:
        raise FileNotFoundError(f"No *_rg.xvg files found in {INPUT_DIR}")

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
            n_points=("rg_nm", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_rg_nm=("rg_nm", "mean"),
            median_rg_nm=("rg_nm", "median"),
            std_rg_nm=("rg_nm", "std"),
            min_rg_nm=("rg_nm", "min"),
            max_rg_nm=("rg_nm", "max"),
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
            mean_rg_nm=("mean_rg_nm", "mean"),
            sd_mean_rg_nm=("mean_rg_nm", "std"),
            mean_max_rg_nm=("max_rg_nm", "mean"),
            sd_max_rg_nm=("max_rg_nm", "std"),
        )
        .sort_values(["system", "selection", "variant"])
    )

    return condition_summary


def calculate_wt_minus_v8a_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates V8A - WT differences for matching groups and replicates.
    Positive difference means V8A has a higher radius of gyration than WT.
    """
    pivot = replicate_summary.pivot_table(
        index=["system", "selection", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_rg_nm", "max_rg_nm"],
    )

    rows = []
    for index, row in pivot.iterrows():
        system, selection, comparison_group, replicate = index

        try:
            wt_mean = row[("mean_rg_nm", "wt")]
            v8a_mean = row[("mean_rg_nm", "v8a")]
            wt_max = row[("max_rg_nm", "wt")]
            v8a_max = row[("max_rg_nm", "v8a")]
        except KeyError:
            continue

        if pd.isna(wt_mean) or pd.isna(v8a_mean):
            continue

        rows.append(
            {
                "system": system,
                "selection": selection,
                "comparison_group": comparison_group,
                "replicate": replicate,
                "mean_rg_difference_v8a_minus_wt_nm": v8a_mean - wt_mean,
                "max_rg_difference_v8a_minus_wt_nm": v8a_max - wt_max,
            }
        )

    return pd.DataFrame(rows)


def calculate_t_tests(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        wt = sub[sub["variant"] == "wt"]
        v8a = sub[sub["variant"] == "v8a"]

        if wt.empty or v8a.empty:
            continue

        mean_t, mean_p = stats.ttest_ind(
            wt["mean_rg_nm"],
            v8a["mean_rg_nm"],
            equal_var=False,
        )


        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),

            "wt_mean_rg_nm": wt["mean_rg_nm"].mean(),
            "wt_sd_rg_nm": wt["mean_rg_nm"].std(),

            "v8a_mean_rg_nm": v8a["mean_rg_nm"].mean(),
            "v8a_sd_rg_nm": v8a["mean_rg_nm"].std(),

            "mean_rg_t_statistic": mean_t,
            "mean_rg_p_value": mean_p,
            "mean_rg_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)


def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_radius_of_gyration_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "rg_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_rg_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_rg_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_rg_t_tests.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


TITLE_MAP = {
    "pro_mon_protein": "Radius of Gyration of WT and V8A DivIVA Monomers",
    "pro_dim_protein": "Radius of Gyration of WT and V8A DivIVA Dimers",
    "pro_dim_chain_a": "Radius of Gyration of WT and V8A DivIVA Dimers (Chain A)",
    "pro_dim_chain_b": "Radius of Gyration of WT and V8A DivIVA Dimers (Chain B)",
    "mem_mon_protein": "Radius of Gyration of Membrane-Associated WT and V8A DivIVA Monomers",
    "mem_dim_protein": "Radius of Gyration of Membrane-Associated WT and V8A DivIVA Dimers",
    "mem_dim_chain_a": "Radius of Gyration of Membrane-Associated WT and V8A DivIVA Dimers (Chain A)",
    "mem_dim_chain_b": "Radius of Gyration of Membrane-Associated WT and V8A DivIVA Dimers (Chain B)",
}


def pretty_title(text: str) -> str:
    return TITLE_MAP.get(text, text.replace("_", " ").upper())


def plot_wt_vs_v8a_replicate_pairs_full_protein(df: pd.DataFrame) -> None:
    """
    Makes WT vs V8A Rg plots per replicate, only for full protein selections.
    Expected output once all systems are present:
        4 systems x 3 replicates = 12 graphs
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
                variant_df["rg_nm"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("Radius of gyration (nm)")
        plt.xlim(0, 300)
        # Adjust this if your Rg values fall outside this range.
        plt.ylim(1.2, 2.2)

        legend = plt.legend(title=f"{replicate.upper()}", loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{system}_{replicate}_wt_vs_v8a_full_protein_rg.png")


def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:
    """
    For each matching group, plot the replicate average +/- SD for WT and V8A.
    """
    grouped = (
        df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(mean_rg_nm=("rg_nm", "mean"), sd_rg_nm=("rg_nm", "std"))
    )

    grouped["lower"] = grouped["mean_rg_nm"] - grouped["sd_rg_nm"].fillna(0)
    grouped["upper"] = grouped["mean_rg_nm"] + grouped["sd_rg_nm"].fillna(0)

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
                variant_df["mean_rg_nm"],
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

        # plt.title(f"{pretty_title(comparison_group)}")
        plt.xlabel("Time (ns)")
        plt.ylabel("Radius of gyration (nm)")
        plt.xlim(0, 300)
        # Adjust this if your Rg values fall outside this range.
        plt.ylim(1.2, 2.2)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"02_{comparison_group}_wt_vs_v8a_mean_sd_rg.png")


def plot_wt_vs_v8a_mean_rg_bars(replicate_summary: pd.DataFrame) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        variants = set(sub["variant"])
        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_rg_nm",
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
            y="mean_rg_nm",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        # plt.title(f"{pretty_title(comparison_group)}")
        plt.xlabel("Variant")
        plt.ylabel(f"Mean radius of gyration after {EQUILIBRATION_CUTOFF_NS} ns (nm)")
        save_plot(f"03_{comparison_group}_mean_rg_bar.png")


def plot_all_mean_rg_comparisons_one_figure(replicate_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(16, 10))

    comparison_order = [
        "pro_mon_protein",
        "mem_mon_protein",
        "pro_dim_protein",
        "mem_dim_protein",
    ]

    plot_df = replicate_summary.copy()
    plot_df = plot_df[plot_df["comparison_group"].isin(comparison_order)]
    plot_df["comparison_group"] = pd.Categorical(
        plot_df["comparison_group"],
        categories=comparison_order,
        ordered=True,
    )

    plot_df = plot_df.sort_values("comparison_group")

    if plot_df.empty:
        return

    sns.barplot(
        data=plot_df,
        x="comparison_group",
        y="mean_rg_nm",
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
    plt.ylim(1.2, 2)
    legend.set_title(None)
    # plt.title("WT vs V8A mean radius of gyration across all groups")
    plt.xlabel("Comparison group")
    plt.ylabel(f"Mean radius of gyration (nm)")
    custom_labels = [
        "Soluble Monomer",
        "Membrane Monomer",
        "Soluble Dimer",
        "Membrane Dimer",
    ]

    available_labels = [
        label for group, label in zip(comparison_order, custom_labels)
        if group in set(plot_df["comparison_group"].dropna().astype(str))
    ]
    plt.xticks(ticks=range(len(available_labels)), labels=available_labels, rotation=25, ha="right")
    save_plot("04_all_groups_wt_vs_v8a_mean_rg.png")


def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs_full_protein(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_rg_bars(replicate_summary)
    plot_all_mean_rg_comparisons_one_figure(replicate_summary)


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
