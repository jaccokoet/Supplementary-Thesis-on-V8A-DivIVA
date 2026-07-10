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

CONVERT_PS_TO_NS = True
EQUILIBRATION_CUTOFF_NS = 50

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 20,
    "axes.titlesize": 35,
    "axes.labelsize": 35,
    "xtick.labelsize": 20,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 20,
})


# -----------------------------
# Custom colors
# -----------------------------
light_pink = "#ff98cb"
dark_pink = "#ec3499"
light_blue = "#65a9ed"
dark_blue = "#1546c7"

VARIANT_COLORS = {
    "wt": dark_blue,
    "v8a": dark_pink,
}

VARIANT_FILL_COLORS = {
    "wt": light_blue,
    "v8a": light_pink,
}


# -----------------------------
# Reading and parsing
# -----------------------------
def make_dirs() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_filename(filename: str) -> dict:
    """
    Example:
        mem_dim_v8a_repl1_helix_a_cc_a_contacts.xvg
        mem_dim_v8a_repl1_helix_b_cc_b_contacts.xvg
    """
    stem = filename.replace("_contacts.xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem|pro)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)_"
        r"helix_(?P<helix>a|b)_cc_(?P<selection>a|b)$"
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
                number_contacts = float(parts[1])
                rows.append((time, number_contacts))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric contact data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "number_contacts"])
    df["time_ns"] = df["time_raw"] / 1000 if CONVERT_PS_TO_NS else df["time_raw"]

    return df


def load_all_data() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*_contacts.xvg"))

    if not files:
        raise FileNotFoundError(f"No *_contacts.xvg files found in {INPUT_DIR}")

    all_frames = []

    for filepath in files:
        meta = parse_filename(filepath.name)
        df = read_xvg(filepath)

        for key, value in meta.items():
            df[key] = value

        df["source_file"] = filepath.name
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    combined["replicate_number"] = (
        combined["replicate"]
        .str.replace("repl", "", regex=False)
        .astype(int)
    )

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
            n_points=("number_contacts", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_number_contacts=("number_contacts", "mean"),
            median_number_contacts=("number_contacts", "median"),
            std_number_contacts=("number_contacts", "std"),
            min_number_contacts=("number_contacts", "min"),
            max_number_contacts=("number_contacts", "max"),
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
            mean_number_contacts=("mean_number_contacts", "mean"),
            sd_mean_number_contacts=("mean_number_contacts", "std"),
            mean_max_number_contacts=("max_number_contacts", "mean"),
            sd_max_number_contacts=("max_number_contacts", "std"),
        )
        .sort_values(["system", "selection", "variant"])
    )

    return condition_summary


def calculate_wt_vs_v8a_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = replicate_summary.pivot_table(
        index=["system", "selection", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_number_contacts", "max_number_contacts"],
    )

    rows = []

    for index, row in pivot.iterrows():
        system, selection, comparison_group, replicate = index

        try:
            rows.append({
                "system": system,
                "selection": selection,
                "comparison_group": comparison_group,
                "replicate": replicate,
                "mean_contacts_difference_v8a_minus_wt":
                    row[("mean_number_contacts", "v8a")]
                    - row[("mean_number_contacts", "wt")],
                "max_contacts_difference_v8a_minus_wt":
                    row[("max_number_contacts", "v8a")]
                    - row[("max_number_contacts", "wt")],
            })
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
            wt["mean_number_contacts"],
            v8a["mean_number_contacts"],
            equal_var=False,
        )

        rows.append({
            "comparison_group": comparison_group,
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_number_contacts": wt["mean_number_contacts"].mean(),
            "wt_sd_number_contacts": wt["mean_number_contacts"].std(),
            "v8a_mean_number_contacts": v8a["mean_number_contacts"].mean(),
            "v8a_sd_number_contacts": v8a["mean_number_contacts"].std(),
            "mean_contacts_t_statistic": mean_t,
            "mean_contacts_p_value": mean_p,
            "mean_contacts_significant_p_lt_0_05": mean_p < 0.05,
        })

    return pd.DataFrame(rows)


def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_contacts_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "contacts_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_contacts_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_contacts_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_contacts_t_tests.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_wt_vs_v8a_replicate_pairs(df: pd.DataFrame) -> None:
    for (comparison_group, replicate), sub in df.groupby(["comparison_group", "replicate"]):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["number_contacts"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel("Number of contacts")
        plt.xlim(0, 300)
        plt.ylim(0, 120)


        legend = plt.legend(title=replicate.upper(), loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{comparison_group}_{replicate}_wt_vs_v8a_contacts.png")


def plot_wt_vs_v8a_mean_sd(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(
            mean_number_contacts=("number_contacts", "mean"),
            sd_number_contacts=("number_contacts", "std"),
        )
    )

    grouped["lower"] = grouped["mean_number_contacts"] - grouped["sd_number_contacts"].fillna(0)
    grouped["upper"] = grouped["mean_number_contacts"] + grouped["sd_number_contacts"].fillna(0)

    for comparison_group, sub in grouped.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_number_contacts"],
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
        plt.ylabel("Number of contacts")
        plt.xlim(0, 300)
        plt.ylim(0, 120)


        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"02_{comparison_group}_wt_vs_v8a_mean_sd_contacts.png")


def plot_wt_vs_v8a_mean_contacts_bars(replicate_summary: pd.DataFrame) -> None:
    for comparison_group, sub in replicate_summary.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="mean_number_contacts",
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
            y="mean_number_contacts",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        plt.xlabel("Variant")
        plt.ylabel(f"Mean contacts after {EQUILIBRATION_CUTOFF_NS} ns")

        save_plot(f"03_{comparison_group}_mean_contacts_bar.png")


def plot_all_mean_contacts_comparisons_one_figure(
    replicate_summary: pd.DataFrame,
) -> None:
    plt.figure(figsize=(18, 10))
    plt.ylim(0, 120)

    comparison_order = [
        "pro_mon_a",
        "mem_mon_a",
        "pro_dim_a",
        "pro_dim_b",
        "mem_dim_a",
        "mem_dim_b",
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
        y="mean_number_contacts",
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

    plt.xlabel("Comparison group")
    plt.ylabel("Mean number of contacts")

    custom_labels = [
        "Soluble Monomer",
        "Membrane Monomer",
        "Soluble Dimer A",
        "Soluble Dimer B",
        "Membrane Dimer A",
        "Membrane Dimer B",
    ]

    plt.xticks(
        ticks=range(len(custom_labels)),
        labels=custom_labels,
        rotation=25,
        ha="right",
    )

    save_plot("04_all_groups_wt_vs_v8a_mean_contacts.png")


def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs(df)
    plot_wt_vs_v8a_mean_sd(df)
    plot_wt_vs_v8a_mean_contacts_bars(replicate_summary)
    plot_all_mean_contacts_comparisons_one_figure(replicate_summary)

# -----------------------------
# Extra contact analyses
# -----------------------------

LOW_CONTACT_CUTOFF = 60
ROLLING_WINDOW_NS = 5


def add_contact_loss_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - contacts_relative_to_start
    - contact_loss_from_start
    - rolling_mean_contacts
    """

    df = df.copy()

    start_contacts = (
        df.sort_values("time_ns")
        .groupby(["source_file"])["number_contacts"]
        .first()
        .rename("start_contacts")
    )

    df = df.merge(start_contacts, on="source_file", how="left")

    df["contacts_relative_to_start"] = (
        df["number_contacts"] / df["start_contacts"]
    )

    df["contact_loss_from_start"] = (
        df["start_contacts"] - df["number_contacts"]
    )

    # Estimate how many frames fit into chosen rolling window
    time_step = (
        df.sort_values("time_ns")
        .groupby("source_file")["time_ns"]
        .diff()
        .median()
    )

    rolling_window_frames = max(1, int(ROLLING_WINDOW_NS / time_step))

    df["rolling_mean_contacts"] = (
        df.sort_values("time_ns")
        .groupby("source_file")["number_contacts"]
        .transform(
            lambda x: x.rolling(
                window=rolling_window_frames,
                center=True,
                min_periods=1,
            ).mean()
        )
    )

    return df


def calculate_low_contact_fraction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates how often each replicate drops below the low-contact cutoff.
    """

    stats_df = df[df["time_ns"] >= EQUILIBRATION_CUTOFF_NS].copy()

    summary = (
        stats_df.groupby(
            [
                "system",
                "selection",
                "comparison_group",
                "variant",
                "replicate",
                "condition",
            ],
            as_index=False,
        )
        .agg(
            total_frames=("number_contacts", "size"),
            low_contact_frames=(
                "number_contacts",
                lambda x: (x < LOW_CONTACT_CUTOFF).sum(),
            ),
            fraction_low_contacts=(
                "number_contacts",
                lambda x: (x < LOW_CONTACT_CUTOFF).mean(),
            ),
            mean_contact_loss_from_start=(
                "contact_loss_from_start",
                "mean",
            ),
            max_contact_loss_from_start=(
                "contact_loss_from_start",
                "max",
            ),
        )
    )

    return summary


def plot_contact_distribution(df: pd.DataFrame) -> None:
    """
    Shows whether WT and V8A have different contact-count distributions.
    """

    plot_df = df[df["time_ns"] >= EQUILIBRATION_CUTOFF_NS].copy()

    for comparison_group, sub in plot_df.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(12, 8))

        sns.kdeplot(
            data=sub,
            x="number_contacts",
            hue="variant",
            hue_order=["wt", "v8a"],
            fill=True,
            common_norm=False,
            alpha=0.25,
            palette={"wt": dark_blue, "v8a": dark_pink},
        )
        sns.move_legend(
            plt.gca(),
            "upper left",
            title=None,
            labels=["WT", "V8A"],
        )

        plt.xlabel("Number of contacts")
        plt.ylabel("Density")
        plt.ylim(0, 0.1)
        plt.xlim(0, 120)

        save_plot(f"05_{comparison_group}_contact_distribution.png")


def plot_contact_loss_over_time(df: pd.DataFrame) -> None:
    """
    Shows loss of contacts compared with the first frame.
    Higher values mean more contacts were lost.
    """

    grouped = (
        df.groupby(["comparison_group", "variant", "time_ns"], as_index=False)
        .agg(
            mean_contact_loss=("contact_loss_from_start", "mean"),
            sd_contact_loss=("contact_loss_from_start", "std"),
        )
    )

    grouped["lower"] = grouped["mean_contact_loss"] - grouped["sd_contact_loss"].fillna(0)
    grouped["upper"] = grouped["mean_contact_loss"] + grouped["sd_contact_loss"].fillna(0)

    for comparison_group, sub in grouped.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_contact_loss"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1.5,
            )

            plt.fill_between(
                variant_df["time_ns"],
                variant_df["lower"],
                variant_df["upper"],
                color=VARIANT_FILL_COLORS[variant],
                alpha=0.20,
            )

        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Time (ns)")
        plt.ylabel("Contact loss from start")
        plt.xlim(0, 300)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"06_{comparison_group}_contact_loss_over_time.png")


def plot_low_contact_fraction_bars(low_contact_summary: pd.DataFrame) -> None:
    """
    Shows how often each system drops below the chosen contact cutoff.
    """

    for comparison_group, sub in low_contact_summary.groupby("comparison_group"):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(8, 8))

        sns.barplot(
            data=sub,
            x="variant",
            y="fraction_low_contacts",
            order=["wt", "v8a"],
            errorbar="sd",
            capsize=0.15,
            palette=[dark_blue, dark_pink],
            err_kws={
                "color": "black",
                "linewidth": 2,
            },
        )

        sns.stripplot(
            data=sub,
            x="variant",
            y="fraction_low_contacts",
            order=["wt", "v8a"],
            color="black",
            alpha=1,
            size=7,
        )

        plt.xlabel("Variant")
        plt.ylabel(f"Fraction of frames with < {LOW_CONTACT_CUTOFF} contacts")
        plt.ylim(0, 1)

        save_plot(f"07_{comparison_group}_low_contact_fraction.png")


def plot_rolling_contacts(df: pd.DataFrame) -> None:
    """
    Smoothed contact plot, useful when the raw trace is noisy.
    """

    for (comparison_group, replicate), sub in df.groupby(["comparison_group", "replicate"]):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["rolling_mean_contacts"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=2,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel(f"Rolling mean contacts ({ROLLING_WINDOW_NS} ns window)")
        plt.xlim(0, 300)

        legend = plt.legend(title=replicate.upper(), loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"08_{comparison_group}_{replicate}_rolling_contacts.png")


def make_extra_contact_analyses(df: pd.DataFrame) -> pd.DataFrame:
    df = add_contact_loss_columns(df)

    low_contact_summary = calculate_low_contact_fraction(df)

    low_contact_summary.to_csv(
        OUTPUT_DIR / "contacts_low_contact_fraction_summary.csv",
        index=False,
    )

    df.to_csv(
        OUTPUT_DIR / "all_contacts_with_contact_loss_columns.csv",
        index=False,
    )

    plot_contact_distribution(df)
    plot_contact_loss_over_time(df)
    plot_low_contact_fraction_bars(low_contact_summary)
    plot_rolling_contacts(df)

    return df
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
    differences = calculate_wt_vs_v8a_differences(replicate_summary)
    t_tests = calculate_t_tests(replicate_summary)

    save_tables(
        df,
        replicate_summary,
        condition_summary,
        differences,
        t_tests,
    )

    make_all_plots(df, replicate_summary)
    df = make_extra_contact_analyses(df)

    print("\nDone!")


if __name__ == "__main__":
    main()