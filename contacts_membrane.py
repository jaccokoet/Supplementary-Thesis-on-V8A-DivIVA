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

# GROMACS gmx mindist output is usually in ps.
# Your example file has Time (ps), so this should usually be True.
CONVERT_PS_TO_NS = True

# Optional: ignore early simulation time for summary statistics.
# Example: set to 50 if you only want stats after 50 ns.
EQUILIBRATION_CUTOFF_NS = 50

# gmx mindist contact cutoff used in your file title: Number of Contacts < 0.6 nm
CONTACT_CUTOFF_NM = 0.6

# For final analysis, set to False if you want missing replicate warnings only.
REQUIRE_COMPLETE_THREE_REPLICATES = False
EXPECTED_REPLICATES = {"repl1", "repl2", "repl3"}

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
# Custom colors: same style as your RMSD script
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
# Lipid counts in the complete bilayer
# -----------------------------
# You gave upper/lower leaflet counts. For normalization we use the total bilayer count.
# TMCL1 and TMCL2 are combined as TMCL because your filenames use TMCL.
LIPID_COUNTS = {
    "DMPA": 16 + 16,
    "POPE": 28 + 28,
    "POPI": 16 + 16,
    "TMCL": (40 + 40) + (40 + 40),
    "DPGL": 48 + 48,
    "POOT": 72 + 72,      # filenames use POOT, your composition says POOTG
    "POOTG": 72 + 72,     # accepted as alias if filenames ever use POOTG
    "PAL": 32 + 32,
    "LPSA": 16 + 16,
}

LIPID_ORDER = ["DMPA", "POPI", "POPE", "TMCL", "DPGL", "PAL", "POOT", "LPSA"]
SYSTEM_ORDER = ["pro_mon", "pro_dim", "mem_mon", "mem_dim"]

SYSTEM_LABELS = {
    "pro_mon": "Soluble Monomer",
    "pro_dim": "Soluble Dimer",
    "mem_mon": "Membrane Monomer",
    "mem_dim": "Membrane Dimer",
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
        mem_dim_v8a_repl1_mbd_DMPA_contacts.xvg
        mem_dim_v8a_repl1_mbd_DMPA_mindist.xvg
        pro_mon_wt_repl2_mbd_POPE_contacts.xvg

    The script only analyses *_contacts.xvg by default.
    *_mindist.xvg files are ignored unless you change the file glob in load_all_data().
    """
    stem = filename.replace(".xvg", "")

    pattern = re.compile(
        r"^(?P<environment>mem|pro)_"
        r"(?P<oligomer>mon|dim)_"
        r"(?P<variant>wt|v8a)_"
        r"(?P<replicate>repl\d+)_"
        r"(?P<selection>mbd|protein|chain_a|chain_b)_"
        r"(?P<lipid>[A-Za-z0-9]+)_"
        r"(?P<analysis_type>contacts|mindist)$"
    )

    match = pattern.match(stem)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    data = match.groupdict()
    data["lipid"] = data["lipid"].upper()

    # Harmonize POOT/POOTG naming.
    if data["lipid"] == "POOTG":
        data["lipid"] = "POOT"

    data["system"] = f"{data['environment']}_{data['oligomer']}"
    data["comparison_group"] = f"{data['system']}_{data['selection']}_{data['lipid']}"
    data["condition"] = f"{data['system']}_{data['variant']}_{data['selection']}_{data['lipid']}"
    return data


def read_xvg(filepath: Path) -> pd.DataFrame:
    """
    Read a GROMACS .xvg file while skipping metadata lines starting with # or @.
    Uses the first two numeric columns as time and number of contacts.
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
                contacts = float(parts[1])
                rows.append((time, contacts))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric contact data found in {filepath}")

    df = pd.DataFrame(rows, columns=["time_raw", "contacts"])
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
    combined["replicate_number"] = combined["replicate"].str.replace("repl", "", regex=False).astype(int)
    combined["lipid_total_in_bilayer"] = combined["lipid"].map(LIPID_COUNTS)

    missing_counts = sorted(combined.loc[combined["lipid_total_in_bilayer"].isna(), "lipid"].unique())
    if missing_counts:
        raise ValueError(
            "These lipids are present in filenames but missing from LIPID_COUNTS: "
            + ", ".join(missing_counts)
        )

    # Normalized contacts make lipid-specific enrichment comparable despite different lipid abundances.
    combined["contacts_per_lipid"] = combined["contacts"] / combined["lipid_total_in_bilayer"]

    combined = combined.sort_values(
        ["system", "selection", "lipid", "variant", "replicate_number", "time_ns"]
    ).reset_index(drop=True)

    return combined


# -----------------------------
# Summaries and statistics
# -----------------------------

def exclude_problematic_mem_dim_wt_repl1(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        ~(
            (df["system"] == "mem_dim")
            & (df["variant"] == "wt")
            & (df["replicate"] == "repl1")
        )
    ].copy()

def calculate_replicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df[df["time_ns"] >= EQUILIBRATION_CUTOFF_NS].copy()

    summary = (
        stats_df.groupby(
            [
                "system", "environment", "oligomer", "selection", "lipid",
                "comparison_group", "variant", "replicate", "condition",
                "lipid_total_in_bilayer",
            ],
            as_index=False,
        )
        .agg(
            n_points=("contacts", "size"),
            start_ns=("time_ns", "min"),
            end_ns=("time_ns", "max"),
            mean_contacts=("contacts", "mean"),
            median_contacts=("contacts", "median"),
            std_contacts=("contacts", "std"),
            min_contacts=("contacts", "min"),
            max_contacts=("contacts", "max"),
            mean_contacts_per_lipid=("contacts_per_lipid", "mean"),
            median_contacts_per_lipid=("contacts_per_lipid", "median"),
            std_contacts_per_lipid=("contacts_per_lipid", "std"),
        )
        .sort_values(["system", "selection", "lipid", "variant", "replicate"])
    )

    return summary


def calculate_wt_vs_v8a_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    condition_summary = (
        replicate_summary.groupby(
            ["system", "selection", "lipid", "comparison_group", "variant"],
            as_index=False,
        )
        .agg(
            replicate_count=("replicate", "nunique"),
            mean_contacts=("mean_contacts", "mean"),
            sd_mean_contacts=("mean_contacts", "std"),
            mean_contacts_per_lipid=("mean_contacts_per_lipid", "mean"),
            sd_mean_contacts_per_lipid=("mean_contacts_per_lipid", "std"),
            mean_max_contacts=("max_contacts", "mean"),
            sd_max_contacts=("max_contacts", "std"),
        )
        .sort_values(["system", "selection", "lipid", "variant"])
    )

    return condition_summary


def calculate_v8a_minus_wt_differences(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    pivot = replicate_summary.pivot_table(
        index=["system", "selection", "lipid", "comparison_group", "replicate"],
        columns="variant",
        values=["mean_contacts", "mean_contacts_per_lipid", "max_contacts"],
    )

    rows = []
    for index, row in pivot.iterrows():
        system, selection, lipid, comparison_group, replicate = index

        try:
            rows.append({
                "system": system,
                "selection": selection,
                "lipid": lipid,
                "comparison_group": comparison_group,
                "replicate": replicate,
                "mean_contacts_difference_v8a_minus_wt": row[("mean_contacts", "v8a")] - row[("mean_contacts", "wt")],
                "mean_contacts_per_lipid_difference_v8a_minus_wt": row[("mean_contacts_per_lipid", "v8a")] - row[("mean_contacts_per_lipid", "wt")],
                "max_contacts_difference_v8a_minus_wt": row[("max_contacts", "v8a")] - row[("max_contacts", "wt")],
            })
        except KeyError:
            continue

    return pd.DataFrame(rows)


def calculate_t_tests(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Welch t-tests per system/selection/lipid using replicate means.
    With n=3 per group this is exploratory, not strong proof.
    If one side has fewer than 2 replicates, p-values are left as NA.
    """
    summary = replicate_summary.copy()

    # Exclude problematic replicate
    summary = summary[
        ~(
            (summary["system"] == "mem_dim")
            & (summary["variant"] == "wt")
            & (summary["replicate"] == "repl1")
        )
    ].copy()

    rows = []

    for comparison_group, sub in summary.groupby("comparison_group"):


        wt = sub[sub["variant"] == "wt"]
        v8a = sub[sub["variant"] == "v8a"]

        if wt.empty or v8a.empty:
            continue

        if len(wt) >= 2 and len(v8a) >= 2:
            contact_t, contact_p = stats.ttest_ind(
                wt["mean_contacts"],
                v8a["mean_contacts"],
                equal_var=False,
            )
            norm_t, norm_p = stats.ttest_ind(
                wt["mean_contacts_per_lipid"],
                v8a["mean_contacts_per_lipid"],
                equal_var=False,
            )
        else:
            contact_t, contact_p, norm_t, norm_p = pd.NA, pd.NA, pd.NA, pd.NA

        rows.append({
            "comparison_group": comparison_group,
            "system": sub["system"].iloc[0],
            "selection": sub["selection"].iloc[0],
            "lipid": sub["lipid"].iloc[0],
            "wt_replicates": len(wt),
            "v8a_replicates": len(v8a),
            "wt_mean_contacts": wt["mean_contacts"].mean(),
            "wt_sd_contacts": wt["mean_contacts"].std(),
            "v8a_mean_contacts": v8a["mean_contacts"].mean(),
            "v8a_sd_contacts": v8a["mean_contacts"].std(),
            "v8a_minus_wt_mean_contacts": v8a["mean_contacts"].mean() - wt["mean_contacts"].mean(),
            "contacts_t_statistic": contact_t,
            "contacts_p_value": contact_p,
            "contacts_significant_p_lt_0_05": contact_p < 0.05 if pd.notna(contact_p) else pd.NA,
            "wt_mean_contacts_per_lipid": wt["mean_contacts_per_lipid"].mean(),
            "wt_sd_contacts_per_lipid": wt["mean_contacts_per_lipid"].std(),
            "v8a_mean_contacts_per_lipid": v8a["mean_contacts_per_lipid"].mean(),
            "v8a_sd_contacts_per_lipid": v8a["mean_contacts_per_lipid"].std(),
            "v8a_minus_wt_mean_contacts_per_lipid": v8a["mean_contacts_per_lipid"].mean() - wt["mean_contacts_per_lipid"].mean(),
            "contacts_per_lipid_t_statistic": norm_t,
            "contacts_per_lipid_p_value": norm_p,
            "contacts_per_lipid_significant_p_lt_0_05": norm_p < 0.05 if pd.notna(norm_p) else pd.NA,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["system", "selection", "lipid"])
    return result


def check_missing_replicates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["system", "selection", "lipid", "variant"]):
        system, selection, lipid, variant = keys
        present = set(sub["replicate"].unique())
        missing = EXPECTED_REPLICATES - present
        if missing:
            rows.append({
                "system": system,
                "selection": selection,
                "lipid": lipid,
                "variant": variant,
                "present_replicates": ",".join(sorted(present)),
                "missing_replicates": ",".join(sorted(missing)),
            })

    missing_df = pd.DataFrame(rows)
    if REQUIRE_COMPLETE_THREE_REPLICATES and not missing_df.empty:
        raise ValueError(
            "Some system/variant/lipid groups are missing replicates. "
            "Set REQUIRE_COMPLETE_THREE_REPLICATES = False to allow this."
        )
    return missing_df


def save_tables(
    df: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    condition_summary: pd.DataFrame,
    differences: pd.DataFrame,
    t_tests: pd.DataFrame,
    missing_replicates: pd.DataFrame,
) -> None:
    df.to_csv(OUTPUT_DIR / "all_lipid_contacts_long_format.csv", index=False)
    replicate_summary.to_csv(OUTPUT_DIR / "lipid_contacts_summary_by_replicate.csv", index=False)
    condition_summary.to_csv(OUTPUT_DIR / "wt_vs_v8a_lipid_contacts_summary_by_group.csv", index=False)
    differences.to_csv(OUTPUT_DIR / "wt_vs_v8a_lipid_contacts_differences_by_replicate.csv", index=False)
    t_tests.to_csv(OUTPUT_DIR / "wt_vs_v8a_lipid_contacts_t_tests.csv", index=False)
    missing_replicates.to_csv(OUTPUT_DIR / "missing_lipid_contact_replicates.csv", index=False)


# -----------------------------
# Plotting
# -----------------------------
def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def pretty_system(system: str) -> str:
    return SYSTEM_LABELS.get(system, system.replace("_", " ").title())


def ordered_lipids_present(df: pd.DataFrame) -> list[str]:
    present = list(df["lipid"].dropna().unique())
    ordered = [lipid for lipid in LIPID_ORDER if lipid in present]
    extra = sorted([lipid for lipid in present if lipid not in ordered])
    return ordered + extra

def plot_wt_vs_v8a_replicate_pairs(df: pd.DataFrame) -> None:
    plot_df = df.copy()

    for (system, selection, lipid, replicate), sub in plot_df.groupby(
        ["system", "selection", "lipid", "replicate"]
    ):
        variants = set(sub["variant"])

        if not {"wt", "v8a"}.issubset(variants):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")

            plt.plot(
                variant_df["time_ns"],
                variant_df["contacts"],
                label=variant.upper(),
                color=VARIANT_COLORS[variant],
                linewidth=1,
            )

        plt.xlabel("Time (ns)")
        plt.ylabel(f"Number of contacts < {CONTACT_CUTOFF_NM} nm")
        plt.xlim(0,300)
        plt.ylim(0,300)

        legend = plt.legend(
            title=replicate.upper(),
            loc="upper left",
        )

        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(
            f"00_{system}_{selection}_{lipid}_{replicate}_wt_vs_v8a_contacts.png"
        )

def plot_time_series_mean_sd(df: pd.DataFrame) -> None:
    """
    For each system and lipid, plot replicate average ± SD over time for WT and V8A.
    """
    plot_df = df.copy()

    # Exclude problematic membrane dimer WT replicate 1
    plot_df = plot_df[
        ~(
            (plot_df["system"] == "mem_dim")
            & (plot_df["variant"] == "wt")
            & (plot_df["replicate"] == "repl1")
        )
    ].copy()

    grouped = (
        plot_df.groupby(
            ["system", "selection", "lipid", "variant", "time_ns"],
            as_index=False,
        )
        .agg(
            mean_contacts=("contacts", "mean"),
            sd_contacts=("contacts", "std"),
        )
    )
    grouped["lower"] = grouped["mean_contacts"] - grouped["sd_contacts"].fillna(0)
    grouped["upper"] = grouped["mean_contacts"] + grouped["sd_contacts"].fillna(0)

    for (system, selection, lipid), sub in grouped.groupby(["system", "selection", "lipid"]):
        if not {"wt", "v8a"}.issubset(set(sub["variant"])):
            continue

        plt.figure(figsize=(14, 8))

        for variant in ["wt", "v8a"]:
            variant_df = sub[sub["variant"] == variant].sort_values("time_ns")
            if variant_df.empty:
                continue

            plt.plot(
                variant_df["time_ns"],
                variant_df["mean_contacts"],
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
        plt.ylabel(f"Number of contacts < {CONTACT_CUTOFF_NM} nm")
        plt.xlim(0,300)
        plt.ylim(0,300)

        legend = plt.legend(loc="upper left")
        for line in legend.get_lines():
            line.set_linewidth(4)

        save_plot(f"01_{system}_{selection}_{lipid}_wt_vs_v8a_contacts_mean_sd.png")


def plot_lipid_contact_bars_by_system(replicate_summary: pd.DataFrame, normalized: bool = False) -> None:
    """
    One bar plot per system, showing WT vs V8A for each lipid.
    Replicate means are the plotted units.
    """
    plot_df = replicate_summary.copy()

    plot_df = plot_df[
        ~(
                (plot_df["system"] == "mem_dim")
                & (plot_df["variant"] == "wt")
                & (plot_df["replicate"] == "repl1")
        )
    ].copy()

    y_col = "mean_contacts_per_lipid" if normalized else "mean_contacts"
    y_label = "Mean contacts per lipid" if normalized else "Mean number of contacts"
    suffix = "normalized_per_lipid" if normalized else "raw_contacts"

    for (system, selection), sub in plot_df.groupby(["system", "selection"]):
        if not {"wt", "v8a"}.issubset(set(sub["variant"])):
            continue

        lipid_order = ordered_lipids_present(sub)
        lipid_order = [
            "POPI",
            "DPGL",
            "DMPA",
            "POOT",
            "PAL",
            "POPE",
            "TMCL",
        ]
        plt.figure(figsize=(16, 10))

        sns.barplot(
            data=sub,
            x="lipid",
            y=y_col,
            hue="variant",
            order=lipid_order,
            hue_order=["wt", "v8a"],
            errorbar="sd",
            capsize=0.15,
            palette={"wt": dark_blue, "v8a": dark_pink},
            err_kws={
                "color": "black",
                "linewidth": 2,
            },
        )

        sns.stripplot(
            data=sub,
            x="lipid",
            y=y_col,
            hue="variant",
            order=lipid_order,
            hue_order=["wt", "v8a"],
            dodge=True,
            color="black",
            alpha=1,
            size=6,
            legend=False,
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles[:2], labels=["WT", "V8A"], loc="upper left", title=None)
        plt.xlabel("Lipid")
        plt.ylabel(f"{y_label}")
        plt.ylim(bottom=0)
        plt.xticks(rotation=25, ha="right")

        save_plot(f"02_{system}_{selection}_lipid_specific_{suffix}_bar.png")


def plot_v8a_minus_wt_heatmap(t_tests: pd.DataFrame, normalized: bool = False) -> None:
    """
    Heatmap of V8A - WT effect size per lipid and system.
    Positive values mean more contacts in V8A.
    Only dimers are shown.
    """

    plot_df = t_tests.copy()

    # Keep only dimers
    #plot_df = plot_df[
    #    plot_df["system"].isin(["pro_dim", "mem_dim"])
    #].copy()

    if plot_df.empty:
        return

    if normalized:
        value_col = "v8a_minus_wt_mean_contacts_per_lipid"
        wt_sd_col = "wt_sd_contacts_per_lipid"
        v8a_sd_col = "v8a_sd_contacts_per_lipid"
        suffix = "normalized_per_lipid"
        cbar_label = "V8A - WT contacts per lipid"
    else:
        value_col = "v8a_minus_wt_mean_contacts"
        wt_sd_col = "wt_sd_contacts"
        v8a_sd_col = "v8a_sd_contacts"
        suffix = "raw_contacts"
        cbar_label = "V8A - WT mean contacts"

    plot_df["system_label"] = (
        plot_df["system"]
        .map(SYSTEM_LABELS)
        .fillna(plot_df["system"])
    )

    system_order = [
        SYSTEM_LABELS[s]
        for s in ["pro_dim", "mem_dim"]
        if s in set(plot_df["system"])
    ]

    lipid_order = [
        lipid
        for lipid in LIPID_ORDER
        if lipid in set(plot_df["lipid"])
    ]

    value_matrix = plot_df.pivot_table(
        index="lipid",
        columns="system_label",
        values=value_col,
    )

    wt_sd_matrix = plot_df.pivot_table(
        index="lipid",
        columns="system_label",
        values=wt_sd_col,
    )

    v8a_sd_matrix = plot_df.pivot_table(
        index="lipid",
        columns="system_label",
        values=v8a_sd_col,
    )

    value_matrix = value_matrix.reindex(
        index=lipid_order,
        columns=system_order,
    )

    wt_sd_matrix = wt_sd_matrix.reindex(
        index=lipid_order,
        columns=system_order,
    )

    v8a_sd_matrix = v8a_sd_matrix.reindex(
        index=lipid_order,
        columns=system_order,
    )

    if value_matrix.empty:
        return

    annot_matrix = value_matrix.copy().astype(object)

    for lipid in value_matrix.index:
        for system in value_matrix.columns:

            effect = value_matrix.loc[lipid, system]
            wt_sd = wt_sd_matrix.loc[lipid, system]
            v8a_sd = v8a_sd_matrix.loc[lipid, system]

            if pd.isna(effect):
                annot_matrix.loc[lipid, system] = ""
            else:
                annot_matrix.loc[lipid, system] = (
                    f"{effect:.3f}\n"
                    f"WT SD={wt_sd:.3f}\n"
                    f"V8A SD={v8a_sd:.3f}"
                )

    plt.figure(figsize=(10, 9))

    sns.heatmap(
        value_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
    )

    plt.xlabel("System")
    plt.ylabel("Lipid")
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)

    save_plot(
        f"03_heatmap_v8a_minus_wt_lipid_contacts_{suffix}.png"
    )


def plot_all_systems_one_figure(replicate_summary: pd.DataFrame, normalized: bool = False) -> None:
    """
    Compact overview: all systems × lipids in one faceted figure.
    """
    plot_df = replicate_summary.copy()

    plot_df = plot_df[
        ~(
                (plot_df["system"] == "mem_dim")
                & (plot_df["variant"] == "wt")
                & (plot_df["replicate"] == "repl1")
        )
    ].copy()

    y_col = "mean_contacts_per_lipid" if normalized else "mean_contacts"
    y_label = "Mean contacts per available lipid" if normalized else "Mean number of contacts"
    suffix = "normalized_per_lipid" if normalized else "raw_contacts"

    plot_df = plot_df.copy()
    plot_df["system_label"] = plot_df["system"].map(SYSTEM_LABELS).fillna(plot_df["system"])
    plot_df["system_label"] = pd.Categorical(
        plot_df["system_label"],
        categories=[SYSTEM_LABELS[s] for s in SYSTEM_ORDER],
        ordered=True,
    )

    lipid_order = ordered_lipids_present(plot_df)

    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="lipid",
        y=y_col,
        hue="variant",
        col="system_label",
        col_wrap=2,
        order=lipid_order,
        hue_order=["wt", "v8a"],
        errorbar="sd",
        capsize=0.15,
        palette={"wt": dark_blue, "v8a": dark_pink},
        height=6,
        aspect=1.5,
        sharey=False,
    )

    g.set_axis_labels("Lipid", y_label)
    g.set_titles("{col_name}")
    g._legend.set_title("")
    for text in g._legend.texts:
        text.set_text(text.get_text().upper())

    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    plt.tight_layout()
    plt.savefig(GRAPH_DIR / f"04_all_systems_lipid_specific_contacts_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_all_plots(df: pd.DataFrame, replicate_summary: pd.DataFrame, t_tests: pd.DataFrame) -> None:
    plot_wt_vs_v8a_replicate_pairs(df)
    plot_time_series_mean_sd(df)
    plot_lipid_contact_bars_by_system(replicate_summary, normalized=False)
    plot_lipid_contact_bars_by_system(replicate_summary, normalized=True)
    plot_v8a_minus_wt_heatmap(t_tests, normalized=False)
    plot_v8a_minus_wt_heatmap(t_tests, normalized=True)
    plot_all_systems_one_figure(replicate_summary, normalized=False)
    plot_all_systems_one_figure(replicate_summary, normalized=True)


# -----------------------------
# Main script
# -----------------------------
def main() -> None:
    make_dirs()

    print(f"Reading lipid contact XVG files from: {INPUT_DIR}")
    df = load_all_data()

    # Exclude membrane-associated WT dimer replicate 1 from all summaries and graphs
    #df = exclude_problematic_mem_dim_wt_repl1(df)

    print(f"Loaded {df['source_file'].nunique()} contact files")
    print(f"Total data points: {len(df):,}")

    missing_replicates = check_missing_replicates(df)
    if not missing_replicates.empty:
        print("\nWarning: some groups are missing expected replicates.")
        print(missing_replicates.to_string(index=False))
        print("\nThis is okay for now if repl1/repl3 mem_mon_wt are still being produced.")

    replicate_summary = calculate_replicate_summary(df)
    condition_summary = calculate_wt_vs_v8a_summary(replicate_summary)
    differences = calculate_v8a_minus_wt_differences(replicate_summary)
    t_tests = calculate_t_tests(replicate_summary)

    save_tables(df, replicate_summary, condition_summary, differences, t_tests, missing_replicates)
    make_all_plots(df, replicate_summary, t_tests)

    print("\nDone!")
    print(f"Tables saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
