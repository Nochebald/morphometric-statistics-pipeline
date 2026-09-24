import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import linregress, spearmanr, pearsonr
import itertools
import streamlit.components.v1 as components

# Graceful imports for multivariate analysis (Module 9)
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from statsmodels.multivariate.manova import MANOVA
    HAS_MANOVA = True
except ImportError:
    HAS_MANOVA = False

# Graceful import for multiple testing
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(page_title="Stats Framework", layout="wide")

# --- PERSISTENT STATE ---
if "do_chain" not in st.session_state:
    st.session_state.do_chain = False
if "wb_metrics" not in st.session_state:
    st.session_state.wb_metrics = []          
if "wb_pairs" not in st.session_state:
    st.session_state.wb_pairs = []             

if st.session_state.get("pending_nav"):
    st.session_state["nav_module"] = st.session_state["pending_nav"]
    st.session_state["pending_nav"] = None


def pair_key(x, y):
    return tuple(sorted((x, y)))


def wb_add_metrics(names):
    for n in names:
        if n and n not in st.session_state.wb_metrics:
            st.session_state.wb_metrics.append(n)


def wb_add_pairs(pairs):
    for x, y in pairs:
        k = pair_key(x, y)
        if k not in st.session_state.wb_pairs:
            st.session_state.wb_pairs.append(k)


def wb_clear():
    st.session_state.wb_metrics = []
    st.session_state.wb_pairs = []


def wb_pair_options_present(options):
    wb_set = set(st.session_state.wb_pairs)
    matches = []
    for opt in options:
        if " vs " not in opt:
            continue
        x, y = opt.split(" vs ", 1)
        if pair_key(x, y) in wb_set:
            matches.append(opt)
    return matches


def ensure_valid_selection(key, options, fallback=None):
    fallback = fallback or []
    current = st.session_state.get(key)
    if current is None:
        st.session_state[key] = list(fallback)
    else:
        st.session_state[key] = [v for v in current if v in options]


def ensure_valid_single_selection(key, options, fallback=None):
    current = st.session_state.get(key)
    if current is None or current not in options:
        st.session_state[key] = fallback if fallback in (options or [None]) else (options[0] if options else None)


def merge_into_selection(key, new_values):
    current = st.session_state.get(key, [])
    st.session_state[key] = list(dict.fromkeys(list(current) + list(new_values)))


def get_col_index(columns, primary_keywords, fallback_keywords=None):
    fallback_keywords = fallback_keywords or []
    cols_lower = [str(c).lower() for c in columns]
    for i, col in enumerate(cols_lower):
        if all(k in col for k in primary_keywords):
            return i
    for i, col in enumerate(cols_lower):
        if all(k in col for k in fallback_keywords):
            return i
    return 0


def stars_from_p(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def safe_shapiro(data):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 3 or np.allclose(data, data[0]):
        return 1.0
    try:
        sample = data if len(data) <= 5000 else np.random.default_rng(0).choice(data, 5000, replace=False)
        return float(stats.shapiro(sample)[1])
    except Exception:
        return 1.0


@st.cache_data(show_spinner=False)
def load_csv(file_bytes, source_label):
    import io
    d = pd.read_csv(io.BytesIO(file_bytes))
    d = d.loc[:, ~d.columns.str.contains("^Unnamed")]
    if d.columns.duplicated().any():
        d = d.loc[:, ~d.columns.duplicated()]
    d["Dataset_Source"] = source_label
    return d


def ordered_datasets(present):
    """Preserve the user's Primary/Secondary upload order (matches ds_color_map) instead of
    alphabetically sorting -- otherwise e.g. 'KO' < 'WT' would silently swap which genotype
    is 'first' (and therefore which color it gets) depending on the labels' spelling."""
    present = set(present)
    ordered = [d for d in (label_1, label_2) if d in present]
    ordered += [d for d in sorted(present) if d not in ordered]
    return ordered


def add_iqr_median_callout(ax, x_pos, q1, median, q3, arrow_color="red", text_color="black", fontsize=14):
    """Reference-style callout arrows labeling the IQR box and median line of one
    violin/raincloud group (mirrors the classic 'anatomy of a violin plot' figure).

    The arrow tips point at the true data locations (xy, in data coords), but the text
    itself is placed at fixed axes-fraction heights (60%/40%) rather than at the data
    y-values. A data-unit-based offset isn't reliable here: text height is fixed in
    points while the gap between the median and the IQR midpoint varies with the data's
    scale, so a fixed fraction of the axes is the only offset that can't collapse to
    zero and cause the two labels to overlap.
    """
    iqr_mid = (q1 + q3) / 2
    xlim = ax.get_xlim()
    room = max(1.3, (xlim[1] - x_pos) * 1.6)
    ax.set_xlim(xlim[0], max(xlim[1], x_pos + room))
    text_x = x_pos + room * 0.55

    gap = 0.28
    iqr_frac, med_frac = (0.5 + gap / 2, 0.5 - gap / 2) if iqr_mid >= median else (0.5 - gap / 2, 0.5 + gap / 2)

    ax.annotate("Interquartile range", xy=(x_pos + 0.12, iqr_mid), xycoords="data",
                xytext=(text_x, iqr_frac), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.6, shrinkA=0, shrinkB=2),
                fontsize=fontsize, va="center", ha="left", color=text_color, zorder=10)
    ax.annotate("Median", xy=(x_pos + 0.12, median), xycoords="data",
                xytext=(text_x, med_frac), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=1.6, shrinkA=0, shrinkB=2),
                fontsize=fontsize, va="center", ha="left", color=text_color, zorder=10)


def styled_plot(kind, data, x=None, y=None, palette="Set2", figsize=(5, 3),
                ds_color_map=None, solid_color=None, annotate=False):
    """Violin/box/bar/raincloud wrapper.

    ds_color_map: optional {category: hex_color} dict (e.g. {"WT": "#2E75B6", "KO": "#C0392B"}).
    Applied only when every category actually present in `data[x]` is one of its keys, so an
    arbitrary custom grouping column still falls back to the general `palette`.
    solid_color: explicit single color for the x=None (single-group) case, overriding `palette`.
    annotate: if True, add IQR/Median callout arrows to the last group (Violin/Raincloud only) --
    a one-time "how to read this" label rather than repeating it on every group.
    """
    if kind == "Raincloud":
        return raincloud_plot(data, x, y, palette=palette, figsize=figsize,
                              ds_color_map=ds_color_map, solid_color=solid_color, annotate=annotate)

    fig, ax = plt.subplots(figsize=figsize)
    has_hue = x is not None
    order = list(pd.unique(data[x].dropna())) if has_hue else None

    effective_palette = palette
    if has_hue and ds_color_map and order and set(order) <= set(ds_color_map.keys()):
        effective_palette = ds_color_map

    color = solid_color if (not has_hue and solid_color) else (None if has_hue else sns.color_palette(palette, 1)[0])
    kwargs = dict(data=data, y=y, ax=ax)
    if has_hue:
        kwargs.update(x=x, hue=x, palette=effective_palette, order=order, hue_order=order, legend=False)
    else:
        kwargs.update(color=color)

    if kind == "Violin":
        sns.violinplot(inner="box", **kwargs)
        if annotate:
            last_cat = order[-1] if order else None
            vals = pd.to_numeric(data.loc[data[x] == last_cat, y] if last_cat is not None else data[y], errors="coerce").dropna()
            if len(vals) >= 4:
                q1, median, q3 = np.percentile(vals, [25, 50, 75])
                add_iqr_median_callout(ax, (len(order) - 1) if order else 0, q1, median, q3)
    elif kind == "Boxplot":
        sns.boxplot(**kwargs)
    else:
        sns.barplot(**kwargs)
    return fig, ax


def raincloud_plot(data, x, y, palette="Set2", figsize=(5, 3), ax=None,
                    ds_color_map=None, solid_color=None, annotate=False):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if x is None:
        groups = [("", pd.to_numeric(data[y], errors="coerce").dropna().to_numpy())]
    else:
        cats = list(pd.unique(data[x].dropna()))
        groups = [(c, pd.to_numeric(data.loc[data[x] == c, y], errors="coerce").dropna().to_numpy()) for c in cats]
    groups = [(c, v) for c, v in groups if len(v) > 0]

    use_ds_colors = bool(ds_color_map) and x is not None and all(c in ds_color_map for c, _ in groups)
    palette_colors = sns.color_palette(palette, max(len(groups), 1))
    band = 0.36
    box_width = 0.10
    strip_offset = 0.16
    last_box_stats = None

    for i, (cat, vals) in enumerate(groups):
        if x is None and solid_color:
            color = solid_color
        elif use_ds_colors:
            color = ds_color_map[cat]
        else:
            color = palette_colors[i % len(palette_colors)]

        if len(vals) >= 2 and np.ptp(vals) > 0:
            kde = stats.gaussian_kde(vals)
            y_grid = np.linspace(vals.min(), vals.max(), 200)
            density = kde(y_grid)
            density = density / density.max() * band
            ax.fill_betweenx(y_grid, i, i + density, color=color, alpha=0.65, linewidth=0.8, edgecolor="black", zorder=2)

        ax.boxplot([vals], positions=[i + 0.02], widths=box_width, vert=True,
                  patch_artist=True, showfliers=False, zorder=3,
                  medianprops=dict(color="black", linewidth=1.4),
                  boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.0),
                  whiskerprops=dict(color="black", linewidth=1.0),
                  capprops=dict(color="black", linewidth=1.0))
        if len(vals) >= 4:
            last_box_stats = (i, *np.percentile(vals, [25, 50, 75]))

        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(np.full(len(vals), i - strip_offset) + jitter, vals, s=8, color=color,
                  alpha=0.5, edgecolors="none", zorder=1)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(c) for c, _ in groups])
    ax.set_ylabel(y)
    if annotate and last_box_stats is not None:
        add_iqr_median_callout(ax, *last_box_stats)
    return fig, ax


def qq_plot(ax, data, label=None, color="#2E75B6"):
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
    ax.scatter(osm, osr, s=12, color=color, alpha=0.6, edgecolors="none", label=label)
    ax.plot(osm, slope * osm + intercept, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title(f"QQ-Plot{f' ({label})' if label else ''} — R²={r**2:.3f}", fontsize=14)
    return ax


def ecdf(data):
    data = np.sort(np.asarray(data, dtype=float))
    data = data[~np.isnan(data)]
    n = len(data)
    y = np.arange(1, n + 1) / n
    return data, y


def mad_outlier_mask(series, thresh=3.5):
    x = series.to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0: return np.zeros(len(x), dtype=bool)
    modified_z = 0.6745 * (x - med) / mad
    return np.abs(modified_z) > thresh


def mahalanobis_outlier_mask(df_subset, thresh=3.0):
    X = df_subset.to_numpy(dtype=float)
    valid_rows = ~np.isnan(X).any(axis=1)
    mask = np.zeros(len(X), dtype=bool)
    Xv = X[valid_rows]
    if len(Xv) < X.shape[1] + 2: return mask
    cov = np.cov(Xv, rowvar=False)
    try: inv_cov = np.linalg.pinv(cov)
    except np.linalg.LinAlgError: return mask
    mean = Xv.mean(axis=0)
    diff = Xv - mean
    dist2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    cutoff = thresh ** 2 * X.shape[1]
    flagged = dist2 > cutoff
    mask[valid_rows] = flagged
    return mask


def evaluate_continuous_stats(data_list, metric_name):
    norm_p_vals = [safe_shapiro(d) for d in data_list]
    is_normal = all(p >= 0.05 for p in norm_p_vals)

    if len(data_list) > 1:
        try: levene_p = stats.levene(*data_list)[1]
        except Exception: levene_p = 1.0
        is_equal_var = levene_p >= 0.05
    else:
        is_equal_var = True

    group_count = len(data_list)
    audit = [
        {"Assumption": "Normality (Shapiro-Wilk)", "Result": "Passed (Gaussian) ✅" if is_normal else "Failed (Skewed) ❌"},
        {"Assumption": "Equal Variance (Levene's)", "Result": "Passed (Equal) ✅" if is_equal_var else "Failed (Unequal) ❌"},
    ]

    stat = np.nan
    if group_count == 2:
        if is_normal and is_equal_var:
            test_name = "Student's T-Test"
            stat, p_val = stats.ttest_ind(*data_list, equal_var=True)
        elif is_normal and not is_equal_var:
            test_name = "Welch's T-Test"
            stat, p_val = stats.ttest_ind(*data_list, equal_var=False)
        else:
            test_name = "Mann-Whitney U"
            stat, p_val = stats.mannwhitneyu(*data_list)
    elif group_count > 2:
        if is_normal and is_equal_var:
            test_name = "One-Way ANOVA"
            stat, p_val = stats.f_oneway(*data_list)
        else:
            test_name = "Kruskal-Wallis"
            stat, p_val = stats.kruskal(*data_list)
    else:
        test_name = "Insufficient Groups"
        p_val = np.nan

    return test_name, stat, p_val, audit, is_normal, is_equal_var


def aggregate_by_id(sub_df, id_col, value_col):
    return sub_df.groupby(id_col)[value_col].mean().dropna()


def run_paired_test(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y
    if len(diffs) < 3: return "Insufficient Pairs", np.nan, True
    is_normal = safe_shapiro(diffs) >= 0.05
    if is_normal:
        _, p = stats.ttest_rel(x, y)
        name = "Paired T-Test"
    else:
        try: _, p = stats.wilcoxon(x, y)
        except Exception: p = np.nan
        name = "Wilcoxon Signed-Rank"
    return name, p, is_normal


def dumbbell_plot(ids, vals1, vals2, label1="WT", label2="KO", figsize=(5, 4), color1="#2E75B6", color2="#C0392B", show_legend=True):
    ids = np.asarray(ids)
    vals1 = np.asarray(vals1, dtype=float)
    vals2 = np.asarray(vals2, dtype=float)
    order = np.argsort(vals1 - vals2)
    ids_sorted, v1, v2 = ids[order], vals1[order], vals2[order]
    y_pos = np.arange(len(ids_sorted))

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(ids_sorted)):
        ax.plot([v1[i], v2[i]], [y_pos[i], y_pos[i]], color="0.65", linewidth=1.2, zorder=1)
    ax.scatter(v1, y_pos, color=color1, s=45, label=label1, zorder=2, edgecolor="black", linewidth=0.6)
    ax.scatter(v2, y_pos, color=color2, s=45, label=label2, zorder=2, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(i) for i in ids_sorted], fontsize=14)
    if show_legend:
        ax.legend(frameon=True)
    sns.despine(left=True)
    fig.tight_layout()
    return fig, ax


def paired_bootstrap_corr(x, y, method="pearson", n_boot=2000, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    xm, ym = x[idx], y[idx]
    if method == "spearman":
        xm = stats.rankdata(xm, axis=1)
        ym = stats.rankdata(ym, axis=1)
    xc = xm - xm.mean(axis=1, keepdims=True)
    yc = ym - ym.mean(axis=1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return r[np.isfinite(r)]


def render_forest_plot(rows, datasets, multi_dataset, method_label,
                        dataset_color_mode="Blue/Red", palette_name="Set2",
                        value_key="R", x_range=(-1.05, 1.05), x_ticks=None,
                        pos_label="Positive R", neg_label="Negative R",
                        sample_label="Bootstrap R Samples", zero_line=True,
                        value_fmt="{:.2f}", xlabel=None,
                        forest_ds1_color="#2E75B6", forest_ds2_color="#C0392B", show_legend=True):
    pairs_order = list(dict.fromkeys(r["Pair"] for r in rows))
    if not multi_dataset:
        pairs_order.sort(key=lambda p: next(r[value_key] for r in rows if r["Pair"] == p), reverse=True)

    n_pairs = len(pairs_order)
    n_sub_max = max((len([r for r in rows if r["Pair"] == p]) for p in pairs_order), default=1)

    sub_spacing = 0.66
    row_height = 1.55 if not multi_dataset else max(2.1, sub_spacing * n_sub_max + 1.3)
    fig_h = max(4.3, n_pairs * row_height * 0.75 + 1.5)
    plot_w = 9.5
    legend_w = 3.4  # extra width reserved so the external legend has room and isn't clipped
    fig_w = plot_w + legend_w if show_legend else plot_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    pos_color, neg_color = forest_ds1_color, forest_ds2_color
    if multi_dataset:
        if dataset_color_mode == "Blue/Red" and len(datasets) == 2:
            ds_colors = {datasets[0]: forest_ds1_color, datasets[1]: forest_ds2_color}
        else:
            palette_colors = sns.color_palette(palette_name, n_colors=len(datasets))
            ds_colors = dict(zip(datasets, palette_colors))
    else:
        ds_colors = {}

    span = x_range[1] - x_range[0]
    jitter_std = max(span * 0.012, 1e-6)

    y_ticks, y_labels, all_y = [], [], []
    for i, pair in enumerate(pairs_order):
        base_y = (n_pairs - 1 - i) * row_height
        subset = [r for r in rows if r["Pair"] == pair]
        n_sub = len(subset)
        offsets = (np.arange(n_sub) - (n_sub - 1) / 2) * sub_spacing if n_sub > 1 else [0.0]

        for j, row in enumerate(subset):
            y = base_y + offsets[j]
            all_y.append(y)

            samples = row["Samples"]
            jitter = np.random.default_rng(0).normal(0, jitter_std, size=len(samples))
            ax.scatter(samples, y + jitter, s=2, color="0.55", alpha=0.12,
                       linewidths=0, zorder=1, rasterized=True)

            val = row[value_key]
            color = ds_colors[row["Dataset"]] if multi_dataset else (pos_color if val >= 0 else neg_color)
            err_low = max(0.0, val - row["CI_Low"])
            err_high = max(0.0, row["CI_High"] - val)
            ax.errorbar(val, y, xerr=[[err_low], [err_high]], fmt="o",
                        color=color, ecolor="black", elinewidth=1.6, capsize=5,
                        capthick=1.6, markersize=8, markeredgecolor="black",
                        markeredgewidth=0.6, zorder=3)

            text_y = y - (0.40 if not multi_dataset else 0.23)
            label_val = value_fmt.format(val)
            ax.text(val, text_y, f"{label_val}{row['Stars']}", ha="center",
                    va="top", fontsize=14, fontweight="bold", zorder=4)
            all_y.append(text_y)

        y_ticks.append(base_y)
        y_labels.append(pair)

    if zero_line: ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=2)
    ax.set_xlim(*x_range)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if x_ticks is not None: ax.set_xticks(x_ticks)
    else: ax.set_xticks(np.linspace(x_range[0], x_range[1], 9))
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=14)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(min(all_y) - 0.6, max(all_y) + 0.73)
    ax.set_xlabel(xlabel if xlabel is not None else f"{method_label} R Correlation Coefficient", fontsize=14)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1)

    if multi_dataset:
        handles = [Line2D([0], [0], marker="o", linestyle="None", markersize=8,
                          markerfacecolor=ds_colors[d], markeredgecolor="black", label=d) for d in datasets]
    else:
        handles = [
            Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor=pos_color, markeredgecolor="black", label=pos_label),
            Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor=neg_color, markeredgecolor="black", label=neg_label),
        ]
    handles += [
        Line2D([0], [0], color="black", linewidth=1.6, label="95% Bootstrap CI"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=4, color="0.55", alpha=0.6, label=sample_label),
    ]
    if show_legend:
        ax.legend(handles=handles, title="Metrics", loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=True, edgecolor="black", fontsize=14, title_fontsize=14)
        # loc="best" was picking whichever spot overlapped least with the data -- but the bootstrap
        # sample clouds spread across nearly the whole plot width, so "least bad" still overlapped.
        # Placing the legend outside the axes (in the extra width reserved above) guarantees it
        # never sits on top of a data point, regardless of how dense the plot is.
        fig.tight_layout(rect=[0, 0, plot_w / fig_w, 1])
    else:
        fig.tight_layout()
    return fig


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0: return np.nan
    d = (x.mean() - y.mean()) / pooled_sd
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return d * correction


def bootstrap_cohens_d(x, y, n_boot=2000, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    idx_x = rng.integers(0, nx, size=(n_boot, nx))
    idx_y = rng.integers(0, ny, size=(n_boot, ny))
    xb, yb = x[idx_x], y[idx_y]
    pooled_sd = np.sqrt(((nx - 1) * xb.var(axis=1, ddof=1) + (ny - 1) * yb.var(axis=1, ddof=1)) / (nx + ny - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (xb.mean(axis=1) - yb.mean(axis=1)) / pooled_sd
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    d = d * correction
    return d[np.isfinite(d)]


def build_correlation_matrix(data_subset, sel, corr_func):
    pair_stats = []
    p_values = []
    for x_var, y_var in itertools.combinations(sel, 2):
        try:
            x_arr = data_subset[x_var].to_numpy().ravel()
            y_arr = data_subset[y_var].to_numpy().ravel()
            if len(x_arr) != len(y_arr):
                raise ValueError("mismatched sample sizes")
            r, p = corr_func(x_arr, y_arr)
        except Exception:
            r, p = np.nan, np.nan
        p_values.append(p if np.isfinite(p) else 1.0)
        pair_stats.append({"Var1": x_var, "Var2": y_var, "R": r, "Raw_P": p})

    if HAS_STATSMODELS and p_values:
        _, fdr_q, _, _ = multipletests(p_values, method="fdr_bh")
        _, bonf_q, _, _ = multipletests(p_values, method="bonferroni")
        for i, s in enumerate(pair_stats):
            s["FDR_Q"] = fdr_q[i]
            s["Bonferroni_Q"] = bonf_q[i]
    else:
        for s in pair_stats:
            s["FDR_Q"], s["Bonferroni_Q"] = s["Raw_P"], s["Raw_P"]

    def get_corrected_p(x, y):
        for s in pair_stats:
            if (s["Var1"] == x and s["Var2"] == y) or (s["Var1"] == y and s["Var2"] == x):
                return s["R"], s["Bonferroni_Q"]
        return np.nan, np.nan

    g = sns.PairGrid(data_subset[sel], height=2.8)
    g.map_upper(sns.scatterplot, alpha=0.5)
    g.map_diag(sns.kdeplot, fill=True)

    def heatmap(x, y, **kwargs):
        r, q = get_corrected_p(x.name, y.name)
        ax = plt.gca()
        if np.isfinite(r):
            ax.set_facecolor(plt.get_cmap("coolwarm")((r + 1) / 2))
            ax.text(0.5, 0.5, f"{r:.2f}\n{'*' if q < 0.05 else ''}", transform=ax.transAxes,
                    ha="center", va="center", fontweight="bold")
        else:
            ax.set_facecolor("0.9")
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center")

    g.map_lower(heatmap)

    # seaborn's PairGrid sets an xlabel/ylabel on every subplot by default, not just the
    # outer edges -- at larger font sizes those interior labels collide with each other.
    # Enforce the standard "only the true edges are labeled" convention explicitly, and
    # cap each subplot to 2 tick values so the (now bigger, unrotated) numbers never
    # crowd into the axis-name label sitting right below them.
    n_vars = len(sel)
    for i in range(n_vars):
        for j in range(n_vars):
            ax = g.axes[i, j]
            ax.set_xlabel(sel[j] if i == n_vars - 1 else "")
            ax.set_ylabel(sel[i] if j == 0 else "")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=2))

    g.fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
    return g.fig, pair_stats


def radar_chart(categories, series_dict, colors=None, figsize=(5.5, 5.5), show_legend=True):
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = colors or sns.color_palette("Set2", len(series_dict))
    for (name, values), color in zip(series_dict.items(), colors):
        vals = list(values) + [values[0]]
        ax.plot(angles, vals, color=color, linewidth=2, label=name)
        ax.fill(angles, vals, color=color, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_yticklabels([])
    if show_legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig, ax


# =====================================================================================
# SIDEBAR
# =====================================================================================
st.sidebar.title("🧰 The Framework")
MODULE_ORDER = [
    "1. Overview & Group Distributions",
    "2. Statistical Comparison (WT vs KO)",
    "3. Correlation Structure",
    "4. Myelin Biology Deep-Dive",
    "5. Multivariate & Clustering",
    "6. Descriptive Statistics",
    "7. StatAll Comprehensive",
    "8. Scatter Explorer",
    "9. Allometric Scaling (Log-Log)",
    "10. Final Summary Report",
]
MODULE_ICONS = {
    "1. Overview & Group Distributions": "📊",
    "2. Statistical Comparison (WT vs KO)": "🧬",
    "3. Correlation Structure": "🔗",
    "4. Myelin Biology Deep-Dive": "🧪",
    "5. Multivariate & Clustering": "🔬",
    "6. Descriptive Statistics": "📈",
    "7. StatAll Comprehensive": "🚀",
    "8. Scatter Explorer": "🔭",
    "9. Allometric Scaling (Log-Log)": "📐",
    "10. Final Summary Report": "🏁",
}

st.sidebar.caption("**Guided WT vs KO workflow:** steps 1–5 · **Extra tools:** 6–9 · **10:** final report")
app_mode = st.sidebar.radio(
    "Select Step", MODULE_ORDER, key="nav_module",
    format_func=lambda x: f"{MODULE_ICONS.get(x, '')}  {x}",
)
color_palette = st.sidebar.selectbox("Color Palette", ["Set2", "viridis", "coolwarm", "mako"])
plot_type = st.sidebar.radio("Plot Type (Modules 1, 3, 4)", ["Violin", "Boxplot", "Bar", "Raincloud"])
genotype_scheme = st.sidebar.selectbox("Genotype Color Scheme", ["Blue / Red", "Blue / Orange"],
                                       help="Applied consistently to Primary/Secondary dataset colors across every plot in the app.")
GENOTYPE_SCHEMES = {"Blue / Red": ("#2E75B6", "#C0392B"), "Blue / Orange": ("#2E75B6", "#E67E22")}
ds1_color, ds2_color = GENOTYPE_SCHEMES[genotype_scheme]
show_iqr_median_labels = st.sidebar.checkbox("Show IQR/Median Labels (Violin & Raincloud)", value=True)
show_legend = st.sidebar.checkbox("Show Legends on Plots", value=True)
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 14, "legend.title_fontsize": 14,
})

st.sidebar.divider()
with st.sidebar.expander(f"🗂️ Significance Workbench ({len(st.session_state.wb_metrics)} metrics, {len(st.session_state.wb_pairs)} pairs)", expanded=False):
    st.caption("Mark significant results in step 2 or 3, then load them into any other step.")
    if st.session_state.wb_metrics:
        st.markdown("**Metrics:** " + ", ".join(st.session_state.wb_metrics))
    else:
        st.caption("No metrics added yet.")
    if st.session_state.wb_pairs:
        st.markdown("**Pairs:** " + "; ".join(f"{x} vs {y}" for x, y in st.session_state.wb_pairs))
    else:
        st.caption("No pairs added yet.")
    if st.button("🧹 Clear Workbench", use_container_width=True):
        wb_clear()
        st.rerun()

st.title("Automated Computational Framework")

# --- DUAL UPLOAD SYSTEM & CUSTOM LABELS ---
c1, c2 = st.columns(2)
with c1:
    uploaded_file = st.file_uploader("Upload Primary CSV", type=["csv"])
    label_1 = st.text_input("Primary Dataset Label (e.g. WT)", value="WT", max_chars=30)
with c2:
    uploaded_file_2 = st.file_uploader("Upload Secondary CSV [Optional]", type=["csv"])
    label_2 = st.text_input("Secondary Dataset Label (e.g. KO)", value="KO", max_chars=30)

if uploaded_file is not None:
    label_1 = label_1.strip() or "WT"
    label_2 = label_2.strip() or "KO"
    if uploaded_file_2 is not None and label_1 == label_2:
        st.warning(f"⚠️ Both dataset labels are '{label_1}' — renaming to keep the WT/KO split from breaking.")
        label_1, label_2 = f"{label_1} (1)", f"{label_2} (2)"
    ds_color_map = {label_1: ds1_color, label_2: ds2_color}

    try:
        df1 = load_csv(uploaded_file.getvalue(), label_1)
    except Exception as e:
        st.error(f"Could not read the primary CSV: {e}")
        st.stop()
    df = df1.copy()

    if uploaded_file_2 is not None:
        try:
            df2 = load_csv(uploaded_file_2.getvalue(), label_2)
            df = pd.concat([df1, df2], ignore_index=True)
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
            st.success("✅ Dual Datasets Successfully Merged.")
        except Exception as e:
            st.error(f"Could not read the secondary CSV: {e}")
            st.stop()

    # --- CHAIN REACTION ---
    st.divider()
    do_chain = st.toggle("Run Chain Reaction (Applies to all loaded datasets)", value=st.session_state.do_chain)
    st.session_state.do_chain = do_chain
    if do_chain:
        num_cols_pre = df.select_dtypes(include=np.number).columns.tolist()
        if not num_cols_pre:
            st.warning("⚠️ No numeric columns available for the Chain Reaction calculation.")
        else:
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                v_in = st.selectbox("Inner Vol", options=num_cols_pre, index=get_col_index(num_cols_pre, ["inner", "volume"], ["volume"]))
                s_in = st.selectbox("Inner SA", options=num_cols_pre, index=get_col_index(num_cols_pre, ["inner", "surface"], ["surface"]))
            with cc2:
                plane = st.selectbox("Plane Count", options=num_cols_pre, index=get_col_index(num_cols_pre, ["plane"]))
                v_my = st.selectbox("Myelin Vol", options=num_cols_pre, index=get_col_index(num_cols_pre, ["myelin", "volume"]))
            with cc3:
                s_my = st.selectbox("Myelin SA", options=num_cols_pre, index=get_col_index(num_cols_pre, ["myelin", "surface"]))
                z = st.number_input("Z-Step (μm)", value=0.05)

            with np.errstate(divide="ignore", invalid="ignore"):
                df["Length (μm)"] = df[plane] * z
                df["Inner Diameter (μm)"] = np.sqrt(4 * (df[v_in] / df["Length (μm)"]) / np.pi)
                df["G-Ratio (Vol)"] = np.sqrt(df[v_in] / (df[v_in] + df[v_my]))
                df["G-Ratio (SA)"] = df[s_in] / df[s_my]
                df["SA:V Ratio"] = df[s_in] / df[v_in]
                df["Sphericity"] = (np.pi ** (1 / 3) * (6 * df[v_in]) ** (2 / 3)) / df[s_in]
                df["Estimated Fiber Diameter (μm)"] = df["Inner Diameter (μm)"] / df["G-Ratio (Vol)"]
                df["Myelin Thickness (μm)"] = (df["Estimated Fiber Diameter (μm)"] - df["Inner Diameter (μm)"]) / 2
            st.success("✅ Chain Reaction Applied.")

    # --- OUTLIER DETECTION & TRIMMING ---
    if "outlier_excluded_idx" not in st.session_state:
        st.session_state.outlier_excluded_idx = []
    if st.session_state.outlier_excluded_idx:
        df = df.drop(index=[i for i in st.session_state.outlier_excluded_idx if i in df.index], errors="ignore")

    st.divider()
    with st.expander(f"🧹 Outlier Detection & Trimming (optional) — {len(st.session_state.outlier_excluded_idx)} row(s) removed so far", expanded=False):
        if st.session_state.outlier_excluded_idx:
            if st.button("↩️ Reset Trimming (restore all rows)", key="reset_outlier_trim"):
                st.session_state.outlier_excluded_idx = []
                st.rerun()

        pre_num_cols = df.select_dtypes(include=np.number).columns.tolist()
        outlier_method = st.radio(
            "Method",
            ["None", "Per-Column MAD (robust Z-score)", "Multivariate (Mahalanobis distance)"],
            horizontal=True,
            help="MAD flags extreme values one column at a time. Mahalanobis flags rows that are "
                 "jointly unusual across several columns at once (e.g. a biologically impossible "
                 "combination of volume and surface area), even if no single value looks extreme alone.",
        )
        if outlier_method == "Per-Column MAD (robust Z-score)":
            mad_cols = st.multiselect("Columns to screen", options=pre_num_cols, default=pre_num_cols)
            mad_thresh = st.slider("Modified Z-score threshold", 2.0, 6.0, 3.5, 0.5)
            if mad_cols:
                combined_mask = np.zeros(len(df), dtype=bool)
                for c in mad_cols:
                    combined_mask |= mad_outlier_mask(df[c], thresh=mad_thresh)
                st.write(f"**{combined_mask.sum()} of {len(df)} rows** flagged as outliers on at least one column (>{mad_thresh} MAD).")
                if combined_mask.sum() > 0 and st.button("Apply Trimming", key="apply_mad_trim"):
                    newly_excluded = df.index[combined_mask].tolist()
                    st.session_state.outlier_excluded_idx = sorted(set(st.session_state.outlier_excluded_idx) | set(newly_excluded))
                    st.rerun()
        elif outlier_method == "Multivariate (Mahalanobis distance)":
            maha_cols = st.multiselect("Columns to include", options=pre_num_cols, default=pre_num_cols[:4])
            maha_thresh = st.slider("Distance threshold (sigma-equivalent)", 2.0, 6.0, 3.0, 0.5)
            if len(maha_cols) >= 2:
                valid_idx = df[maha_cols].dropna().index
                mask = mahalanobis_outlier_mask(df.loc[valid_idx, maha_cols], thresh=maha_thresh)
                flagged_idx = valid_idx[mask]
                st.write(f"**{len(flagged_idx)} of {len(df)} rows** flagged as multivariate outliers (>{maha_thresh}σ-equivalent).")
                if len(flagged_idx) > 0 and st.button("Apply Trimming", key="apply_maha_trim"):
                    st.session_state.outlier_excluded_idx = sorted(set(st.session_state.outlier_excluded_idx) | set(flagged_idx.tolist()))
                    st.rerun()
            else:
                st.caption("Pick at least 2 columns for a multivariate distance.")

    st.subheader("📊 Master Dataset Preview")
    st.dataframe(df, use_container_width=True)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Workbench items are just column-name strings, so if a *different* CSV gets loaded
    # (or a metric gets renamed by Chain Reaction), stale entries silently make every
    # "load from Workbench" button show 0 and look broken rather than erroring loudly.
    stale_metrics = [m for m in st.session_state.wb_metrics if m not in num_cols]
    stale_pairs = [(x, y) for x, y in st.session_state.wb_pairs if x not in num_cols or y not in num_cols]
    if stale_metrics or stale_pairs:
        st.warning(
            f"⚠️ The Workbench has {len(stale_metrics)} metric(s) and {len(stale_pairs)} pair(s) that don't "
            "match any column in the currently loaded data (likely left over from a different CSV). "
            "Workbench buttons elsewhere will show a count of 0 until these are cleared."
        )
        if st.button("🧹 Prune Stale Workbench Entries"):
            st.session_state.wb_metrics = [m for m in st.session_state.wb_metrics if m in num_cols]
            st.session_state.wb_pairs = [(x, y) for x, y in st.session_state.wb_pairs if x in num_cols and y in num_cols]
            st.rerun()

    # --- STEP NAVIGATION BANNER ---
    st.divider()
    step_idx = MODULE_ORDER.index(app_mode)
    st.progress((step_idx + 1) / len(MODULE_ORDER), text=f"Step {step_idx + 1} of {len(MODULE_ORDER)}")
    nav_prev, nav_title, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("⬅️ Previous", disabled=step_idx == 0, use_container_width=True):
            st.session_state.pending_nav = MODULE_ORDER[step_idx - 1]
            st.rerun()
    with nav_title:
        st.markdown(f"<h4 style='text-align:center; margin:0;'>{MODULE_ICONS.get(app_mode, '')}&nbsp; {app_mode}</h4>", unsafe_allow_html=True)
    with nav_next:
        if st.button("Next ➡️", disabled=step_idx == len(MODULE_ORDER) - 1, use_container_width=True):
            st.session_state.pending_nav = MODULE_ORDER[step_idx + 1]
            st.rerun()
    st.divider()

    # --- MODULE 1: GROUP COMPARISON ---
    if app_mode == "1. Overview & Group Distributions":
        MAX_READABLE_CATEGORIES = 40
        grp = st.selectbox("Group Variable", options=df.columns)
        df_grp = df.copy()
        df_grp[grp] = df_grp[grp].astype(str)
        n_unique = df_grp[grp].nunique()

        plot_df = df_grp
        if n_unique > MAX_READABLE_CATEGORIES:
            st.warning(
                f"⚠️ **'{grp}'** has **{n_unique} unique values** — this looks like an ID "
                "column rather than a grouping variable, and plotting it directly would be unreadable."
            )
            limit_mode = st.radio(
                "How would you like to proceed?",
                ["Show top N most frequent categories", "Pick specific categories", "Plot all anyway (not recommended)"],
                horizontal=True,
            )
            if limit_mode == "Show top N most frequent categories":
                top_n = st.slider("Number of categories to show", 2, min(MAX_READABLE_CATEGORIES, n_unique), 10)
                keep_cats = df_grp[grp].value_counts().nlargest(top_n).index
                plot_df = df_grp[df_grp[grp].isin(keep_cats)]
            elif limit_mode == "Pick specific categories":
                all_cats = sorted(df_grp[grp].unique())
                chosen_cats = st.multiselect("Categories to include", options=all_cats, default=all_cats[:10])
                plot_df = df_grp[df_grp[grp].isin(chosen_cats)] if chosen_cats else df_grp.iloc[0:0]

        available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
        if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m1_load_wb"):
            merge_into_selection("m1_metrics_select", available_from_wb)
            st.rerun()
        ensure_valid_selection("m1_metrics_select", num_cols)
        met = st.multiselect("Metrics", options=num_cols, key="m1_metrics_select")
        n_cats_shown = max(plot_df[grp].nunique(), 1)
        fig_width = float(np.clip(0.35 * n_cats_shown + 2, 5, 22))
        tick_fontsize = float(np.clip(320 / n_cats_shown, 6, 14))

        for m in met:
            if m == grp:
                st.info(f"Skipping **{m}** — it's currently selected as both the Group Variable and a Metric.")
                continue
            if plot_df.empty:
                st.info("No data to plot for the current category selection.")
                continue
            fig, ax = styled_plot(plot_type, plot_df, x=grp, y=m, palette=color_palette, figsize=(fig_width, 3.5),
                                  ds_color_map=ds_color_map, annotate=show_iqr_median_labels)
            if n_cats_shown <= 60 and plot_type != "Raincloud":
                sns.stripplot(data=plot_df, x=grp, y=m, ax=ax, color="black", alpha=0.3, jitter=True, size=3)
            plt.xticks(rotation=45, ha="right", fontsize=tick_fontsize)
            ax.set(xlabel="", ylabel=m)
            sns.despine()
            st.pyplot(fig, use_container_width=False)

    # --- MODULE 2: DESCRIPTIVE ---
    elif app_mode == "6. Descriptive Statistics":
        if uploaded_file_2:
            st.info("Choose which dataset to summarize — pooling WT and KO together would mix genotypes into one mean/std.")
            target_ds_desc = st.radio("Dataset", ordered_datasets(df["Dataset_Source"].unique()), horizontal=True,
                                       label_visibility="collapsed", key="m2_target_ds")
            desc_df = df[df["Dataset_Source"] == target_ds_desc]
        else:
            target_ds_desc = label_1
            desc_df = df

        available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
        if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m2_load_wb"):
            merge_into_selection("m2_vars_select", available_from_wb)
            st.rerun()
        ensure_valid_selection("m2_vars_select", num_cols)
        cols = st.multiselect(f"Variables (for {target_ds_desc})", options=num_cols, key="m2_vars_select")
        if cols:
            st.dataframe(desc_df[cols].agg(["mean", "std", "count"]).T)

    # --- MODULE 3: COMPREHENSIVE STATALL ---
    elif app_mode == "7. StatAll Comprehensive":
        st.subheader("🚀 StatAll: Global Distribution & Basic Audits")

        if uploaded_file_2:
            st.info("StatAll is designed to provide full statistics for a single dataset at a time. Choose which one to analyze:")
            target_ds = st.radio("Dataset", ordered_datasets(df["Dataset_Source"].unique()), horizontal=True, label_visibility="collapsed")
            stat_df = df[df["Dataset_Source"] == target_ds]
        else:
            target_ds = label_1
            stat_df = df

        available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
        if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m7_statall_load_wb"):
            merge_into_selection("m7_statall_metrics", available_from_wb)
            st.rerun()
            
        ensure_valid_selection("m7_statall_metrics", num_cols)
        metrics = st.multiselect(f"Select Metrics (for {target_ds})", options=num_cols, key="m7_statall_metrics")
        
        master_results = []
        if metrics:
            for m in metrics:
                data = stat_df[m].dropna()
                stats_vals = data.agg(["count", "mean", "median", "std", "sem", "skew", "kurt"])
                shapiro_p = safe_shapiro(data)
                is_normal = shapiro_p >= 0.05

                st.markdown(f"### 📊 Metric: {m}")
                plot_col, qq_col = st.columns(2)
                with plot_col:
                    fig, ax = styled_plot(plot_type, stat_df, x=None, y=m, palette=color_palette,
                                          solid_color=ds_color_map.get(target_ds), annotate=show_iqr_median_labels)
                    sns.despine(left=True)
                    st.pyplot(fig, use_container_width=False)
                with qq_col:
                    fig_qq, ax_qq = plt.subplots(figsize=(5, 3))
                    qq_plot(ax_qq, data, label=target_ds)
                    st.pyplot(fig_qq, use_container_width=False)

                st.table(pd.DataFrame([
                    {"Test": "Shapiro-Wilk", "Purpose": "Normality Check", "Result": "Gaussian ✅" if is_normal else "Skewed ❌"},
                    {"Test": "One-Sample T-Test", "Purpose": "Mean != 0", "Result": "Applicable ✅" if is_normal else "Not Applicable"},
                    {"Test": "Wilcoxon Signed-Rank", "Purpose": "Median != 0", "Result": "Applicable ✅" if not is_normal else "Not Applicable"},
                ]))
                res = stats_vals.to_dict()
                res.update({"Metric": m, "Shapiro_P": shapiro_p, "Distribution": "Normal" if is_normal else "Skewed"})
                master_results.append(res)
            st.subheader("📋 Master Distribution Summary")
            st.dataframe(pd.DataFrame(master_results))

    # --- MODULE 4: TWO-DATASET COMPARISON ---
    elif app_mode == "2. Statistical Comparison (WT vs KO)":
        if uploaded_file_2 is None:
            st.warning("⚠️ Please upload a Secondary CSV at the top of the page to enable this module.")
        else:
            st.subheader("🧬 Two-Dataset Statistical Comparison")

            id_col_options = ["(none)"] + [c for c in df.columns if c != "Dataset_Source"]
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                animal_col = st.selectbox(
                    "Animal / Sample ID (optional)", id_col_options,
                    help="If multiple rows (e.g. axons) come from the same animal, provide its ID here. "
                         "The test will then run on per-animal means (N = animals), not pooled rows -- "
                         "otherwise the test pseudoreplicates.",
                )
            with mcol2:
                paired_col = st.selectbox(
                    "Paired / Matched ID (optional)", id_col_options,
                    help="If each Dataset 1 row has a matched Dataset 2 counterpart (e.g. same animal "
                         "measured twice, or left/right hemisphere), provide the shared ID here to run "
                         "a paired test instead of an independent one.",
                )
            animal_col = None if animal_col == "(none)" else animal_col
            paired_col = None if paired_col == "(none)" else paired_col
            if not animal_col and not paired_col:
                st.caption(
                    "ℹ️ No Animal/Sample ID given — the test below pools all rows together. "
                    "If your rows are repeated measurements from a smaller number of animals, this "
                    "pseudoreplicates and can overstate significance."
                )

            available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
            if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m4_load_wb"):
                merge_into_selection("m4_metrics_select", available_from_wb)
                st.rerun()

            ensure_valid_selection("m4_metrics_select", num_cols)
            metrics = st.multiselect("Select Metrics to Compare", options=num_cols, key="m4_metrics_select")
            master_results = []
            if metrics:
                for m in metrics:
                    ds1 = df[df["Dataset_Source"] == label_1][m].dropna()
                    ds2 = df[df["Dataset_Source"] == label_2][m].dropna()

                    st.markdown(f"### 📊 Metric: {m}")
                    plot_col, qq_col1, qq_col2 = st.columns(3)
                    with plot_col:
                        fig, ax = styled_plot(plot_type, df, x="Dataset_Source", y=m, palette=color_palette,
                                              ds_color_map=ds_color_map, annotate=show_iqr_median_labels)
                        if plot_type != "Raincloud":
                            sns.stripplot(data=df, x="Dataset_Source", y=m, ax=ax, color="black", alpha=0.3, jitter=True)
                        ax.set(xlabel="", ylabel=m)
                        sns.despine()
                        st.pyplot(fig, use_container_width=False)
                    with qq_col1:
                        fig_qq1, ax_qq1 = plt.subplots(figsize=(4, 3))
                        qq_plot(ax_qq1, ds1, label=label_1, color=ds1_color)
                        st.pyplot(fig_qq1, use_container_width=False)
                    with qq_col2:
                        fig_qq2, ax_qq2 = plt.subplots(figsize=(4, 3))
                        qq_plot(ax_qq2, ds2, label=label_2, color=ds2_color)
                        st.pyplot(fig_qq2, use_container_width=False)

                    pooled_test_used, pooled_stat, pooled_p, pooled_audit, pooled_normal, pooled_equal_var = evaluate_continuous_stats([ds1, ds2], m)

                    primary_label = "Pooled (per-row, pseudoreplicated if repeated measures)"
                    primary_test, primary_p, primary_n1, primary_n2 = pooled_test_used, pooled_p, len(ds1), len(ds2)
                    audit_to_show = pooled_audit

                    if paired_col:
                        sub1 = df[df["Dataset_Source"] == label_1][[paired_col, m]].dropna()
                        sub2 = df[df["Dataset_Source"] == label_2][[paired_col, m]].dropna()
                        a1 = aggregate_by_id(sub1, paired_col, m)
                        a2 = aggregate_by_id(sub2, paired_col, m)
                        matched = a1.to_frame("v1").join(a2.to_frame("v2"), how="inner")
                        if len(matched) >= 3:
                            paired_name, paired_p_val, diffs_normal = run_paired_test(matched["v1"], matched["v2"])
                            st.markdown(f"**Paired analysis** ({len(matched)} matched {paired_col} pairs)")
                            fig_db, ax_db = dumbbell_plot(matched.index, matched["v1"], matched["v2"], label1=label_1, label2=label_2, figsize=(5, max(3, len(matched) * 0.25)), color1=ds1_color, color2=ds2_color, show_legend=show_legend)
                            st.pyplot(fig_db, use_container_width=False)
                            primary_label = f"Paired ({paired_name})"
                            primary_test, primary_p = paired_name, paired_p_val
                            primary_n1 = primary_n2 = len(matched)
                            audit_to_show = [{"Assumption": "Normality of Differences (Shapiro-Wilk)",
                                               "Result": "Passed ✅" if diffs_normal else "Failed ❌"}]
                        else:
                            st.warning(f"⚠️ Only {len(matched)} matched '{paired_col}' pair(s) found — falling back to the pooled test below.")
                    elif animal_col:
                        a1 = aggregate_by_id(df[df["Dataset_Source"] == label_1][[animal_col, m]].dropna(), animal_col, m)
                        a2 = aggregate_by_id(df[df["Dataset_Source"] == label_2][[animal_col, m]].dropna(), animal_col, m)
                        if len(a1) >= 2 and len(a2) >= 2:
                            animal_test_used, animal_stat, animal_p, animal_audit, animal_normal, animal_equal_var = evaluate_continuous_stats([a1, a2], m)
                            st.caption(f"Animal-level means: {label_1} N={len(a1)} animals, {label_2} N={len(a2)} animals "
                                       f"(pooled across {len(ds1)} + {len(ds2)} rows).")
                            primary_label = f"Animal-Level ({animal_test_used})"
                            primary_test, primary_p = animal_test_used, animal_p
                            primary_n1, primary_n2 = len(a1), len(a2)
                            audit_to_show = animal_audit
                        else:
                            st.warning("⚠️ Not enough animals per group for an animal-level test — falling back to the pooled test below.")

                    st.table(pd.DataFrame(audit_to_show))
                    sig_txt = "Yes ✅" if np.isfinite(primary_p) and primary_p < 0.05 else "No ❌"
                    st.success(f"**Primary Test ({primary_label}):** {primary_test} | **N:** {primary_n1} vs {primary_n2} | "
                               f"**P-Value:** {primary_p:.4e} | **Significant:** {sig_txt}")
                    if primary_label.startswith("Pooled") and (animal_col or paired_col):
                        st.caption("(Fell back to the pooled/pseudoreplicated test -- see warning above.)")
                    elif primary_label.startswith("Pooled"):
                        with st.expander("Show pooled per-row test details"):
                            st.write(f"{pooled_test_used}: p = {pooled_p:.4e} (N = {len(ds1)} vs {len(ds2)} rows)")

                    master_results.append({
                        "Metric": m, f"{label_1} (Mean)": ds1.mean(), f"{label_2} (Mean)": ds2.mean(),
                        "Analysis Level": primary_label, "Test Applied": primary_test,
                        "N1": primary_n1, "N2": primary_n2,
                        "P-Value": primary_p, "Significant": bool(np.isfinite(primary_p) and primary_p < 0.05),
                    })
                st.subheader("📋 Advanced Comparison Summary")
                st.dataframe(pd.DataFrame(master_results))

                sig_metrics = [r["Metric"] for r in master_results if r["Significant"]]
                col_add, col_info = st.columns([1, 2])
                with col_add:
                    if st.button(f"➕ Add {len(sig_metrics)} Significant Metric(s) to Workbench",
                                 disabled=not sig_metrics, key="m4_add_wb"):
                        wb_add_metrics(sig_metrics)
                        st.rerun()
                with col_info:
                    if sig_metrics:
                        st.caption("Then jump to step 3 (Correlation Structure) to explore their correlations and bootstrap CIs.")

    # --- MODULE 5: INDIVIDUAL SCATTER ---
    elif app_mode == "8. Scatter Explorer":
        available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
        if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m5_load_wb"):
            merge_into_selection("m5_vars_select", available_from_wb)
            st.rerun()
        ensure_valid_selection("m5_vars_select", num_cols, fallback=num_cols[:2])
        sel = st.multiselect("Variables", options=num_cols, key="m5_vars_select")

        only_wb_pairs = False
        if st.session_state.wb_pairs:
            only_wb_pairs = st.checkbox(
                f"Only plot the {len(st.session_state.wb_pairs)} pair(s) currently in the Workbench",
                value=False,
            )

        if len(sel) > 1:
            if only_wb_pairs:
                wb_set = set(st.session_state.wb_pairs)
                pairs_to_plot = [c for c in itertools.combinations(sel, 2) if pair_key(*c) in wb_set]
                if not pairs_to_plot:
                    st.info("None of the selected variables form a pair that's currently in the Workbench.")
            else:
                pairs_to_plot = list(itertools.combinations(sel, 2))

            for x_var, y_var in pairs_to_plot:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.scatterplot(data=df, x=x_var, y=y_var,
                                hue="Dataset_Source" if uploaded_file_2 else None, ax=ax,
                                alpha=0.5, palette=ds_color_map if uploaded_file_2 else None,
                                legend=show_legend if uploaded_file_2 else False)
                ax.set(xlabel=x_var, ylabel=y_var)
                sns.despine()
                st.pyplot(fig, use_container_width=False)

    # --- MODULE 6: ADVANCED CORRELATION HEATMAP & FOREST PLOTS ---
    elif app_mode == "3. Correlation Structure":
        corr_method = st.radio("Correlation Method", ["Pearson", "Spearman"], horizontal=True)
        corr_func = pearsonr if corr_method == "Pearson" else spearmanr

        available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
        if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m6_load_wb"):
            merge_into_selection("m6_var_matrix", available_from_wb)
            st.rerun()
        ensure_valid_selection("m6_var_matrix", num_cols, fallback=num_cols[:4])
        sel = st.multiselect("Variables for Matrix", options=num_cols, key="m6_var_matrix")

        if len(sel) > 1:
            clean_data = df[sel + (["Dataset_Source"] if uploaded_file_2 else [])].dropna()

            if uploaded_file_2:
                corr_scope = st.radio(
                    "Correlation Scope", ["Pooled (all data)", "Per-Genotype (side-by-side)"],
                    horizontal=True,
                    help="Pooling WT+KO together before correlating can produce spurious correlations "
                         "driven purely by genotype group membership (Simpson's paradox). Per-Genotype "
                         "computes each correlation matrix independently within groups.",
                )
            else:
                corr_scope = "Pooled (all data)"

            if corr_scope.startswith("Pooled"):
                fig_grid, pair_stats = build_correlation_matrix(clean_data, sel, corr_func)
                st.pyplot(fig_grid, use_container_width=False)
                pair_df = pd.DataFrame(pair_stats)
                pair_df["Pair"] = pair_df["Var1"] + " vs " + pair_df["Var2"]
            else:
                datasets_for_heatmap = ordered_datasets(clean_data["Dataset_Source"].unique())
                heatmap_cols = st.columns(len(datasets_for_heatmap))
                all_pair_rows = []
                for col, ds in zip(heatmap_cols, datasets_for_heatmap):
                    with col:
                        st.markdown(f"**{ds}**")
                        subset = clean_data[clean_data["Dataset_Source"] == ds]
                        fig_grid, pair_stats_ds = build_correlation_matrix(subset, sel, corr_func)
                        st.pyplot(fig_grid, use_container_width=False)
                        for s in pair_stats_ds:
                            s["Dataset"] = ds
                        all_pair_rows.extend(pair_stats_ds)
                pair_df = pd.DataFrame(all_pair_rows)
                pair_df["Pair"] = pair_df["Var1"] + " vs " + pair_df["Var2"]

            st.divider()
            st.subheader("📋 Pairwise Correlation Table")
            sig_basis_label = st.selectbox(
                "Significance Criterion",
                ["Bonferroni Q < 0.05", "FDR Q < 0.05", "Raw P < 0.05"],
                help="Which corrected (or uncorrected) p-value determines the 'Significant' flag below.",
            )
            sig_col = {"Bonferroni Q < 0.05": "Bonferroni_Q", "FDR Q < 0.05": "FDR_Q", "Raw P < 0.05": "Raw_P"}[sig_basis_label]

            pair_df["Significant"] = pair_df[sig_col] < 0.05
            display_cols = ["Pair"] + (["Dataset"] if "Dataset" in pair_df.columns else []) + ["R", "Raw_P", "FDR_Q", "Bonferroni_Q", "Significant"]
            st.dataframe(
                pair_df[display_cols].style.format({
                    "R": "{:.3f}", "Raw_P": "{:.2e}", "FDR_Q": "{:.2e}", "Bonferroni_Q": "{:.2e}",
                }),
                use_container_width=True,
            )

            sig_pairs_df = pair_df[pair_df["Significant"]]
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                if st.button(f"➕ Add {len(sig_pairs_df)} Significant Pair(s) to Workbench",
                             disabled=sig_pairs_df.empty, key="m6_add_pairs_wb"):
                    wb_add_pairs(list(zip(sig_pairs_df["Var1"], sig_pairs_df["Var2"])))
                    st.rerun()
            with col_p2:
                sig_vars = sorted(set(sig_pairs_df["Var1"]) | set(sig_pairs_df["Var2"]))
                if st.button(f"➕ Add {len(sig_vars)} Variable(s) from Significant Pairs to Workbench",
                             disabled=not sig_vars, key="m6_add_vars_wb"):
                    wb_add_metrics(sig_vars)
                    st.rerun()

            st.divider()
            st.subheader("🌲 Bootstrapped Confidence Intervals (Forest Plot)")

            forest_pair_options = [f"{x} vs {y}" for x, y in itertools.combinations(sel, 2)]
            wb_matches = wb_pair_options_present(forest_pair_options)
            sig_pair_strs = sorted(set(sig_pairs_df["Pair"].tolist()))

            bcol1, bcol2, bcol3 = st.columns(3)
            with bcol1:
                if st.button(f"📥 Load Significant Pairs ({len(sig_pair_strs)})",
                             disabled=not sig_pair_strs, key="m6_load_sig_pairs"):
                    st.session_state.m6_forest_pairs = sig_pair_strs
                    st.rerun()
            with bcol2:
                if st.button(f"🗂️ Load from Workbench ({len(wb_matches)})",
                             disabled=not wb_matches, key="m6_load_wb_pairs"):
                    st.session_state.m6_forest_pairs = wb_matches
                    st.rerun()
            with bcol3:
                if st.button("🧹 Clear Selection", key="m6_clear_pairs"):
                    st.session_state.m6_forest_pairs = []
                    st.rerun()

            ensure_valid_selection("m6_forest_pairs", forest_pair_options)
            selected_pairs = st.multiselect(
                "Select Pairs for Forest Plot",
                options=forest_pair_options,
                key="m6_forest_pairs",
            )
            n_boot = st.slider("Bootstrap Iterations", min_value=500, max_value=5000, value=2000, step=500)
            dataset_color_mode = "Blue/Red"  # follows the global Genotype Color Scheme in the sidebar

            if selected_pairs and st.button("Run Bootstrapping & Plot", type="primary"):
                with st.spinner(f"Running {n_boot:,} bootstrap iterations..."):
                    rng = np.random.default_rng()
                    method_key = "pearson" if corr_method == "Pearson" else "spearman"
                    datasets = ordered_datasets(clean_data["Dataset_Source"].unique()) if uploaded_file_2 else [label_1]

                    forest_rows = []
                    skipped = []
                    for pair_str in selected_pairs:
                        x_var, y_var = pair_str.split(" vs ", 1)
                        for ds in datasets:
                            try:
                                ds_data = clean_data[clean_data["Dataset_Source"] == ds] if uploaded_file_2 else clean_data
                                x_data = ds_data[x_var].to_numpy().ravel()
                                y_data = ds_data[y_var].to_numpy().ravel()
                                n = len(x_data)
                                if n != len(y_data) or n < 5:
                                    continue

                                r_val, p_val = corr_func(x_data, y_data)
                                boot_r = paired_bootstrap_corr(x_data, y_data, method=method_key, n_boot=n_boot, rng=rng)
                                if len(boot_r) < 10:
                                    continue
                                ci_low, ci_hi = np.percentile(boot_r, [2.5, 97.5])

                                forest_rows.append({
                                    "Pair": pair_str, "X": x_var, "Y": y_var, "Dataset": ds,
                                    "N": n, "R": r_val, "P_raw": p_val,
                                    "CI_Low": ci_low, "CI_High": ci_hi, "Samples": boot_r,
                                })
                            except Exception as e:
                                skipped.append(f"{pair_str} ({ds}): {e}")

                    if skipped:
                        st.caption(f"⚠️ Skipped {len(skipped)} combination(s): " + "; ".join(skipped))

                    if not forest_rows:
                        st.warning("⚠️ No pair had enough non-missing paired observations (n ≥ 5) to bootstrap.")
                    else:
                        raw_p = [row["P_raw"] for row in forest_rows]
                        if HAS_STATSMODELS:
                            _, corrected_p, _, _ = multipletests(raw_p, method="bonferroni")
                        else:
                            corrected_p = raw_p
                        for row, cp in zip(forest_rows, corrected_p):
                            row["P_corrected"] = cp
                            row["Stars"] = stars_from_p(cp)

                        fig = render_forest_plot(forest_rows, datasets, bool(uploaded_file_2), corr_method,
                                                  dataset_color_mode=dataset_color_mode, palette_name=color_palette,
                                                  forest_ds1_color=ds1_color, forest_ds2_color=ds2_color, show_legend=show_legend)
                        st.pyplot(fig, use_container_width=False)

                        display_df = pd.DataFrame(forest_rows)[
                            ["Pair", "Dataset", "N", "R", "P_raw", "P_corrected", "CI_Low", "CI_High", "Stars"]
                        ]
                        st.dataframe(
                            display_df.style.format({
                                "R": "{:.3f}", "P_raw": "{:.2e}", "P_corrected": "{:.2e}",
                                "CI_Low": "{:.3f}", "CI_High": "{:.3f}",
                            }),
                            use_container_width=True,
                        )

                        sig_boot_pairs = sorted({row["Pair"] for row in forest_rows if row["Stars"]})
                        if st.button(f"➕ Add {len(sig_boot_pairs)} Significant Bootstrapped Pair(s) to Workbench",
                                     disabled=not sig_boot_pairs, key="m6_add_boot_sig_wb"):
                            wb_add_pairs([tuple(p.split(" vs ", 1)) for p in sig_boot_pairs])
                            st.rerun()

    # --- MODULE 7: ALLOMETRIC SCALING ---
    elif app_mode == "9. Allometric Scaling (Log-Log)":
        st.subheader("📏 Allometric Scaling (Log-Log Regression)")

        wb_pair_choices = [f"{x} vs {y}" for x, y in st.session_state.wb_pairs if x in num_cols and y in num_cols]
        if wb_pair_choices:
            ensure_valid_single_selection("m7_quick_pick", ["(none)"] + wb_pair_choices, fallback="(none)")
            quick_pick = st.selectbox("🗂️ Quick-pick from Workbench pairs", ["(none)"] + wb_pair_choices, key="m7_quick_pick")
            if quick_pick != "(none)":
                qx, qy = quick_pick.split(" vs ", 1)
                st.session_state.m7_x_var, st.session_state.m7_y_var = qx, qy

        ensure_valid_single_selection("m7_x_var", num_cols)
        ensure_valid_single_selection("m7_y_var", num_cols)
        col_x, col_y = st.columns(2)
        with col_x:
            x_var = st.selectbox("X Variable", options=num_cols, key="m7_x_var")
        with col_y:
            y_var = st.selectbox("Y Variable", options=num_cols, key="m7_y_var")

        if x_var == y_var:
            st.warning("⚠️ Please choose two different variables for X and Y.")
        else:
            keep_cols = [x_var, y_var] + (["Dataset_Source"] if uploaded_file_2 else [])
            clean_data = df[keep_cols].dropna()
            clean_data = clean_data[(clean_data[x_var] > 0) & (clean_data[y_var] > 0)]

            if len(clean_data) > 2:
                fig, ax = plt.subplots(figsize=(5, 3))
                datasets = ordered_datasets(clean_data["Dataset_Source"].unique()) if uploaded_file_2 else [label_1]
                for ds in datasets:
                    ds_data = clean_data[clean_data["Dataset_Source"] == ds] if uploaded_file_2 else clean_data
                    if len(ds_data) < 3:
                        st.caption(f"⚠️ Skipping {ds} — fewer than 3 positive, non-missing points.")
                        continue
                    try:
                        slope, intercept, _, _, _ = linregress(np.log10(ds_data[x_var]), np.log10(ds_data[y_var]))
                    except Exception as e:
                        st.caption(f"⚠️ Skipping {ds}: {e}")
                        continue
                    ax.scatter(ds_data[x_var], ds_data[y_var], alpha=0.5, label=f"{ds} (k={slope:.2f})")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set(xlabel=x_var, ylabel=y_var)
                if show_legend:
                    ax.legend()
                sns.despine()
                st.pyplot(fig, use_container_width=False)
            else:
                st.warning("⚠️ Not enough positive, non-missing paired observations to fit a log-log regression.")

    # --- MODULE 8: MYELIN BIOLOGY ---
    elif app_mode == "4. Myelin Biology Deep-Dive":
        if uploaded_file_2 is None:
            st.warning("⚠️ Please upload a Secondary CSV at the top of the page to enable this module.")
        else:
            tab_scatter, tab_binned, tab_ecdf, tab_effect, tab_radar = st.tabs([
                "G-Ratio vs Diameter", "Binned G-Ratio", "ECDF + KS Test",
                "Effect-Size Forest Plot", "Radar Profile",
            ])
            datasets_m8 = ordered_datasets(df["Dataset_Source"].unique())
            ds_colors_m8 = ds_color_map if len(datasets_m8) == 2 else \
                dict(zip(datasets_m8, sns.color_palette(color_palette, len(datasets_m8))))

            diam_default = get_col_index(num_cols, ["diameter"], ["axon"])
            gratio_default = get_col_index(num_cols, ["g-ratio"], ["gratio"])
            
            wb_pair_choices = [f"{x} vs {y}" for x, y in st.session_state.wb_pairs if x in num_cols and y in num_cols]

            # ---------------- TAB 1: G-Ratio vs Diameter scatter + regression ----------------
            with tab_scatter:
                st.caption("The canonical myelination figure: g-ratio vs. axon diameter, with an independent linear fit per genotype.")
                if wb_pair_choices:
                    ensure_valid_single_selection("m8_quick_scatter", ["(none)"] + wb_pair_choices, fallback="(none)")
                    qp1 = st.selectbox("🗂️ Quick-pick pair from Workbench", ["(none)"] + wb_pair_choices, key="m8_quick_scatter")
                    if qp1 != "(none)":
                        qx, qy = qp1.split(" vs ", 1)
                        st.session_state.m8_x = qx
                        st.session_state.m8_y = qy

                c1, c2 = st.columns(2)
                with c1:
                    ensure_valid_single_selection("m8_x", num_cols, fallback=num_cols[diam_default])
                    x_col = st.selectbox("Diameter (X)", num_cols, key="m8_x")
                with c2:
                    ensure_valid_single_selection("m8_y", num_cols, fallback=num_cols[gratio_default])
                    y_col = st.selectbox("G-Ratio (Y)", num_cols, key="m8_y")

                if x_col == y_col:
                    st.warning("⚠️ Please choose two different columns.")
                else:
                    clean = df[[x_col, y_col, "Dataset_Source"]].dropna()
                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    fit_rows = []
                    for ds in datasets_m8:
                        sub = clean[clean["Dataset_Source"] == ds]
                        if len(sub) < 3:
                            continue
                        color = ds_colors_m8[ds]
                        ax.scatter(sub[x_col], sub[y_col], alpha=0.35, s=16, color=color, label=None, edgecolors="none")
                        slope, intercept, r, p, se = linregress(sub[x_col], sub[y_col])
                        xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2.2,
                                label=f"{ds} fit (R²={r**2:.2f})")
                        fit_rows.append({"Dataset": ds, "N": len(sub), "Slope": slope, "Intercept": intercept,
                                          "R": r, "R²": r**2, "P-Value": p})
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    if show_legend:
                        ax.legend(frameon=True)
                    sns.despine()
                    st.pyplot(fig, use_container_width=False)
                    if fit_rows:
                        st.dataframe(pd.DataFrame(fit_rows).style.format({
                            "Slope": "{:.4f}", "Intercept": "{:.4f}", "R": "{:.3f}", "R²": "{:.3f}", "P-Value": "{:.2e}",
                        }), use_container_width=True)

            # ---------------- TAB 2: Binned mean +/- SEM ----------------
            with tab_binned:
                st.caption("Axon diameter binned into equal-width bins; mean ± SEM g-ratio per bin per genotype.")
                if wb_pair_choices:
                    ensure_valid_single_selection("m8_quick_binned", ["(none)"] + wb_pair_choices, fallback="(none)")
                    qp2 = st.selectbox("🗂️ Quick-pick pair from Workbench", ["(none)"] + wb_pair_choices, key="m8_quick_binned")
                    if qp2 != "(none)":
                        qx, qy = qp2.split(" vs ", 1)
                        st.session_state.m8_xb = qx
                        st.session_state.m8_yb = qy

                c1, c2, c3 = st.columns(3)
                with c1:
                    ensure_valid_single_selection("m8_xb", num_cols, fallback=num_cols[diam_default])
                    xb_col = st.selectbox("Diameter (X)", num_cols, key="m8_xb")
                with c2:
                    ensure_valid_single_selection("m8_yb", num_cols, fallback=num_cols[gratio_default])
                    yb_col = st.selectbox("G-Ratio (Y)", num_cols, key="m8_yb")
                with c3:
                    n_bins = st.slider("Number of bins", 4, 30, 10)

                if xb_col == yb_col:
                    st.warning("⚠️ Please choose two different columns.")
                else:
                    clean = df[[xb_col, yb_col, "Dataset_Source"]].dropna()
                    bin_edges = np.linspace(clean[xb_col].min(), clean[xb_col].max(), n_bins + 1)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    fig, ax = plt.subplots(figsize=(6, 4.5))
                    for ds in datasets_m8:
                        sub = clean[clean["Dataset_Source"] == ds].copy()
                        sub["bin"] = pd.cut(sub[xb_col], bins=bin_edges, labels=False, include_lowest=True)
                        grouped = sub.groupby("bin")[yb_col].agg(["mean", "sem", "count"])
                        grouped = grouped[grouped["count"] >= 2]
                        xs = bin_centers[grouped.index.to_numpy()]
                        color = ds_colors_m8[ds]
                        ax.errorbar(xs, grouped["mean"], yerr=grouped["sem"], fmt="o-", color=color,
                                    ecolor=color, capsize=4, markeredgecolor="black", markeredgewidth=0.5,
                                    label=ds, linewidth=1.8)
                    ax.set_xlabel(f"{xb_col} (binned)")
                    ax.set_ylabel(yb_col)
                    if show_legend:
                        ax.legend(frameon=True)
                    sns.despine()
                    st.pyplot(fig, use_container_width=False)

            # ---------------- TAB 3: ECDF + KS test ----------------
            with tab_ecdf:
                st.caption("Cumulative distributions often reveal a shift more clearly than box/violin plots.")
                available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
                if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m8_ecdf_load_wb"):
                    merge_into_selection("m8_ecdf_metrics", available_from_wb)
                    st.rerun()
                ensure_valid_selection("m8_ecdf_metrics", num_cols, fallback=num_cols[:1])
                ecdf_metrics = st.multiselect("Metrics", options=num_cols, key="m8_ecdf_metrics")
                for m in ecdf_metrics:
                    d1 = df[df["Dataset_Source"] == datasets_m8[0]][m].dropna()
                    d2 = df[df["Dataset_Source"] == datasets_m8[1]][m].dropna() if len(datasets_m8) > 1 else pd.Series(dtype=float)
                    fig, ax = plt.subplots(figsize=(5.5, 3.5))
                    for ds, d, in zip(datasets_m8, [d1, d2]):
                        if len(d) == 0:
                            continue
                        xs, ys = ecdf(d)
                        ax.step(xs, ys, where="post", color=ds_colors_m8[ds], linewidth=2, label=ds)
                    ax.set_xlabel(m)
                    ax.set_ylabel("Cumulative Probability")
                    if show_legend:
                        ax.legend(frameon=True)
                    sns.despine()
                    st.pyplot(fig, use_container_width=False)
                    if len(d1) > 0 and len(d2) > 0:
                        ks_stat, ks_p = stats.ks_2samp(d1, d2)
                        sig_txt = "Yes ✅" if ks_p < 0.05 else "No ❌"
                        st.success(f"**{m}** — Kolmogorov-Smirnov: D={ks_stat:.3f}, p={ks_p:.4e}, Significant: {sig_txt}")

            # ---------------- TAB 4: Cohen's d effect-size forest plot ----------------
            with tab_effect:
                st.caption("Standardized mean difference (Hedges'-corrected Cohen's d) with bootstrap 95% CI, across all selected metrics at once.")
                available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
                if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m8_load_wb"):
                    merge_into_selection("m8_effect_metrics", available_from_wb)
                    st.rerun()
                ensure_valid_selection("m8_effect_metrics", num_cols, fallback=num_cols[:4])
                effect_metrics = st.multiselect("Metrics", options=num_cols, key="m8_effect_metrics")

                animal_col_m8 = st.selectbox(
                    "Animal / Sample ID (optional, avoids pseudoreplication)",
                    ["(none)"] + [c for c in df.columns if c != "Dataset_Source"], key="m8_animal_col",
                )
                animal_col_m8 = None if animal_col_m8 == "(none)" else animal_col_m8
                n_boot_m8 = st.slider("Bootstrap Iterations", 500, 5000, 2000, 500, key="m8_nboot")

                if effect_metrics and len(datasets_m8) == 2 and st.button("Run Effect-Size Analysis", type="primary", key="m8_run_effect"):
                    rng = np.random.default_rng()
                    rows = []
                    for m in effect_metrics:
                        g1 = df[df["Dataset_Source"] == datasets_m8[0]][[m] + ([animal_col_m8] if animal_col_m8 else [])].dropna()
                        g2 = df[df["Dataset_Source"] == datasets_m8[1]][[m] + ([animal_col_m8] if animal_col_m8 else [])].dropna()
                        if animal_col_m8:
                            v1 = aggregate_by_id(g1, animal_col_m8, m).to_numpy()
                            v2 = aggregate_by_id(g2, animal_col_m8, m).to_numpy()
                        else:
                            v1 = g1[m].to_numpy()
                            v2 = g2[m].to_numpy()
                        if len(v1) < 2 or len(v2) < 2:
                            continue
                        d_val = cohens_d(v1, v2)
                        _, _, p_val, _, _, _ = evaluate_continuous_stats([v1, v2], m)
                        boot_d = bootstrap_cohens_d(v1, v2, n_boot=n_boot_m8, rng=rng)
                        if len(boot_d) < 10:
                            continue
                        ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])
                        rows.append({"Pair": m, "Dataset": "Effect", "D": d_val, "CI_Low": ci_lo, "CI_High": ci_hi,
                                     "Samples": boot_d, "P_raw": p_val, "N1": len(v1), "N2": len(v2)})

                    if not rows:
                        st.warning("⚠️ Not enough data per metric to compute an effect size.")
                    else:
                        raw_p = [r["P_raw"] for r in rows]
                        if HAS_STATSMODELS:
                            _, corrected_p, _, _ = multipletests(raw_p, method="bonferroni")
                        else:
                            corrected_p = raw_p
                        for r, cp in zip(rows, corrected_p):
                            r["Stars"] = stars_from_p(cp)
                            r["P_corrected"] = cp

                        all_vals = [r["CI_Low"] for r in rows] + [r["CI_High"] for r in rows]
                        pad = max(0.3, (max(all_vals) - min(all_vals)) * 0.15)
                        x_range = (min(all_vals) - pad, max(all_vals) + pad)

                        fig = render_forest_plot(
                            rows, datasets_m8, multi_dataset=False, method_label="Cohen's d",
                            value_key="D", x_range=x_range, x_ticks=np.round(np.linspace(x_range[0], x_range[1], 7), 2),
                            pos_label=f"{datasets_m8[0]} > {datasets_m8[1]}", neg_label=f"{datasets_m8[1]} > {datasets_m8[0]}",
                            sample_label="Bootstrap d Samples", value_fmt="d={:.2f}",
                            xlabel=f"Cohen's d  ({datasets_m8[0]} − {datasets_m8[1]})",
                            forest_ds1_color=ds1_color, forest_ds2_color=ds2_color, show_legend=show_legend,
                        )
                        st.pyplot(fig, use_container_width=False)

                        disp = pd.DataFrame(rows)[["Pair", "D", "N1", "N2", "P_raw", "P_corrected", "CI_Low", "CI_High", "Stars"]]
                        st.dataframe(disp.style.format({
                            "D": "{:.3f}", "P_raw": "{:.2e}", "P_corrected": "{:.2e}", "CI_Low": "{:.3f}", "CI_High": "{:.3f}",
                        }), use_container_width=True)

                        sig_effect_metrics = [r["Pair"] for r in rows if r["Stars"]]
                        if st.button(f"➕ Add {len(sig_effect_metrics)} Significant Metric(s) to Workbench",
                                     disabled=not sig_effect_metrics, key="m8_add_wb"):
                            wb_add_metrics(sig_effect_metrics)
                            st.rerun()
                elif len(datasets_m8) != 2:
                    st.info("Effect-size analysis needs exactly two datasets.")

            # ---------------- TAB 5: Radar / spider morphological profile ----------------
            with tab_radar:
                st.caption("Each metric is z-scored across both genotypes, then averaged per genotype — "
                           "showing which traits diverge most, on a common scale.")
                
                available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]
                if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m8_radar_wb"):
                    merge_into_selection("m8_radar_metrics", available_from_wb)
                    st.rerun()

                ensure_valid_selection("m8_radar_metrics", num_cols, fallback=num_cols[:5])
                radar_metrics = st.multiselect("Metrics (3+ recommended)", options=num_cols, key="m8_radar_metrics")

                if len(radar_metrics) >= 3:
                    clean = df[radar_metrics + ["Dataset_Source"]].dropna()
                    z = clean.copy()
                    for m in radar_metrics:
                        mu, sigma = clean[m].mean(), clean[m].std()
                        z[m] = (clean[m] - mu) / sigma if sigma > 0 else 0.0
                    profile = z.groupby("Dataset_Source")[radar_metrics].mean()
                    
                    lo, hi = profile.to_numpy().min(), profile.to_numpy().max()
                    span = max(hi - lo, 1e-6)
                    series = {ds: ((profile.loc[ds] - lo) / span).tolist() for ds in profile.index if ds in datasets_m8}
                    colors = [ds_colors_m8[ds] for ds in series]
                    fig, ax = radar_chart(radar_metrics, series, colors=colors, show_legend=show_legend)
                    st.pyplot(fig, use_container_width=False)
                    st.caption("Values are relative (z-scored then min-max scaled for display) — read shape and direction, not absolute magnitude.")
                else:
                    st.info("Pick at least 3 metrics to draw a radar profile.")

    # --- MODULE 9: MULTIVARIATE ANALYSIS & CLUSTERING ---
    elif app_mode == "5. Multivariate & Clustering":
        if not HAS_SKLEARN:
            st.error("scikit-learn isn't installed in this environment, so PCA and ROC/AUC are unavailable. "
                      "Run `pip install scikit-learn` and restart the app.")
        tab_pca, tab_cluster, tab_roc, tab_manova = st.tabs([
            "PCA", "Hierarchical Clustering", "ROC / AUC", "MANOVA",
        ])
        has_two = uploaded_file_2 is not None
        
        available_from_wb = [m for m in st.session_state.wb_metrics if m in num_cols]

        # ---------------- TAB 1: PCA ----------------
        with tab_pca:
            st.caption("Condenses several correlated metrics into 2 axes -- shows whether the groups form separate "
                       "'phenotypic fingerprints' rather than relying on any single metric.")
            if HAS_SKLEARN:
                if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m9_pca_wb"):
                    merge_into_selection("m9_pca_vars", available_from_wb)
                    st.rerun()
                ensure_valid_selection("m9_pca_vars", num_cols, fallback=num_cols[:5])
                pca_vars = st.multiselect("Metrics to include", options=num_cols, key="m9_pca_vars")
                color_by = "Dataset_Source" if has_two else st.selectbox(
                    "Color points by", options=[c for c in df.columns if c not in num_cols] or ["(none)"])

                if len(pca_vars) >= 2:
                    clean = df[pca_vars + ([color_by] if color_by and color_by != "(none)" else [])].dropna()
                    X = StandardScaler().fit_transform(clean[pca_vars].to_numpy())
                    pca = PCA(n_components=min(len(pca_vars), 2))
                    scores = pca.fit_transform(X)

                    fig, ax = plt.subplots(figsize=(6, 5))
                    if color_by and color_by != "(none)" and color_by in clean.columns:
                        groups = clean[color_by].astype(str)
                        cats = sorted(groups.unique())
                        colors_map = ds_color_map if set(cats) <= {label_1, label_2} \
                            else dict(zip(cats, sns.color_palette(color_palette, len(cats))))
                        for cat in cats:
                            mask = (groups == cat).to_numpy()
                            ax.scatter(scores[mask, 0], scores[mask, 1] if scores.shape[1] > 1 else np.zeros(mask.sum()),
                                      alpha=0.55, s=22, color=colors_map[cat], label=cat, edgecolors="none")
                        if show_legend:
                            ax.legend(frameon=True)
                    else:
                        ax.scatter(scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores)),
                                  alpha=0.55, s=22, color=ds1_color, edgecolors="none")
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
                    if scores.shape[1] > 1:
                        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
                    ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
                    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
                    sns.despine()
                    st.pyplot(fig, use_container_width=False)

                    var_fig, var_ax = plt.subplots(figsize=(4, 2.5))
                    pcs_full = PCA(n_components=len(pca_vars)).fit(X)
                    var_ax.bar(range(1, len(pca_vars) + 1), pcs_full.explained_variance_ratio_ * 100, color="#2E75B6")
                    var_ax.set_xlabel("Principal Component")
                    var_ax.set_ylabel("% Variance Explained")
                    sns.despine()
                    st.pyplot(var_fig, use_container_width=False)

                    loadings = pd.DataFrame(pca.components_.T, index=pca_vars,
                                            columns=[f"PC{i+1}" for i in range(scores.shape[1])])
                    st.markdown("**Loadings** (contribution of each metric to each component)")
                    st.dataframe(loadings.style.format("{:.3f}"), use_container_width=True)
                else:
                    st.info("Pick at least 2 metrics for PCA.")

        # ---------------- TAB 2: Hierarchical clustering (clustermap) ----------------
        with tab_cluster:
            st.caption("Clusters both fibers/subjects (rows) and metrics (columns) by similarity -- "
                       "can reveal sub-populations within a genotype that a single average would hide.")
            if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m9_cluster_wb"):
                merge_into_selection("m9_cluster_vars", available_from_wb)
                st.rerun()
            ensure_valid_selection("m9_cluster_vars", num_cols, fallback=num_cols[:6])
            cluster_vars = st.multiselect("Metrics to include", options=num_cols, key="m9_cluster_vars")
            max_rows = st.slider("Max rows to cluster (for speed/readability)", 20, 500, 150, key="m9_cluster_maxrows")

            if len(cluster_vars) >= 2:
                clean = df[cluster_vars + (["Dataset_Source"] if has_two else [])].dropna()
                if len(clean) > max_rows:
                    clean = clean.sample(max_rows, random_state=0)
                z = (clean[cluster_vars] - clean[cluster_vars].mean()) / clean[cluster_vars].std()
                z = z.fillna(0)

                row_colors = None
                if has_two:
                    row_colors = clean["Dataset_Source"].map(ds_color_map)

                with st.spinner("Clustering..."):
                    cg = sns.clustermap(z, row_colors=row_colors, cmap="coolwarm", center=0,
                                        figsize=(7, max(5, len(cluster_vars) * 0.4 + 4)),
                                        xticklabels=True, yticklabels=False, dendrogram_ratio=(0.15, 0.15))
                st.pyplot(cg.fig, use_container_width=False)
                if has_two:
                    ds2_emoji = "🔴" if genotype_scheme == "Blue / Red" else "🟠"
                    st.caption(f"Row color: 🔵 {label_1}  {ds2_emoji} {label_2}")
            else:
                st.info("Pick at least 2 metrics to cluster.")

        # ---------------- TAB 3: ROC / AUC ----------------
        with tab_roc:
            st.caption("How well does a single metric discriminate the two groups on its own?")
            if not has_two:
                st.info("ROC/AUC needs two datasets (a binary label).")
            elif HAS_SKLEARN:
                if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m9_roc_wb"):
                    merge_into_selection("m9_roc_vars", available_from_wb)
                    st.rerun()
                ensure_valid_selection("m9_roc_vars", num_cols, fallback=num_cols[:4])
                roc_vars = st.multiselect("Metrics to evaluate", options=num_cols, key="m9_roc_vars")
                if roc_vars:
                    clean = df[roc_vars + ["Dataset_Source"]].dropna()
                    y_true = (clean["Dataset_Source"] == label_2).astype(int).to_numpy()
                    fig, ax = plt.subplots(figsize=(5.5, 5))
                    roc_rows = []
                    for m in roc_vars:
                        score = clean[m].to_numpy()
                        auc = roc_auc_score(y_true, score)
                        flipped = auc < 0.5
                        fpr, tpr, _ = roc_curve(y_true, -score if flipped else score)
                        auc_show = 1 - auc if flipped else auc
                        ax.plot(fpr, tpr, linewidth=2, label=f"{m} (AUC={auc_show:.2f}{' ↓' if flipped else ''})")
                        roc_rows.append({"Metric": m, "AUC": auc_show, "Direction": f"{label_2} > {label_1}" if flipped else f"{label_1} > {label_2}"})
                    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    if show_legend:
                        ax.legend(fontsize=14, loc="lower right")
                    sns.despine()
                    st.pyplot(fig, use_container_width=False)
                    st.dataframe(pd.DataFrame(roc_rows).sort_values("AUC", ascending=False)
                                .style.format({"AUC": "{:.3f}"}), use_container_width=True)
                    st.caption(f"'↓' means the metric is *lower* in {label_2} -- AUC is reported flipped (1-AUC) so higher always means more discriminative.")

        # ---------------- TAB 4: MANOVA ----------------
        with tab_manova:
            st.caption("Tests whether the groups differ across several (possibly correlated) metrics jointly, "
                       "protecting against inflated false-positive risk from testing each one separately.")
            if not has_two:
                st.info("MANOVA needs two datasets (the grouping factor).")
            elif not HAS_MANOVA:
                st.error("statsmodels' MANOVA isn't available in this environment.")
            else:
                if available_from_wb and st.button(f"📥 Load {len(available_from_wb)} Metric(s) from Workbench", key="m9_manova_wb"):
                    merge_into_selection("m9_manova_vars", available_from_wb)
                    st.rerun()
                ensure_valid_selection("m9_manova_vars", num_cols, fallback=num_cols[:3])
                manova_vars = st.multiselect("Dependent variables (2+)", options=num_cols, key="m9_manova_vars")
                if len(manova_vars) >= 2:
                    clean = df[manova_vars + ["Dataset_Source"]].dropna()
                    dv_terms = " + ".join(f"Q('{v}')" for v in manova_vars)
                    formula = f"{dv_terms} ~ Dataset_Source"
                    try:
                        result = MANOVA.from_formula(formula, data=clean)
                        manova_result = result.mv_test()
                        summary_txt = str(manova_result)
                        st.code(summary_txt, language=None)
                    except Exception as e:
                        st.error(f"MANOVA failed to run: {e}")
                else:
                    st.info("Pick at least 2 dependent variables.")

    # --- MODULE 10: FINAL SUMMARY REPORT ---
    elif app_mode == "10. Final Summary Report":
        st.caption("A one-stop recap of everything flagged as significant across the workflow, "
                   "plus the key figures for a paper or presentation.")

        has_two_final = uploaded_file_2 is not None
        datasets_final = [label_1, label_2] if has_two_final else [label_1]

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Datasets Loaded", len(datasets_final))
        rc2.metric("Rows (total)", len(df))
        rc3.metric("Numeric Metrics", len(num_cols))

        st.markdown("#### 🗂️ Workbench Contents")
        if not st.session_state.wb_metrics and not st.session_state.wb_pairs:
            st.info("The Workbench is empty. Go back to **2. Statistical Comparison** or **3. Correlation "
                    "Structure**, mark significant results, and come back here — or just pick metrics "
                    "manually below.")
        else:
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.markdown("**Significant Metrics**")
                st.write(", ".join(st.session_state.wb_metrics) if st.session_state.wb_metrics else "(none)")
            with wcol2:
                st.markdown("**Significant Pairs**")
                st.write("; ".join(f"{x} vs {y}" for x, y in st.session_state.wb_pairs) if st.session_state.wb_pairs else "(none)")

        st.divider()
        st.markdown("#### ⚙️ Report Settings")
        default_metrics = st.session_state.wb_metrics if st.session_state.wb_metrics else num_cols[:4]
        ensure_valid_selection("m10_report_metrics", num_cols, fallback=default_metrics)
        report_metrics = st.multiselect("Metrics to summarize", options=num_cols, key="m10_report_metrics")
        n_boot_final = st.slider("Bootstrap Iterations", 500, 5000, 2000, 500, key="m10_nboot")

        if st.button("🏁 Generate Comprehensive Report", type="primary", disabled=not report_metrics):
            rng = np.random.default_rng()

            # ---------------- Key findings table (full auto-selected test + effect size) ----------------
            if has_two_final and len(datasets_final) == 2:
                st.markdown("### 📋 Key Findings")
                finding_rows = []
                for m in report_metrics:
                    v1 = df[df["Dataset_Source"] == datasets_final[0]][m].dropna().to_numpy()
                    v2 = df[df["Dataset_Source"] == datasets_final[1]][m].dropna().to_numpy()
                    if len(v1) < 2 or len(v2) < 2:
                        continue
                    d_val = cohens_d(v1, v2)
                    test_name, stat, p_val, _, is_normal, is_equal_var = evaluate_continuous_stats([v1, v2], m)
                    finding_rows.append({
                        "Metric": m,
                        f"{datasets_final[0]} Mean±SD": f"{v1.mean():.3f} ± {v1.std():.3f}",
                        f"{datasets_final[1]} Mean±SD": f"{v2.mean():.3f} ± {v2.std():.3f}",
                        "N1": len(v1), "N2": len(v2),
                        "Normal": "Yes ✅" if is_normal else "No ❌",
                        "Equal Var": "Yes ✅" if is_equal_var else "No ❌",
                        "Test Used": test_name,
                        "Test Statistic": stat,
                        "Cohen's d": d_val,
                        "P-Value": p_val,
                    })
                if finding_rows:
                    raw_p = [r["P-Value"] for r in finding_rows]
                    if HAS_STATSMODELS and raw_p:
                        _, corr_p, _, _ = multipletests(raw_p, method="bonferroni")
                    else:
                        corr_p = raw_p
                    for r, cp in zip(finding_rows, corr_p):
                        r["P (Bonferroni)"] = cp
                        r["Significant"] = bool(cp < 0.05)

                    findings_df = pd.DataFrame(finding_rows)
                    st.caption("**Test Used** is auto-selected the same way as in step 2 (Normality → Equal Variance → "
                               "parametric/non-parametric test); **Test Statistic** is that test's raw statistic "
                               "(t, U, F, or H depending on which test was selected); **Cohen's d** is the standardized "
                               "effect size (positive = higher in " + datasets_final[0] + ").")
                    st.dataframe(findings_df.style.format({
                        "Test Statistic": "{:.3f}", "Cohen's d": "{:.3f}", "P-Value": "{:.2e}", "P (Bonferroni)": "{:.2e}",
                    }), use_container_width=True)

                    n_sig = int(findings_df["Significant"].sum())
                    st.success(f"**{n_sig} of {len(findings_df)}** metrics differ significantly between "
                               f"{datasets_final[0]} and {datasets_final[1]} (Bonferroni-corrected p<0.05).")

                    csv_bytes = findings_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Key Findings (CSV)", csv_bytes,
                                       "key_findings_summary.csv", "text/csv")

                    # ---------------- Effect-size forest plot ----------------
                    st.markdown("### 🌲 Effect-Size Forest Plot")
                    forest_rows_final = []
                    for m in report_metrics:
                        v1 = df[df["Dataset_Source"] == datasets_final[0]][m].dropna().to_numpy()
                        v2 = df[df["Dataset_Source"] == datasets_final[1]][m].dropna().to_numpy()
                        if len(v1) < 2 or len(v2) < 2:
                            continue
                        d_val = cohens_d(v1, v2)
                        _, _, p_val, _, _, _ = evaluate_continuous_stats([v1, v2], m)
                        boot_d = bootstrap_cohens_d(v1, v2, n_boot=n_boot_final, rng=rng)
                        if len(boot_d) < 10:
                            continue
                        ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])
                        forest_rows_final.append({"Pair": m, "Dataset": "Effect", "D": d_val,
                                                   "CI_Low": ci_lo, "CI_High": ci_hi, "Samples": boot_d, "P_raw": p_val})
                    if forest_rows_final:
                        raw_p2 = [r["P_raw"] for r in forest_rows_final]
                        corr_p2 = multipletests(raw_p2, method="bonferroni")[1] if HAS_STATSMODELS else raw_p2
                        for r, cp in zip(forest_rows_final, corr_p2):
                            r["Stars"] = stars_from_p(cp)
                        all_vals = [r["CI_Low"] for r in forest_rows_final] + [r["CI_High"] for r in forest_rows_final]
                        pad = max(0.3, (max(all_vals) - min(all_vals)) * 0.15)
                        x_range = (min(all_vals) - pad, max(all_vals) + pad)
                        fig = render_forest_plot(
                            forest_rows_final, datasets_final, multi_dataset=False, method_label="Cohen's d",
                            value_key="D", x_range=x_range, x_ticks=np.round(np.linspace(x_range[0], x_range[1], 7), 2),
                            pos_label=f"{datasets_final[0]} > {datasets_final[1]}",
                            neg_label=f"{datasets_final[1]} > {datasets_final[0]}",
                            sample_label="Bootstrap d Samples", value_fmt="d={:.2f}",
                            xlabel=f"Cohen's d  ({datasets_final[0]} − {datasets_final[1]})",
                            show_legend=show_legend,
                        )
                        st.pyplot(fig, use_container_width=False)

                        forest_disp = pd.DataFrame(forest_rows_final)[
                            ["Pair", "D", "CI_Low", "CI_High", "P_raw", "Stars"]
                        ].rename(columns={"D": "Cohen's d", "CI_Low": "95% CI Low", "CI_High": "95% CI High",
                                           "P_raw": "P-Value", "Stars": "Sig."})
                        st.dataframe(forest_disp.style.format({
                            "Cohen's d": "{:.3f}", "95% CI Low": "{:.3f}", "95% CI High": "{:.3f}", "P-Value": "{:.2e}",
                        }), use_container_width=True)

            else:
                st.info("Upload a second CSV to include comparative effect-size findings in the report.")

            # ---------------- Significant Distributions Grid ----------------
            if report_metrics:
                st.markdown("### 📊 Distributions of Significant Metrics")
                cols_dist = st.columns(min(len(report_metrics), 3))
                for i, m in enumerate(report_metrics):
                    with cols_dist[i % 3]:
                        fig, ax = styled_plot(plot_type, df, x="Dataset_Source" if has_two_final else None, y=m,
                                              palette=color_palette, figsize=(4, 3),
                                              ds_color_map=ds_color_map if has_two_final else None,
                                              solid_color=ds_color_map.get(label_1) if not has_two_final else None,
                                              annotate=show_iqr_median_labels)
                        if plot_type != "Raincloud" and has_two_final:
                            sns.stripplot(data=df, x="Dataset_Source", y=m, ax=ax, color="black", alpha=0.3, jitter=True, size=3)
                        ax.set(xlabel="")
                        sns.despine()
                        st.pyplot(fig, use_container_width=False)
            
            # ---------------- PCA Plot ----------------
            if HAS_SKLEARN and has_two_final and len(report_metrics) >= 2:
                st.markdown("### 🧬 Principal Component Analysis (PCA)")
                clean = df[report_metrics + ["Dataset_Source"]].dropna()
                if len(clean) > 5:
                    X = StandardScaler().fit_transform(clean[report_metrics].to_numpy())
                    pca = PCA(n_components=2)
                    scores = pca.fit_transform(X)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ds_colors = ds_color_map if len(datasets_final) == 2 else dict(zip(datasets_final, sns.color_palette(color_palette, len(datasets_final))))
                    for ds in datasets_final:
                        mask = (clean["Dataset_Source"] == ds).to_numpy()
                        ax.scatter(scores[mask, 0], scores[mask, 1], alpha=0.6, s=35, color=ds_colors.get(ds, "#333"), label=ds, edgecolors="none")
                    
                    if show_legend:
                        ax.legend(frameon=True)
                    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                    if scores.shape[1] > 1:
                        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                    sns.despine()
                    st.pyplot(fig, use_container_width=False)

            # ---------------- Radar Chart ----------------
            if has_two_final and len(report_metrics) >= 3:
                st.markdown("### 🕸️ Morphological Radar Profile")
                clean = df[report_metrics + ["Dataset_Source"]].dropna()
                z = clean.copy()
                for m in report_metrics:
                    mu, sigma = clean[m].mean(), clean[m].std()
                    z[m] = (clean[m] - mu) / sigma if sigma > 0 else 0.0
                profile = z.groupby("Dataset_Source")[report_metrics].mean()
                
                lo, hi = profile.to_numpy().min(), profile.to_numpy().max()
                span = max(hi - lo, 1e-6)
                series = {ds: ((profile.loc[ds] - lo) / span).tolist() for ds in profile.index if ds in datasets_final}
                
                ds_colors = ds_color_map if len(datasets_final) == 2 else dict(zip(datasets_final, sns.color_palette(color_palette, len(datasets_final))))
                colors = [ds_colors.get(ds, "#000") for ds in series]
                
                fig, ax = radar_chart(report_metrics, series, colors=colors, figsize=(5.5, 5.5), show_legend=show_legend)
                st.pyplot(fig, use_container_width=False)

            # ---------------- Workbench Pair Scatters ----------------
            wb_pairs_available = [(x, y) for x, y in st.session_state.wb_pairs if x in num_cols and y in num_cols]
            if wb_pairs_available:
                st.markdown("### 📈 Pairwise Relationships (Workbench Pairs)")
                cols_scat = st.columns(2)
                for i, (x_var, y_var) in enumerate(wb_pairs_available):
                    with cols_scat[i % 2]:
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.scatterplot(data=df, x=x_var, y=y_var, hue="Dataset_Source" if has_two_final else None, palette=ds_color_map if has_two_final else None, alpha=0.5, ax=ax, legend=show_legend if has_two_final else False)
                        sns.despine()
                        st.pyplot(fig, use_container_width=False)
                        
                # ---------------- Correlation forest for workbench pairs ----------------
                st.markdown("### 🔗 Significant Correlations (from Workbench)")
                corr_rows_final = []
                skipped_pairs = []

                for x_var, y_var in wb_pairs_available:
                    keep_cols = list(dict.fromkeys([x_var, y_var])) + (["Dataset_Source"] if has_two_final else [])
                    if len(keep_cols) < (3 if has_two_final else 2):
                        skipped_pairs.append(f"{x_var} vs {y_var} (same column selected twice)")
                        continue
                    pair_df = df[keep_cols].dropna()

                    for ds in datasets_final:
                        sub = pair_df[pair_df["Dataset_Source"] == ds] if has_two_final else pair_df
                        if len(sub) < 5:
                            continue
                        try:
                            x_arr = sub[x_var].to_numpy().ravel()
                            y_arr = sub[y_var].to_numpy().ravel()
                            if len(x_arr) != len(y_arr) or len(x_arr) < 5:
                                raise ValueError("mismatched or insufficient sample size")
                            r_val, p_val = pearsonr(x_arr, y_arr)
                            boot_r = paired_bootstrap_corr(x_arr, y_arr, "pearson", n_boot_final, rng)
                            if len(boot_r) < 10:
                                continue
                            ci_lo, ci_hi = np.percentile(boot_r, [2.5, 97.5])
                            corr_rows_final.append({"Pair": f"{x_var} vs {y_var}", "Dataset": ds, "R": r_val,
                                                     "CI_Low": ci_lo, "CI_High": ci_hi, "Samples": boot_r, "P_raw": p_val})
                        except Exception as e:
                            skipped_pairs.append(f"{x_var} vs {y_var} ({ds}): {e}")

                if skipped_pairs:
                    st.caption(f"⚠️ Skipped {len(skipped_pairs)} pair/dataset combination(s): " + "; ".join(skipped_pairs))

                if corr_rows_final:
                    raw_p3 = [r["P_raw"] for r in corr_rows_final]
                    corr_p3 = multipletests(raw_p3, method="bonferroni")[1] if HAS_STATSMODELS else raw_p3
                    for r, cp in zip(corr_rows_final, corr_p3):
                        r["Stars"] = stars_from_p(cp)
                    fig2 = render_forest_plot(corr_rows_final, datasets_final, has_two_final, "Pearson", show_legend=show_legend)
                    st.pyplot(fig2, use_container_width=False)

            st.divider()
            
            # --- CUSTOM PDF EXPORT BUTTON ---
            col_print, col_tip = st.columns([1, 4])
            with col_print:
                components.html(
                    """
                    <script>
                    function printDoc() {
                        window.parent.print();
                    }
                    </script>
                    <style>
                    .btn {
                        background-color: #2E75B6;
                        color: white;
                        padding: 10px 16px;
                        border: none;
                        border-radius: 6px;
                        cursor: pointer;
                        font-family: sans-serif;
                        font-size: 14px;
                        font-weight: 600;
                        width: 100%;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        transition: background-color 0.2s ease;
                    }
                    .btn:hover { background-color: #1F5A8F; }
                    </style>
                    <button class="btn" onclick="printDoc()">🖨️ Export PDF</button>
                    """,
                    height=45
                )
            with col_tip:
                st.caption("💡 **Tip:** Use the 'Export PDF' button (or `Ctrl+P` / `Cmd+P`) to save this entire report layout exactly as seen. "
                           "Every figure above is generated fresh from your current data and settings.")

else:
    st.info("⬆️ Upload a primary CSV to begin.")