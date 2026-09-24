# Morphometric Statistics & Data Analysis Pipeline

An interactive Python/Streamlit application for **statistical analysis, visualization, and interpretation of quantitative biological datasets**.

The pipeline was originally developed for morphometric analysis of microscopy-derived datasets, particularly measurements obtained from **SBF-SEM reconstructions of peripheral nerves and neuronal ultrastructure**, but most modules can also be used with other quantitative CSV datasets.

The application combines descriptive statistics, hypothesis testing, correlation analysis, effect-size estimation, bootstrapping, multivariate analysis, allometric scaling, and publication-oriented visualization within a single graphical interface.

---

## Project status

> **Research prototype / work in progress**
>
> The application is functional and is currently used for research analysis, but it remains under active development.

This repository is made public primarily to demonstrate my work in:

- computational bioimaging;
- quantitative morphometry;
- biological data analysis;
- statistical workflow development;
- scientific visualization;
- reproducible analysis tools.

The current version should not be considered production-ready statistical software.

The interface, analysis modules, dependencies, and internal workflow may change as development continues.

**No open-source license is currently provided.**

---

# Main features

## CSV-based analysis

The application accepts quantitative data stored in CSV files.

It supports:

- one primary dataset;
- an optional second dataset for group comparisons;
- custom dataset labels;
- automatic identification of numeric variables;
- interactive metric selection;
- side-by-side analysis of experimental groups.

A typical use case is the comparison of two experimental groups such as:

```text
WT vs KO
Control vs Treatment
Region A vs Region B
Condition 1 vs Condition 2
```

The group labels are fully customizable.

---

# Analysis workflow

The application currently contains **10 analysis modules**:

1. Overview & Group Distributions
2. Statistical Comparison
3. Correlation Structure
4. Myelin Biology Deep-Dive
5. Multivariate & Clustering
6. Descriptive Statistics
7. StatAll Comprehensive
8. Scatter Explorer
9. Allometric Scaling
10. Final Summary Report

The modules can be used independently or as part of a guided analysis workflow.

---

# 1. Overview & Group Distributions

Provides visual exploration of selected variables across groups.

Available plot types include:

- Violin plots
- Boxplots
- Bar plots
- Raincloud plots

The interface allows selection of:

- grouping variable;
- quantitative metrics;
- plotting style;
- color palette;
- optional IQR/median annotations.

For variables with large numbers of categories, the application provides controls to prevent unreadable plots.

---

# 2. Statistical Comparison

Performs statistical comparison between two uploaded datasets.

The pipeline evaluates the data distribution and selects an appropriate test based on statistical assumptions.

Supported tests include:

- Student's t-test
- Welch's t-test
- Mann-Whitney U test
- Paired t-test
- Wilcoxon signed-rank test

For larger multi-group analyses, the underlying statistical framework also supports:

- one-way ANOVA;
- Kruskal-Wallis testing.

The workflow includes:

- Shapiro-Wilk normality assessment;
- Levene's variance test;
- automated test selection;
- sample-size reporting;
- significance reporting.

---

## Animal/sample-level analysis

Biological datasets frequently contain multiple measurements originating from the same experimental unit.

For example:

```text
Mouse 1
 ├── Axon 1
 ├── Axon 2
 ├── Axon 3
 └── Axon 4
```

Treating each axon as an independent biological replicate can result in pseudoreplication.

The application therefore allows the user to specify an:

```text
Animal / Sample ID
```

When provided, multiple observations from the same experimental unit can be aggregated before statistical testing.

The resulting test is then performed using the number of independent animals/samples rather than the total number of individual measurements.

---

## Paired / matched analysis

The application also supports matched experimental designs.

A shared paired identifier can be provided when measurements correspond between datasets, for example:

```text
before / after
left / right
control / treatment from the same animal
repeated measurement
```

The application then performs an appropriate paired statistical test.

---

# 3. Correlation Structure

Provides detailed analysis of relationships between quantitative variables.

Supported correlation methods:

- Pearson correlation
- Spearman correlation

The application generates:

- correlation matrices;
- scatter plots;
- correlation coefficients;
- raw p-values;
- corrected p-values;
- significance indicators.

For two-dataset analyses, correlations can be calculated either:

```text
Pooled across both groups
```

or:

```text
Separately within each group
```

This is useful for avoiding correlations that are driven primarily by differences between experimental groups rather than by relationships within the groups themselves.

---

## Multiple-testing correction

When many variable pairs are tested simultaneously, the application calculates:

- raw p-values;
- Benjamini-Hochberg False Discovery Rate (FDR);
- Bonferroni-corrected p-values.

The user can choose which criterion determines statistical significance.

---

## Bootstrap correlation analysis

Selected correlations can be evaluated using bootstrap resampling.

The workflow generates:

- Pearson or Spearman correlation coefficient;
- bootstrap distribution;
- 95% bootstrap confidence interval;
- raw p-value;
- corrected p-value;
- sample size.

Results can be visualized as forest plots.

---

# 4. Myelin Biology Deep-Dive

A specialized analysis module developed for peripheral nerve morphometry.

The module currently contains several analysis tabs.

## G-ratio vs diameter

Plots the relationship between:

```text
Axon diameter
```

and:

```text
G-ratio
```

with independent regression fits for experimental groups.

---

## Binned G-ratio analysis

Allows diameter measurements to be grouped into bins and examines how G-ratio changes across axon-size ranges.

This can help visualize size-dependent changes in myelination.

---

## ECDF analysis

Provides empirical cumulative distribution function (ECDF) visualization.

Distribution differences can be evaluated using the:

```text
Kolmogorov-Smirnov test
```

---

## Effect-size analysis

Quantifies differences between two groups using:

```text
Cohen's d
```

The application also performs bootstrap resampling to estimate:

```text
95% confidence intervals
```

for effect sizes.

Results are visualized using forest plots.

---

## Radar / morphological profile

Multiple morphometric measurements can be combined into a normalized radar plot.

Metrics are standardized before plotting, allowing multidimensional morphological differences between groups to be visualized on a common scale.

---

# 5. Multivariate & Clustering

Provides exploratory multivariate analysis of biological datasets.

Current modules include:

- Principal Component Analysis (PCA)
- Hierarchical clustering
- ROC / AUC analysis
- MANOVA

---

## Principal Component Analysis

Selected variables are standardized and projected into principal-component space.

This allows visualization of multidimensional differences between experimental groups and can help identify whether several correlated measurements together form distinct morphological profiles.

---

## Hierarchical clustering

Provides exploratory clustering of samples based on selected quantitative metrics.

This can be used to examine whether samples naturally group according to experimental condition or morphological phenotype.

---

## ROC / AUC analysis

For two-group datasets, individual metrics can be evaluated for their ability to distinguish between groups.

The module calculates:

```text
Receiver Operating Characteristic (ROC)
Area Under the Curve (AUC)
```

for selected variables.

---

## MANOVA

Multivariate Analysis of Variance can be used to determine whether experimental groups differ jointly across several correlated measurements.

This provides an alternative to interpreting many separate univariate tests independently.

---

# 6. Descriptive Statistics

Provides summary statistics for selected variables.

Current outputs include:

- sample count;
- mean;
- standard deviation.

This module can be applied independently to either dataset when two datasets are loaded.

---

# 7. StatAll Comprehensive

Provides a broader statistical audit of selected variables.

For each metric, the application calculates:

- count;
- mean;
- median;
- standard deviation;
- standard error;
- skewness;
- kurtosis;
- Shapiro-Wilk normality p-value.

The module also generates:

- distribution plots;
- QQ plots;
- normality assessment;
- consolidated summary tables.

---

# 8. Scatter Explorer

Provides flexible exploratory visualization of relationships between selected quantitative variables.

The application can generate pairwise scatter plots for multiple selected variables.

When two datasets are loaded, points can be colored according to experimental group.

Pairs stored in the Significance Workbench can also be selectively visualized.

---

# 9. Allometric Scaling

Provides log-log regression analysis for quantitative biological scaling relationships.

The workflow analyzes relationships of the general form:

```text
Y = C × X^k
```

which becomes:

```text
log(Y) = log(C) + k × log(X)
```

The fitted slope:

```text
k
```

represents the scaling exponent.

This module is useful for examining relationships such as:

```text
Surface Area vs Volume
Organelle Size vs Cell Size
Structural Size vs Axon Size
```

Only positive, non-missing observations are included in log-log regression.

When two datasets are loaded, scaling relationships can be examined independently for each group.

---

# 10. Final Summary Report

The final module consolidates selected results from the analysis workflow.

The report can include:

- dataset information;
- key statistical findings;
- significant variables;
- selected correlations;
- effect sizes;
- bootstrap confidence intervals;
- summary figures;
- selected metrics from the Significance Workbench.

The generated report can be saved as a PDF using the built-in browser print/export function.

---

# Significance Workbench

The application contains a persistent **Significance Workbench**.

Statistically interesting metrics and variable pairs identified during analysis can be added to the Workbench.

These selections can then be reused across other analysis modules without manually reselecting variables.

For example:

```text
Statistical Comparison
        ↓
Significant metrics
        ↓
Significance Workbench
        ↓
Correlation Analysis
        ↓
Bootstrap Forest Plot
        ↓
Final Report
```

This allows the analysis workflow to move progressively from exploratory statistics toward focused interpretation.

---

# Outlier detection

Optional outlier screening is available before downstream analysis.

Two approaches are currently implemented.

## Robust MAD-based detection

Uses the:

```text
Median Absolute Deviation
```

and a modified Z-score to identify extreme observations within individual variables.

The threshold can be adjusted interactively.

---

## Multivariate Mahalanobis detection

Detects observations that are unusual across several variables simultaneously.

This is useful when an observation may not appear extreme for any single metric but represents an unusual combination of measurements.

Detected observations can be temporarily excluded from downstream analyses.

The original dataset can be restored at any time.

---

# Derived morphometric parameters

The application includes an optional calculation workflow for deriving additional structural measurements from microscopy-derived variables.

Depending on the supplied columns, calculated metrics can include:

```text
Length
Inner diameter
Volume-based G-ratio
Surface-area-based G-ratio
Surface-area-to-volume ratio
Sphericity
Estimated fiber diameter
Myelin thickness
```

For example:

```text
Length = Plane Count × Z-Step
```

The calculation module was developed for morphometric analysis of reconstructed nerve fibers.

---

# Statistical methods

Depending on the selected module and dataset structure, the application currently implements methods including:

### Distribution testing

```text
Shapiro-Wilk test
Levene's test
```

### Parametric comparisons

```text
Student's t-test
Welch's t-test
Paired t-test
One-way ANOVA
```

### Non-parametric comparisons

```text
Mann-Whitney U
Wilcoxon signed-rank
Kruskal-Wallis
Kolmogorov-Smirnov
```

### Correlation

```text
Pearson correlation
Spearman correlation
```

### Multiple-testing correction

```text
Benjamini-Hochberg FDR
Bonferroni correction
```

### Resampling

```text
Bootstrap confidence intervals
```

### Effect size

```text
Cohen's d
Bootstrap Cohen's d
```

### Multivariate analysis

```text
Principal Component Analysis
Hierarchical clustering
ROC / AUC
MANOVA
```

### Scaling analysis

```text
Log-log linear regression
Allometric scaling
```

---

# Visualization

The application generates a range of scientific plots, including:

- violin plots;
- boxplots;
- raincloud plots;
- bar plots;
- strip plots;
- QQ plots;
- scatter plots;
- correlation matrices;
- heatmaps;
- bootstrap forest plots;
- effect-size forest plots;
- ECDF plots;
- radar plots;
- ROC curves;
- PCA plots;
- allometric scaling plots.

Plot appearance can be adjusted through the Streamlit sidebar.

---

# Technologies

The application is built primarily using:

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **SciPy**
- **Matplotlib**
- **Seaborn**
- **Statsmodels**
- **scikit-learn**

---

# Repository structure

```text
morphometric-statistics-pipeline/
│
├── app.py
│   Main Streamlit application containing
│   the analysis and visualization workflow.
│
├── run_stats.sh
│   Launcher for the local Streamlit application.
│
├── requirements.txt
│   Python dependencies.
│
├── .gitignore
│   Excludes local environments, research data,
│   generated outputs, and temporary files.
│
└── README.md
```

Research datasets are intentionally **not included** in this repository.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Nochebald/morphometric-statistics-pipeline.git
cd morphometric-statistics-pipeline
```

---

## 2. Create a virtual environment

```bash
python3 -m venv env
```

Activate it:

```bash
source env/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the full multivariate-analysis functionality, including PCA and ROC/AUC analysis, **scikit-learn** is required.

If it is not included in the current requirements file:

```bash
pip install scikit-learn
```

---

# Running the application

Make the launcher executable if necessary:

```bash
chmod +x run_stats.sh
```

Then start the application:

```bash
./run_stats.sh
```

Alternatively:

```bash
source env/bin/activate
python -m streamlit run app.py
```

Streamlit will display a local URL similar to:

```text
http://localhost:8501
```

and normally opens the application automatically in the browser.

---

# Stopping the application

Press:

```text
Ctrl+C
```

in the terminal where Streamlit is running.

---

# Input data

The application currently expects:

```text
CSV
```

files.

At least one dataset must be uploaded.

A second CSV can optionally be supplied for direct group comparisons.

Example:

```text
Primary dataset: WT.csv
Secondary dataset: KO.csv
```

The application automatically detects numeric columns that can be used for statistical analysis.

---

# Example workflow

A typical biological analysis may follow:

```text
Upload CSV datasets
        │
        ▼
Inspect distributions
        │
        ▼
Optional outlier screening
        │
        ▼
Two-group statistical comparison
        │
        ▼
Add significant variables to Workbench
        │
        ▼
Correlation analysis
        │
        ▼
Bootstrap confidence intervals
        │
        ▼
Effect-size analysis
        │
        ▼
Multivariate analysis
        │
        ▼
Allometric scaling
        │
        ▼
Generate final report
```

---

# Current limitations

This project remains under active development.

Current limitations include:

- primarily tested using my own biological morphometry datasets;
- no automated test suite yet;
- no standalone installer;
- input is currently primarily CSV-based;
- some modules require specific dataset structures;
- some myelin-specific analyses assume appropriate diameter and G-ratio variables;
- statistical interpretation remains the responsibility of the user;
- some advanced modules require optional dependencies;
- interface and internal organization may change between versions.

---

# Planned development

Possible future improvements include:

- improved automatic data validation;
- broader dataset-format support;
- automated export of statistical tables;
- improved report generation;
- automated figure export;
- additional effect-size measures;
- expanded mixed-effects / hierarchical statistical models;
- improved longitudinal and paired-data support;
- improved handling of biological replication;
- additional regression models;
- additional documentation;
- automated testing;
- example datasets;
- interface refinements.

---

# Intended use

This project is intended primarily for:

- exploratory biological data analysis;
- microscopy-derived morphometric datasets;
- quantitative image-analysis outputs;
- research statistics;
- scientific visualization;
- analysis workflow development;
- preparation of figures and summary statistics.

The application is designed to assist researchers with analysis but does not replace statistical judgment or appropriate experimental design.

It is **not intended for clinical or diagnostic use**.

---

# Development background

The pipeline was developed during quantitative analysis of 3D microscopy datasets, including SBF-SEM reconstructions of peripheral nerve structures.

The original analysis required repeatedly combining:

- descriptive statistics;
- group comparisons;
- correlation analysis;
- multiple-testing correction;
- bootstrap confidence intervals;
- effect-size estimation;
- morphometric scaling;
- scientific visualization.

The application was developed to bring these analyses into a single local interface and reduce the need to repeatedly write separate analysis scripts for each dataset.

---

# Screenshots

Screenshots and example outputs will be added as development continues.

Possible examples include:

### Group distributions

```text
Violin / raincloud plots
```

### Statistical comparison

```text
WT vs KO comparison
```

### Correlation structure

```text
Correlation matrix + bootstrap forest plot
```

### Myelin analysis

```text
G-ratio vs diameter
```

### Multivariate analysis

```text
PCA / clustering
```

### Scaling

```text
Log-log allometric regression
```

### Summary report

```text
Integrated analysis report
```

---

# Author

**Vitaly Borisovs, Ph.D.**

Research Associate  
University of Milan-Bicocca

Research interests:

- computational bioimaging;
- SBF-SEM;
- quantitative morphometry;
- biological image analysis;
- statistics;
- deep learning;
- 3D reconstruction.

GitHub:  
https://github.com/Nochebald

Portfolio:  
https://nochebald.github.io/

---

# License and reuse

**No open-source license is currently provided for this repository.**

The source code is publicly visible primarily for research demonstration and portfolio purposes.

The project remains under active development and is not currently released for unrestricted reuse, redistribution, or incorporation into other software projects.

Please contact the author regarding potential reuse or collaboration.

Copyright © 2026 Vitaly Borisovs.
