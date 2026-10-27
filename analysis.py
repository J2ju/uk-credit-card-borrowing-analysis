"""
================================================================================
MACROECONOMIC DETERMINANTS OF CREDIT CARD BORROWING IN THE UNITED KINGDOM
A monthly time-series analysis, 2000-2026

Jude Adjei | MSc Financial Technology and Business Analytics | University of Chester
================================================================================

WHAT THIS SCRIPT DOES
    Builds every variable from the raw published series, estimates the model,
    and runs each test reported in Chapter 4. Running it top to bottom
    reproduces the dissertation.

INPUT
    regression_ready_data.csv  - six monthly series, January 2000 to July 2026
                                 Row 1 holds the column names.
                                 Row 2 holds the publisher's series codes and
                                 is skipped on loading.

OUTPUT
    Printed to the console and written to analysis_results.txt

STAGES
    0   Setup and configuration
    1   Load the raw data
    2   Build the dependent variable
    3   Build the explanatory variables
    4   Mark the three crisis periods
    5   Assemble the estimation sample
    6   Descriptive statistics
    7   Stationarity testing
    8   Multicollinearity assessment
    9   The main regression
    10  Rank the determinants
    11  Residual diagnostics
    12  Delayed reaction - do households respond with a lag?
    13  Structural stability - did the relationships survive the crises?
    14  Seasonality - is anything left after seasonal adjustment?
    15  Robustness checks
    16  Summary and file output
================================================================================
"""

# ==============================================================================
# STAGE 0 - SETUP AND CONFIGURATION
# Everything that can be changed without touching the analysis lives here.
# ==============================================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import (acorr_ljungbox, het_breuschpagan,
                                          breaks_cusumolsresid)

DATA_FILE = "regression_ready_data.csv"
RESULTS_FILE = "analysis_results.txt"

# Newey-West lag length. Twelve months is the convention for monthly data.
HAC_LAGS = 12

# Crisis windows. Each date is taken from a documented event, never from
# inspecting the residuals - choosing break dates by looking at where the model
# misses would mean fitting the specification to the data it is meant to test.
CRISIS_WINDOWS = {
    "gfc":   ("2007-08-01", "2009-06-30"),   # interbank markets seize -> trough
    "covid": ("2020-03-01", "2021-06-30"),   # first lockdown -> restrictions lifted
    "col":   ("2021-10-01", "2023-12-31"),   # energy price cap rise -> inflation falls back
}

# The two months whose growth is far outside anything else in the series.
# Used only to ask whether they drive the results, never silently removed.
EXTREME_MONTHS = pd.to_datetime(["2010-01-01", "2003-02-01"])

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

OUTPUT_LINES = []


def say(*parts):
    """Print to screen and keep a copy for the results file."""
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    OUTPUT_LINES.append(line)


def heading(title):
    say("\n" + "=" * 86)
    say(title)
    say("=" * 86)


# ==============================================================================
# STAGE 1 - LOAD THE RAW DATA
# The file carries two header rows: names, then the publisher's series codes.
# Row 2 is skipped. Dates arrive as M/D/YYYY text and are converted here.
# ==============================================================================

heading("STAGE 1 - LOADING THE RAW DATA")

raw = pd.read_csv(DATA_FILE, skiprows=[1], encoding="latin-1")

# Map the published column names onto short working names. Keeping this mapping
# visible means anyone can trace a working name back to the series it came from.
COLUMN_NAMES = {
    "Month":                              "month",
    "Credit card balances outstanding":   "balances",        # LPMVZRJ, £m, SA
    "Representative card interest rate":  "card_rate",       # IUMCCTL, %
    "Bank Rate, monthly average":         "bank_rate",       # IUMABEDR, %
    "CPI annual inflation rate":          "inflation",       # D7G7, %
    "Unemployment rate, aged 16+":        "unemployment",    # MGSX, %, SA
    "Claimant count":                     "claimant_count",  # BCJD, 000s, SA
}
raw = raw.rename(columns=COLUMN_NAMES)

raw["month"] = pd.to_datetime(raw["month"], format="%m/%d/%Y")
raw = raw.sort_values("month").reset_index(drop=True)

# Every series except the date should be numeric. Anything that will not convert
# becomes NaN and is reported below rather than failing silently.
for column in ["balances", "card_rate", "bank_rate", "inflation",
               "unemployment", "claimant_count"]:
    raw[column] = pd.to_numeric(raw[column], errors="coerce")

say(f"  Rows loaded        : {len(raw)}")
say(f"  Period covered     : {raw['month'].min():%B %Y} to {raw['month'].max():%B %Y}")
say(f"  Duplicate months   : {raw['month'].duplicated().sum()}")
say("\n  Missing values by series")
for column, count in raw.isna().sum().items():
    if column != "month":
        say(f"    {column:<18s} {count}")

say("\n  Range of each raw series")
say(raw.describe().T[["min", "max", "mean"]].round(3).to_string())


# ==============================================================================
# STAGE 2 - BUILD THE DEPENDENT VARIABLE
#
# The thing to be explained is how fast card borrowing grew this month.
#
# Modelling the balance itself would risk a spurious regression: two series that
# merely trend together produce convincing test statistics with no relationship
# between them (Granger & Newbold, 1974). The log difference removes the trend
# and gives, near enough, the percentage change from one month to the next.
#
#     card_growth(t) = [ ln(balances t) - ln(balances t-1) ] x 100
# ==============================================================================

heading("STAGE 2 - BUILDING THE DEPENDENT VARIABLE")

data = raw.copy()
data["log_balances"] = np.log(data["balances"])
data["card_growth"] = (data["log_balances"] - data["log_balances"].shift(1)) * 100

# Cross-check against a plain percentage change. The two agree closely for
# ordinary months and diverge only where the move is large, which is expected:
# the log difference understates a big rise and overstates a big fall.
plain_percent = data["balances"].pct_change() * 100
gap = (data["card_growth"] - plain_percent).abs()

say(f"  card_growth built from log differences of balances")
say(f"  Agreement with a plain percentage change:")
say(f"    median gap  {gap.median():.4f} percentage points")
say(f"    largest gap {gap.max():.4f} percentage points "
    f"(in {data.loc[gap.idxmax(), 'month']:%B %Y}, the largest monthly move)")
say(f"  Mean monthly growth : {data['card_growth'].mean():.4f} %")
say(f"  Range               : {data['card_growth'].min():.4f} % to "
    f"{data['card_growth'].max():.4f} %")


# ==============================================================================
# STAGE 3 - BUILD THE EXPLANATORY VARIABLES
#
# Three of the four determinants enter as month-on-month changes. Borrowers do
# not respond to a rate being 22 per cent - they have lived with that for years -
# they respond to it moving.
#
# Inflation is the exception. CPI inflation is already a rate of change, so
# differencing it again would give the acceleration of prices, which is not what
# the theory concerns. It enters as published.
#
# The lagged dependent variable is added because card borrowing has momentum.
# Left out, that persistence sits in the error term and disturbs the diagnostics.
# ==============================================================================

heading("STAGE 3 - BUILDING THE EXPLANATORY VARIABLES")

data["change_card_rate"] = data["card_rate"] - data["card_rate"].shift(1)
data["change_bank_rate"] = data["bank_rate"] - data["bank_rate"].shift(1)
data["change_unemployment"] = data["unemployment"] - data["unemployment"].shift(1)
data["change_claimant"] = data["claimant_count"] - data["claimant_count"].shift(1)
data["last_month_growth"] = data["card_growth"].shift(1)

say("  change_card_rate     first difference of the representative card APR")
say("  change_bank_rate     first difference of Bank Rate")
say("  inflation            CPI annual rate, used as published (already a change)")
say("  change_unemployment  first difference of the LFS unemployment rate")
say("  change_claimant      first difference of the claimant count (robustness only)")
say("  last_month_growth    card_growth lagged one month")


# ==============================================================================
# STAGE 4 - MARK THE THREE CRISIS PERIODS
#
# The sample spans twenty-six years containing three episodes in which the market
# did not behave normally. An indicator variable lets the average level of
# borrowing growth differ during each episode while leaving the estimated
# relationships with the determinants alone.
#
# Without them, distortions belonging to a pandemic are absorbed into the
# coefficients on the determinants and mistaken for economics.
# ==============================================================================

heading("STAGE 4 - MARKING THE CRISIS PERIODS")

for name, (start, end) in CRISIS_WINDOWS.items():
    inside = (data["month"] >= start) & (data["month"] <= end)
    data[name] = inside.astype(int)

# One combined flag, used by the stability tests in Stage 13.
data["crisis"] = ((data["gfc"] + data["covid"] + data["col"]) > 0).astype(int)

for name, (start, end) in CRISIS_WINDOWS.items():
    say(f"  {name:<6s} {start[:7]} to {end[:7]}   {int(data[name].sum()):3d} months")
say(f"  {'crisis':<6s} any of the three          {int(data['crisis'].sum()):3d} months")


# ==============================================================================
# STAGE 5 - ASSEMBLE THE ESTIMATION SAMPLE
#
# Differencing costs observations: the first month has no predecessor, and the
# lagged dependent variable costs one more. Dropping the incomplete rows once,
# here, guarantees every test below runs on identical observations - which the
# joint tests in Stages 12, 13 and 14 require in order to be valid.
# ==============================================================================

heading("STAGE 5 - ASSEMBLING THE ESTIMATION SAMPLE")

DETERMINANTS = ["change_card_rate", "change_bank_rate",
                "inflation", "change_unemployment"]
CONTROLS = ["gfc", "covid", "col", "last_month_growth"]
MODEL_TERMS = DETERMINANTS + CONTROLS

LABELS = {
    "const":               "Constant",
    "change_card_rate":    "Card interest rate (d)",
    "change_bank_rate":    "Bank Rate (d)",
    "inflation":           "CPI inflation",
    "change_unemployment": "Unemployment rate (d)",
    "change_claimant":     "Claimant count (d)",
    "gfc":                 "GFC 2007-09",
    "covid":               "COVID-19 2020-21",
    "col":                 "Cost of living 2021-23",
    "last_month_growth":   "Lagged dependent variable",
    "extreme":             "Extreme-month marker",
    "january":             "January marker",
}

df = data.dropna(subset=["card_growth"] + MODEL_TERMS).reset_index(drop=True)

say(f"  Rows in the raw file        : {len(raw)}")
say(f"  Rows after differencing     : {len(df)}")
say(f"  Estimation sample           : {df['month'].min():%B %Y} to "
    f"{df['month'].max():%B %Y}")
say(f"  Crisis months in sample     : {int(df['crisis'].sum())}")
say(f"  Normal months in sample     : {int((1 - df['crisis']).sum())}")


# ==============================================================================
# HELPER FUNCTIONS
# Defined once here so that every stage below estimates and reports identically.
# ==============================================================================

def fit(y, X):
    """Fit by OLS and return two results objects from the same model.

    The first carries Newey-West standard errors, valid when the errors are
    unevenly sized and correlated across months - which monthly financial data
    always are. The second carries plain standard errors and is used only where
    a statistic requires them, such as AIC.

    The coefficients are identical in both. Only the uncertainty differs.
    """
    X = sm.add_constant(X, has_constant="add")
    robust = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    plain = sm.OLS(y, X).fit()
    return robust, plain


def joint_test(model, term_names):
    """Test whether a group of coefficients is jointly zero.

    The test uses the model's own covariance matrix. Where that model was fitted
    with Newey-West errors, so is this test - which matters, because a joint test
    run on plain residual sums of squares quietly reverts to assuming even,
    independent errors and will disagree with the coefficient tests above it.
    """
    all_names = list(model.params.index)
    positions = [all_names.index(name) for name in term_names]
    result = model.f_test(np.eye(len(all_names))[positions])
    return float(np.squeeze(result.fvalue)), float(np.squeeze(result.pvalue))


def report(robust, plain, y, X, title, show_betas=True):
    """Print a regression table, with diagnostics underneath."""
    say(f"\n{title}")
    say(f"  N = {int(robust.nobs)}   R2 = {robust.rsquared:.4f}   "
        f"adj R2 = {robust.rsquared_adj:.4f}   "
        f"F = {robust.fvalue:.4f} (p = {robust.f_pvalue:.6f})")

    header = f"  {'Variable':<28s}{'b':>11s}{'SE':>10s}{'t':>9s}{'p':>9s}"
    if show_betas:
        header += f"{'beta':>10s}"
    say(header)

    sd_y = y.std()
    for name in robust.params.index:
        stars = ("***" if robust.pvalues[name] < 0.01 else
                 "**" if robust.pvalues[name] < 0.05 else
                 "*" if robust.pvalues[name] < 0.10 else "")
        beta = ""
        if show_betas and name != "const":
            beta = f"{robust.params[name] * X[name].std() / sd_y:>10.4f}"
        say(f"  {LABELS.get(name, name):<28s}{robust.params[name]:>11.4f}"
            f"{robust.bse[name]:>10.4f}{robust.tvalues[name]:>9.3f}"
            f"{robust.pvalues[name]:>9.4f}{beta}  {stars}")

    ljung = acorr_ljungbox(plain.resid, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
    breusch = het_breuschpagan(plain.resid, plain.model.exog)[1]
    say(f"  Durbin-Watson {durbin_watson(plain.resid):.4f} | "
        f"Ljung-Box(12) p {ljung:.4f} | Breusch-Pagan p {breusch:.4f} | "
        f"Jarque-Bera p {jarque_bera(plain.resid)[1]:.6f}")
    return robust


# ==============================================================================
# STAGE 6 - DESCRIPTIVE STATISTICS
#
# Read before modelling anything. Skewness near zero means symmetric; excess
# kurtosis near zero means normally tailed. Card growth will return large values
# on both, driven by two months - worth knowing now rather than discovering it
# in the diagnostics at Stage 11.
# ==============================================================================

heading("STAGE 6 - DESCRIPTIVE STATISTICS (estimation sample)")

described = pd.DataFrame({
    "Credit card balances (£m)": df["balances"],
    "Growth in balances (%)":    df["card_growth"],
    "Card interest rate (%)":    df["card_rate"],
    "Bank Rate (%)":             df["bank_rate"],
    "CPI inflation (%)":         df["inflation"],
    "Unemployment rate (%)":     df["unemployment"],
    "Claimant count (000s)":     df["claimant_count"],
})
summary = described.describe().T[["mean", "std", "min", "max"]]
summary["skew"] = described.skew()
summary["kurtosis"] = described.kurtosis()
say(summary.round(3).to_string())


# ==============================================================================
# STAGE 7 - STATIONARITY TESTING
#
# The step that prevents a spurious result. A series with a unit root wanders
# without returning to any fixed mean, and regressing one such series on another
# manufactures significance out of nothing.
#
# Augmented Dickey-Fuller null hypothesis: the series HAS a unit root.
# So a SMALL p-value is the good outcome. This is the reverse of most tests and
# is the most common misreading of ADF output.
#
# Expect the levels to fail and the differenced versions to pass - which is
# precisely why they were differenced at Stage 3.
# ==============================================================================

heading("STAGE 7 - AUGMENTED DICKEY-FULLER TESTS (AIC lag selection, constant)")

say(f"  {'Series':<44s}{'ADF':>10s}{'p':>10s}{'lags':>7s}   verdict")

# Every test runs on the estimation sample, so the stationarity evidence refers
# to exactly the observations the model is fitted on.
adf_series = [
    ("Balances, level (£m)",              df["balances"]),
    ("Card interest rate, level (%)",     df["card_rate"]),
    ("Bank Rate, level (%)",              df["bank_rate"]),
    ("CPI inflation, level (%)",          df["inflation"]),
    ("Unemployment rate, level (%)",      df["unemployment"]),
    ("Claimant count, level (000s)",      df["claimant_count"]),
    ("DV: log difference of balances (%)", df["card_growth"]),
    ("Card interest rate, differenced",   df["change_card_rate"]),
    ("Bank Rate, differenced",            df["change_bank_rate"]),
    ("Unemployment rate, differenced",    df["change_unemployment"]),
    ("Claimant count, differenced",       df["change_claimant"]),
]

for label, series in adf_series:
    clean = series.dropna()
    stat, p_value, lags, *_ = adfuller(clean, autolag="AIC")
    verdict = "stationary" if p_value < 0.05 else "NON-stationary"
    say(f"  {label:<44s}{stat:>10.3f}{p_value:>10.4f}{lags:>7d}   {verdict}")


# ==============================================================================
# STAGE 8 - MULTICOLLINEARITY ASSESSMENT
#
# If two explanatory variables carry the same information the regression cannot
# tell which deserves the credit, so it splits the difference and inflates both
# standard errors.
#
# The variance inflation factor asks how much wider a coefficient's variance is
# because of its overlap with the others. A VIF of 1 means no overlap at all;
# the conventional alarm level is 10.
# ==============================================================================

heading("STAGE 8 - MULTICOLLINEARITY")

say("\n  Correlation between the four determinants")
say(df[DETERMINANTS].corr().round(3).to_string())

say("\n  Variance inflation factors (alarm level 10)")
X_for_vif = sm.add_constant(df[MODEL_TERMS], has_constant="add")
for position, name in enumerate(X_for_vif.columns):
    if name == "const":
        continue
    vif = variance_inflation_factor(X_for_vif.values, position)
    flag = "   <-- above 10" if vif > 10 else ""
    say(f"    {LABELS.get(name, name):<28s} VIF = {vif:7.4f}{flag}")


# ==============================================================================
# STAGE 9 - THE MAIN REGRESSION
#
#   card_growth(t) = a + b1.change_card_rate(t) + b2.change_bank_rate(t)
#                      + b3.inflation(t)        + b4.change_unemployment(t)
#                      + b5.gfc(t) + b6.covid(t) + b7.col(t)
#                      + b8.last_month_growth(t) + e(t)
#
# Ordinary least squares gives the right coefficients. Its standard errors assume
# the errors are equally sized and independent of one another; monthly financial
# data honours neither. The Newey-West estimator stays valid under both, leaving
# the coefficients untouched and widening the standard errors to honest values.
# It can only ever make results look weaker, which is why it is the right choice.
#
# b1 to b4 are the determinants under examination. b5 to b8 are controls:
# reported in full, but not interpreted and not ranked.
# ==============================================================================

heading("STAGE 9 - MAIN REGRESSION")

y = df["card_growth"]
X = df[MODEL_TERMS]
main_robust, main_plain = fit(y, X)
report(main_robust, main_plain, y, X,
       "9a. Full specification, Newey-West standard errors")

# Show the correction at work: identical coefficients, different standard errors.
say("\n9b. What the Newey-West correction changes")
say(f"  {'Variable':<28s}{'b':>11s}{'SE plain':>11s}{'SE HAC':>11s}"
    f"{'p plain':>10s}{'p HAC':>10s}")
for name in main_robust.params.index:
    say(f"  {LABELS.get(name, name):<28s}{main_plain.params[name]:>11.4f}"
        f"{main_plain.bse[name]:>11.4f}{main_robust.bse[name]:>11.4f}"
        f"{main_plain.pvalues[name]:>10.4f}{main_robust.pvalues[name]:>10.4f}")
say("  The coefficients are identical. Only the uncertainty around them moves.")


# ==============================================================================
# STAGE 10 - RANK THE DETERMINANTS
#
# A raw coefficient cannot be compared across variables, because a unit means
# something different for each: one percentage point on Bank Rate is not
# comparable to one percentage point of inflation.
#
# Standardising rescales every variable to mean 0 and standard deviation 1, so
# each coefficient answers the same question - if this variable moves by a
# typical amount, how much does borrowing move, in typical amounts of borrowing?
#
#     beta(k) = b(k) x standard deviation of X(k) / standard deviation of Y
#
# This addresses research objective 4. Only the four determinants are ranked.
# ==============================================================================

heading("STAGE 10 - RANKING BY STANDARDISED COEFFICIENT")

sd_y = df["card_growth"].std()
ranked = sorted(
    [(name,
      main_robust.params[name] * df[name].std() / sd_y,
      main_robust.pvalues[name])
     for name in DETERMINANTS],
    key=lambda row: -abs(row[1]))

say(f"  {'Rank':<6s}{'Determinant':<28s}{'beta':>10s}{'p':>10s}")
for position, (name, beta, p_value) in enumerate(ranked, start=1):
    say(f"  {position:<6d}{LABELS[name]:<28s}{beta:>10.4f}{p_value:>10.4f}")


# ==============================================================================
# STAGE 11 - RESIDUAL DIAGNOSTICS
#
# The residuals are what the model failed to explain. Structure remaining in them
# means something has been missed. The four tests do not all point the same way:
#
#   Durbin-Watson   correlation with last month's error   want a value near 2
#   Ljung-Box       correlation at any of twelve lags     want p above 0.05
#   Breusch-Pagan   unequal error variance                want p above 0.05
#   Jarque-Bera     non-normal residuals                  want p above 0.05
#
# Jarque-Bera is expected to fail, driven by the two extreme months. This is not
# fatal: with more than three hundred observations OLS inference holds
# asymptotically without normality, and the corrected standard errors handle the
# rest. Stage 15 tests directly whether those two months drive the results.
# ==============================================================================

heading("STAGE 11 - RESIDUAL DIAGNOSTICS")

residuals = main_plain.resid

dw_stat = durbin_watson(residuals)
ljung_p = acorr_ljungbox(residuals, lags=[12], return_df=True)["lb_pvalue"].iloc[0]
breusch_p = het_breuschpagan(residuals, main_plain.model.exog)[1]
jb_stat, jb_p, skewness, kurtosis = jarque_bera(residuals)

say(f"  Durbin-Watson    {dw_stat:8.4f}                 "
    f"{'clean' if 1.5 < dw_stat < 2.5 else 'CHECK'}")
say(f"  Ljung-Box (12)   p = {ljung_p:.4f}              "
    f"{'pass' if ljung_p > 0.05 else 'FAIL'}")
say(f"  Breusch-Pagan    p = {breusch_p:.4f}              "
    f"{'pass' if breusch_p > 0.05 else 'FAIL'}")
say(f"  Jarque-Bera      p = {jb_p:.6f}            "
    f"{'pass' if jb_p > 0.05 else 'FAIL (expected - see Stage 15)'}")
say(f"                   residual skew {skewness:.3f}, kurtosis {kurtosis:.3f}")


# ==============================================================================
# STAGE 12 - DELAYED REACTION
#
# A rate change does not reach household behaviour in the same month. Borrowers
# take time to notice, lenders take time to reprice, and the statistics are
# published with a lag.
#
# The temptation is to try lag 1, then lag 2, then lag 3, and report whichever
# comes out significant. With enough attempts one always will; that is fishing.
# The honest approach estimates every lag together and tests them as a block.
#
# A three-month lag needs three earlier months to exist, so the sample shortens.
# Every model compared below runs on that same shorter sample - otherwise the
# joint test is invalid.
#
# This addresses research objective 2.
# ==============================================================================

heading("STAGE 12 - DELAYED REACTION (lags of zero to three months)")

lagged = df.copy()
for rate in ["change_card_rate", "change_bank_rate"]:
    for lag in range(4):
        lagged[f"{rate}_L{lag}"] = lagged[rate].shift(lag)

# --- 12a: one lag at a time, to see the pattern --------------------------------
say(f"\n12a. Each lag estimated on its own")
say(f"  {'Lag':<6s}{'BankRate b':>13s}{'p':>9s}{'CardRate b':>13s}{'p':>9s}"
    f"{'R2':>9s}{'AIC':>10s}")

for lag in range(4):
    columns = ([f"change_card_rate_L{lag}", f"change_bank_rate_L{lag}",
                "inflation", "change_unemployment"] + CONTROLS)
    sample = lagged.dropna(subset=columns)
    robust, plain = fit(sample["card_growth"], sample[columns])
    say(f"  {lag:<6d}{robust.params[f'change_bank_rate_L{lag}']:>13.4f}"
        f"{robust.pvalues[f'change_bank_rate_L{lag}']:>9.4f}"
        f"{robust.params[f'change_card_rate_L{lag}']:>13.4f}"
        f"{robust.pvalues[f'change_card_rate_L{lag}']:>9.4f}"
        f"{robust.rsquared:>9.4f}{plain.aic:>10.2f}")

# --- 12b: all lags together, then tested jointly -------------------------------
ALL_LAG_TERMS = ([f"change_bank_rate_L{lag}" for lag in range(4)] +
                 [f"change_card_rate_L{lag}" for lag in range(4)])
distributed_columns = ALL_LAG_TERMS + ["inflation", "change_unemployment"] + CONTROLS

lag_sample = lagged.dropna(subset=distributed_columns)
lag_robust, lag_plain = fit(lag_sample["card_growth"], lag_sample[distributed_columns])
report(lag_robust, lag_plain, lag_sample["card_growth"],
       lag_sample[distributed_columns],
       "12b. Distributed lag model, both rates at lags 0 to 3", show_betas=False)

say(f"\n12c. Joint significance of each rate across all four lags")
for label, terms in [("Bank Rate", [f"change_bank_rate_L{lag}" for lag in range(4)]),
                     ("Card rate", [f"change_card_rate_L{lag}" for lag in range(4)])]:
    f_stat, p_value = joint_test(lag_robust, terms)
    total = lag_robust.params[terms].sum()
    verdict = "matters" if p_value < 0.05 else "adds nothing"
    say(f"  {label:<12s} lags 0-3:  F = {f_stat:7.4f}   p = {p_value:.4f}   "
        f"sum of coefficients = {total:8.4f}   -> {verdict}")

say("\n  A large p-value here is a finding, not a failure: the delayed response was")
say("  tested for rather than assumed, and there is not one.")


# ==============================================================================
# STAGE 13 - STRUCTURAL STABILITY
#
# The crisis dummies from Stage 4 let the LEVEL of borrowing shift. They do not
# ask whether the RELATIONSHIPS changed - whether a rate rise meant something
# different in April 2020 than in 2004.
#
# Multiplying each determinant by the crisis flag answers that. The interaction
# measures how much a determinant's effect DIFFERS during crisis months.
#
# Watch for an apparent contradiction. The Chow tests below will reject while the
# interaction test does not. That is not a conflict: Chow rejects because the
# INTERCEPT shifts, which is exactly what the three dummies absorb, while the
# interaction test says the SLOPES are stable. Borrowing jumped during the
# crises; the way it responds to its determinants did not change.
# ==============================================================================

heading("STAGE 13 - DID THE RELATIONSHIPS SURVIVE THE CRISES?")

# --- 13a: interaction model ----------------------------------------------------
interacted = df[MODEL_TERMS].copy()
for name in DETERMINANTS:
    interacted[f"{name}_x_crisis"] = df[name] * df["crisis"]
INTERACTION_TERMS = [f"{name}_x_crisis" for name in DETERMINANTS]

crisis_robust, crisis_plain = fit(df["card_growth"], interacted)
report(crisis_robust, crisis_plain, df["card_growth"], interacted,
       "13a. Determinants interacted with the crisis flag", show_betas=False)

f_stat, p_value = joint_test(crisis_robust, INTERACTION_TERMS)
say(f"\n  JOINT TEST  H0: the four slopes are the same in crisis and normal months")
say(f"     F = {f_stat:.4f},  p = {p_value:.4f}  ->  "
    f"{'REJECT: relationships changed' if p_value < 0.05 else 'DO NOT REJECT: relationships are stable'}")

# --- 13b: split sample, as a visual cross-check --------------------------------
say("\n13b. Split-sample estimates")
for label, subset in [("Normal months", df[df["crisis"] == 0]),
                      ("Crisis months", df[df["crisis"] == 1])]:
    split_robust, _ = fit(subset["card_growth"],
                          subset[DETERMINANTS + ["last_month_growth"]])
    say(f"   {label}  (N = {int(split_robust.nobs)}, "
        f"R2 = {split_robust.rsquared:.4f})")
    for name in DETERMINANTS:
        say(f"      {LABELS[name]:<28s} b = {split_robust.params[name]:>9.4f}   "
            f"p = {split_robust.pvalues[name]:.4f}")

# --- 13c: Chow tests at named dates --------------------------------------------
# Written in interaction form rather than by splitting the sample. Splitting
# breaks here: cut at September 2008 and the earlier half has covid and col equal
# to zero in every row, making those columns collinear with the constant.
say("\n13c. Chow tests for a structural break at documented dates")
BREAK_TERMS = DETERMINANTS + ["last_month_growth"]

for label, break_date in [("September 2008 (Lehman)",     "2008-09-01"),
                          ("March 2020 (first lockdown)", "2020-03-01"),
                          ("October 2021 (energy shock)", "2021-10-01")]:
    chow_frame = df[BREAK_TERMS].copy()
    after_break = (df["month"] >= break_date).astype(int)
    chow_frame["after_break"] = after_break
    for name in BREAK_TERMS:
        chow_frame[f"{name}_x_break"] = df[name] * after_break

    chow_robust, _ = fit(df["card_growth"], chow_frame)
    tested = ["after_break"] + [f"{name}_x_break" for name in BREAK_TERMS]
    f_stat, p_value = joint_test(chow_robust, tested)
    say(f"   {label:<30s} F = {f_stat:8.4f},  p = {p_value:.4f}  -> "
        f"{'break' if p_value < 0.05 else 'no break'}")

# --- 13d: CUSUM, which needs no break date --------------------------------------
cusum_stat, cusum_p, _ = breaks_cusumolsresid(
    main_plain.resid, ddof=main_plain.df_model + 1)
say(f"\n13d. CUSUM test of parameter stability")
say(f"     statistic = {cusum_stat:.4f},  p = {cusum_p:.4f}  -> "
    f"{'instability' if cusum_p < 0.05 else 'no evidence of parameter instability'}")


# ==============================================================================
# STAGE 14 - SEASONALITY
#
# The Bank publishes the card series already seasonally adjusted, so in principle
# the festive pattern has gone. In principle.
#
# Adjustment estimates a seasonal shape from the historical record and subtracts
# it. Across twenty-six years that shape may itself have moved - contactless,
# online retail and point-of-sale instalment credit have all changed when people
# reach for a card. A filter fitted over the whole period may not remove a
# pattern that shifted within it. What remains is residual seasonality.
#
# Eleven dummies, not twelve: twelve plus a constant sum to one in every row and
# are perfectly collinear. One month must serve as the baseline.
# ==============================================================================

heading("STAGE 14 - RESIDUAL SEASONALITY")

say("\n14a. Mean growth by calendar month")
by_month = df.groupby(df["month"].dt.month)["card_growth"].agg(
    ["mean", "std", "count"]).round(4)
by_month.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
say(by_month.to_string())

say("\n14b. Joint test of eleven month dummies")
for label, frame in [("Full sample", df),
                     ("Excluding the two extreme months",
                      df[~df["month"].isin(EXTREME_MONTHS)])]:
    frame = frame.reset_index(drop=True)
    month_dummies = pd.get_dummies(frame["month"].dt.month,
                                   prefix="mth", drop_first=True).astype(float)
    seasonal_X = pd.concat([frame[MODEL_TERMS], month_dummies], axis=1)
    seasonal_robust, _ = fit(frame["card_growth"], seasonal_X)
    f_stat, p_value = joint_test(seasonal_robust, list(month_dummies.columns))
    verdict = ("RESIDUAL SEASONALITY PRESENT" if p_value < 0.05
               else "none detected")
    say(f"   {label:<36s} N = {int(seasonal_robust.nobs)},  "
        f"F = {f_stat:.4f},  p = {p_value:.4f}  ->  {verdict}")

# --- 14c: is a high January a real effect, or one unusual year? -----------------
without_extremes = df[~df["month"].isin(EXTREME_MONTHS)].reset_index(drop=True)
without_extremes["january"] = (without_extremes["month"].dt.month == 1).astype(int)
january_robust, january_plain = fit(without_extremes["card_growth"],
                                    without_extremes[MODEL_TERMS + ["january"]])

say("\n14c. A January marker, with the extreme months excluded")
say(f"     b = {january_robust.params['january']:.4f},  "
    f"p = {january_robust.pvalues['january']:.4f}  ->  "
    f"{'significant' if january_robust.pvalues['january'] < 0.05 else 'NOT a simple Christmas spike'}")
say(f"     January mean, full sample        : "
    f"{df[df['month'].dt.month == 1]['card_growth'].mean():.4f} %")
say(f"     January mean, extremes excluded  : "
    f"{without_extremes[without_extremes['month'].dt.month == 1]['card_growth'].mean():.4f} %")
say(f"     All other months                 : "
    f"{without_extremes[without_extremes['month'].dt.month != 1]['card_growth'].mean():.4f} %")
say("     If the second figure sits near the third, one unusual January was doing")
say("     the work, and a comparison of monthly means would have misled.")


# ==============================================================================
# STAGE 15 - ROBUSTNESS CHECKS
#
# A result holding under only one set of choices is not a result. Each check below
# changes exactly one thing and leaves everything else alone.
#
#   15a  Do the two extreme months drive the findings?
#   15b  Does the ranking survive a different labour market measure?
#
# Findings that contradict the main model are reported in full. An examiner
# rewards candour about fragility more than a tidy story.
# ==============================================================================

heading("STAGE 15 - ROBUSTNESS CHECKS")

# --- 15a: a marker for the two extreme months ----------------------------------
df["extreme"] = df["month"].isin(EXTREME_MONTHS).astype(int)
extreme_robust, extreme_plain = fit(df["card_growth"], df[MODEL_TERMS + ["extreme"]])
report(extreme_robust, extreme_plain, df["card_growth"],
       df[MODEL_TERMS + ["extreme"]],
       "15a. Adding a marker for the two extreme months", show_betas=False)

# --- 15b: the claimant count in place of the LFS rate ---------------------------
# The LFS unemployment rate is a survey estimate carrying sampling error. The
# claimant count is an administrative headcount from benefit records - an
# entirely independent source for the same underlying condition.
claimant_terms = (["change_card_rate", "change_bank_rate", "inflation",
                   "change_claimant"] + CONTROLS)
claimant_sample = df.dropna(subset=claimant_terms).reset_index(drop=True)
claimant_robust, claimant_plain = fit(claimant_sample["card_growth"],
                                      claimant_sample[claimant_terms])
report(claimant_robust, claimant_plain, claimant_sample["card_growth"],
       claimant_sample[claimant_terms],
       "15b. Claimant count in place of the LFS unemployment rate")

say("\n  Ranking under this alternative measure")
sd_alt = claimant_sample["card_growth"].std()
alt_determinants = ["change_card_rate", "change_bank_rate",
                    "inflation", "change_claimant"]
alt_ranked = sorted(
    [(name,
      claimant_robust.params[name] * claimant_sample[name].std() / sd_alt,
      claimant_robust.pvalues[name])
     for name in alt_determinants],
    key=lambda row: -abs(row[1]))
for position, (name, beta, p_value) in enumerate(alt_ranked, start=1):
    say(f"   {position}. {LABELS[name]:<28s} beta = {beta:>8.4f}   p = {p_value:.4f}")


# ==============================================================================
# STAGE 16 - SUMMARY AND FILE OUTPUT
# The headline figures, collected for Chapter 4.
# ==============================================================================

heading("STAGE 16 - SUMMARY OF RESULTS")

say(f"  Sample                {df['month'].min():%b %Y} to {df['month'].max():%b %Y}"
    f"   ({len(df)} months)")
say(f"  R squared             {main_robust.rsquared:.4f}")
say(f"  Adjusted R squared    {main_robust.rsquared_adj:.4f}")
say(f"  F statistic           {main_robust.fvalue:.4f}  "
    f"(p = {main_robust.f_pvalue:.6f})")

say("\n  Determinants ranked by standardised coefficient")
for position, (name, beta, p_value) in enumerate(ranked, start=1):
    say(f"    {position}. {LABELS[name]:<28s} beta = {beta:>8.4f}   p = {p_value:.4f}")

say("\n  Residual diagnostics")
say(f"    Durbin-Watson       {dw_stat:.4f}")
say(f"    Ljung-Box (12)      p = {ljung_p:.4f}")
say(f"    Breusch-Pagan       p = {breusch_p:.4f}")
say(f"    Jarque-Bera         p = {jb_p:.6f}")

say("\n  Reading the output")
say("    A low R squared is normal: monthly growth in a financial aggregate is")
say("    largely noise. What matters is whether the model as a whole is")
say("    significant and which determinants are. This is explanatory modelling")
say("    in Shmueli's (2010) sense - the aim is unbiased, interpretable")
say("    coefficients, not forecast accuracy.")

with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
    handle.write("\n".join(OUTPUT_LINES))

say(f"\n[results written to {RESULTS_FILE}]")
