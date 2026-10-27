# Macroeconomic Determinants of UK Credit Card Borrowing (2000–2026)

MSc Financial Technology and Business Analytics dissertation · University of Chester · Jude Adjei

**Question:** to what extent do macroeconomic factors explain monthly variation in the growth of UK credit card borrowing?

## Highlights
- 315 monthly observations (March 2000 – May 2026) from the Bank of England and ONS.
- OLS regression with Newey–West (HAC, 12 lags) standard errors, crisis indicators for the financial crisis, COVID-19 and the cost-of-living period.
- **Labour market conditions were the strongest determinant** (claimant count, standardised β = −0.240, p < 0.0001); **CPI inflation ranked second**; the **card interest rate ranked last** in both specifications.
- Relationships stayed stable through all three crises (slope interaction test p = 0.286, CUSUM p = 0.367).

## What the script does
`analysis.py` rebuilds every variable from the raw published series and runs each test reported in Chapter 4, in 16 stages:

| Stage | Step |
|---|---|
| 1–5 | Load, validate (missing values, duplicates), build log-difference growth, first differences, lags and crisis indicators |
| 6–8 | Descriptive statistics, Augmented Dickey–Fuller tests, correlation and VIF |
| 9–10 | Main regression with Newey–West errors; ranking by standardised coefficients |
| 11 | Residual diagnostics: Durbin–Watson, Ljung–Box, Breusch–Pagan, Jarque–Bera |
| 12 | Distributed-lag model (0–3 months) with joint tests |
| 13 | Structural stability: crisis interactions, Chow tests, CUSUM |
| 14 | Residual seasonality with month indicators |
| 15 | Robustness: extreme-month marker, claimant count in place of LFS unemployment |
| 16 | Summary written to `analysis_results.txt` |

## Run it
```bash
pip install -r requirements.txt
python analysis.py
```
Place `regression_ready_data.csv` next to the script (see `data/README.md`).

## Tools
Python · pandas · NumPy · statsmodels
