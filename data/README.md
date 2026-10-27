# Data sources

All series are published monthly and free to use under the Open Government Licence.

| Series | Code | Publisher | Download |
|---|---|---|---|
| Credit card balances outstanding (£m, SA) | LPMVZRJ | Bank of England | [CSV](https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2000&Dateto=01/Aug/2026&SeriesCodes=LPMVZRJ&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N) |
| Representative credit card interest rate (%) | IUMCCTL | Bank of England | [CSV](https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2000&Dateto=01/Aug/2026&SeriesCodes=IUMCCTL&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N) |
| Bank Rate, monthly average (%) | IUMABEDR | Bank of England | [CSV](https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jan/2000&Dateto=01/Aug/2026&SeriesCodes=IUMABEDR&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N) |
| CPI annual inflation rate (%) | D7G7 | ONS (MM23) | [CSV](https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7g7/mm23) |
| Unemployment rate, 16+ (%, SA) | MGSX | ONS (LMS) | [CSV](https://www.ons.gov.uk/generator?format=csv&uri=/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms) |
| Claimant count (000s, SA) | BCJD | ONS (UNEM) | [CSV](https://www.ons.gov.uk/generator?format=csv&uri=/employmentandlabourmarket/peoplenotinwork/outofworkbenefits/timeseries/bcjd/unem) |

Merge the series on calendar month into `regression_ready_data.csv`. Row 1 holds these column names (quote the ones containing commas); row 2 holds the series codes and is skipped on load:

- `Month`
- `Credit card balances outstanding`
- `Representative card interest rate`
- `Bank Rate, monthly average`
- `CPI annual inflation rate`
- `Unemployment rate, aged 16+`
- `Claimant count`

Dates in `Month` use M/D/YYYY. ONS files contain annual, quarterly and monthly blocks; keep only the monthly rows.
