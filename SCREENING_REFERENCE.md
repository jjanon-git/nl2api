# LSEG Screening & Ranking API Reference for Eval Platform

> **Purpose:** Maps natural language screening/filtering queries to expected API tool calls.
> **Target API:** LSEG Screener (via Refinitiv Data Library) & Datastream Constituent Lists
> **Version:** 1.0.0

---

## 1. Overview: Lookup vs Screening

| Query Type | Example | API Method |
|------------|---------|------------|
| **Lookup** | "What is Apple's revenue?" | `get_data(['AAPL.O'], ['TR.Revenue'])` |
| **Screening** | "Top 5 companies by revenue" | `SCREEN()` + `TOP()` or client-side sort |

---

## 2. SCREEN Expression Syntax

### 2.1 Basic Structure

```
SCREEN(Universe, Filter1, Filter2, ..., Options)
```

### 2.2 Universe Definitions

| Universe | Expression | Description |
|----------|------------|-------------|
| All public equities | `U(IN(Equity(active,public,primary)))` | Active, public, primary listings |
| Include inactive | `U(IN(Equity(active or inactive,public,primary)))` | Include delisted |
| Private companies | `U(IN(Private(OrgType(COM,UNK,MKP))))` | Private companies |
| Dual listings | `U(IN(Equity(active,public,countryprimaryquote)))` | Include dual-listed |

### 2.3 Filter Operators

| Operator | Syntax | Example |
|----------|--------|---------|
| Greater than | `>=`, `>` | `TR.CompanyMarketCap>=5000` |
| Less than | `<=`, `<` | `TR.PE<=20` |
| Equals | `=` | `TR.GICSIndustryCode="501010"` |
| In set | `IN()` | `IN(TR.ExchangeMarketIdCode,"XNYS","XNAS")` |
| Between | `BETWEEN` | `TR.PE BETWEEN 10 AND 20` |

### 2.4 TOP Function (Ranking)

```
TOP(metric, count, nnumber)
```

| Parameter | Description |
|-----------|-------------|
| `metric` | Field to rank by (e.g., `TR.Revenue`) |
| `count` | Number of results (e.g., `10`) |
| `nnumber` | Keyword (always use `nnumber`) |

### 2.5 Options

| Option | Example | Description |
|--------|---------|-------------|
| Currency | `CURN=USD` | Standardize to currency |
| Scale | `Scale=6` | 6 = millions, 9 = billions |

---

## 3. Common Filter Fields

### 3.1 Geographic Filters

| Natural Language | Field | Example Values |
|-----------------|-------|----------------|
| country, market | `TR.ExchangeCountryCode` | `"US"`, `"GB"`, `"JP"` |
| exchange | `TR.ExchangeMarketIdCode` | `"XNYS"`, `"XNAS"`, `"XLON"` |
| region | `TR.RegCountryCode` | `"US"`, `"DE"`, `"CN"` |

### 3.2 Industry/Sector Filters

| Natural Language | Field | Notes |
|-----------------|-------|-------|
| sector | `TR.TRBCEconSectorCode` | TRBC economic sector |
| industry | `TR.TRBCBusinessSectorCode` | TRBC business sector |
| GICS sector | `TR.GICSSectorCode` | GICS classification |
| GICS industry | `TR.GICSIndustryCode` | GICS industry |
| activity | `TR.TRBCActivityCode` | Specific activity |

### 3.3 Size Filters

| Natural Language | Field | Scale |
|-----------------|-------|-------|
| market cap | `TR.CompanyMarketCap` | Default: local currency |
| revenue, sales | `TR.Revenue` | Use `Scale=6` for millions |
| total assets | `TR.TotalAssets` | Use `Scale=6` for millions |
| employees | `TR.Employees` | Count |

### 3.4 Valuation Filters

| Natural Language | Field |
|-----------------|-------|
| PE ratio | `TR.PE` |
| forward PE | `TR.PtoEPSMeanEst` |
| PB ratio | `TR.PriceToBVPerShare` |
| EV/EBITDA | `TR.EVToEBITDA` |
| dividend yield | `TR.DividendYield` |

### 3.5 Performance Filters

| Natural Language | Field |
|-----------------|-------|
| 1-month return | `TR.TotalReturn1Mo` |
| 3-month return | `TR.TotalReturn3Mo` |
| YTD return | `TR.TotalReturnYTD` |
| 1-year return | `TR.TotalReturn1Yr` |
| 52-week high | `TR.Price52WeekHigh` |
| 52-week low | `TR.Price52WeekLow` |

### 3.6 Fundamental Filters

| Natural Language | Field |
|-----------------|-------|
| ROE | `TR.ROE` |
| ROA | `TR.ROA` |
| profit margin | `TR.NetProfitMargin` |
| debt/equity | `TR.DebtToEquity` |
| current ratio | `TR.CurrentRatio` |

---

## 4. Datastream Constituent Lists

### 4.1 Index Constituent Codes

| Index | List Code | Usage |
|-------|-----------|-------|
| S&P 500 | `LS&PCOMP` | `LS&PCOMP\|L` |
| FTSE 100 | `LFTSE100` | `LFTSE100\|L` |
| DAX | `LDAXINDX` | `LDAXINDX\|L` |
| Nikkei 225 | `LSTOKYOSE` | `LSTOKYOSE\|L` |
| NASDAQ 100 | `LNASDAQ` | `LNASDAQ\|L` |
| Euro Stoxx 50 | `LDJES50I` | `LDJES50I\|L` |
| Russell 2000 | `LRUSS2000` | `LRUSS2000\|L` |

### 4.2 Historical Constituents

Format: `L{INDEX}{MMYY}|L`

| Query | Code |
|-------|------|
| S&P 500 Jan 2021 | `LS&PCOMP0121\|L` |
| S&P 500 Dec 2020 | `LS&PCOMP1220\|L` |
| FTSE 100 Jun 2023 | `LFTSE1000623\|L` |

---

## 5. Question-Answer Reference (Test Cases)

### 5.1 Level 1: Simple "Top N" Queries

#### By Market Cap

**Q1:** "What are the 10 largest companies by market cap?"
```python
# Method 1: SCREEN with TOP
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])

# Method 2: Get data and sort client-side
df = get_data([large_universe], ['TR.CommonName', 'TR.CompanyMarketCap'])
df.sort_values('Market Cap', ascending=False).head(10)
```

**Q2:** "Show me the top 5 US companies by market cap"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"US"), TOP(TR.CompanyMarketCap, 5, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])
```

**Q3:** "Get the largest UK companies"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"GB"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])
```

**Q4:** "What are the biggest companies on the NYSE?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeMarketIdCode,"XNYS"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.ExchangeName'])
```

**Q5:** "Show the 10 largest NASDAQ stocks"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeMarketIdCode,"XNAS"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])
```

#### By Revenue

**Q6:** "What are the top 5 companies by revenue?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.Revenue, 5, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.Revenue'])
```

**Q7:** "Show the highest revenue companies in the US"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"US"), TOP(TR.Revenue, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.Revenue'])
```

**Q8:** "Get top 10 companies by sales last quarter"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.RevenueActValue(Period=FQ0), 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.RevenueActValue(Period=FQ0)'])
```

---

### 5.2 Level 2: Sector/Industry Screening

**Q9:** "What are the largest tech companies?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.TRBCEconomicSector'])
```

**Q10:** "Show top 5 banks by market cap"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCBusinessSectorCode,"5510"), TOP(TR.CompanyMarketCap, 5, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])
```

**Q11:** "Get the biggest healthcare companies"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"55"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.TRBCEconomicSector'])
```

**Q12:** "What are the largest oil companies globally?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCBusinessSectorCode,"5010"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.HeadquartersCountry'])
```

**Q13:** "Show top semiconductor companies by revenue"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCActivityCode,"57101010"), TOP(TR.Revenue, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.Revenue'])
```

---

### 5.3 Level 3: Index Constituent Queries

**Q14:** "What companies are in the S&P 500?"
```python
# Datastream method
get_data(tickers='LS&PCOMP|L', fields=['MNEM', 'NAME'], kind=0)

# Or with Refinitiv
get_data(['0#.SPX'], ['TR.CommonName', 'TR.RIC'])
```

**Q15:** "List all FTSE 100 constituents"
```python
get_data(tickers='LFTSE100|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q16:** "Get the DAX 40 companies"
```python
get_data(tickers='LDAXINDX|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q17:** "What stocks are in the Dow Jones?"
```python
get_data(tickers='LDJINDUS|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q18:** "Show NASDAQ 100 constituents"
```python
get_data(tickers='LNASDAQ|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q19:** "What were the S&P 500 constituents in January 2020?"
```python
get_data(tickers='LS&PCOMP0120|L', fields=['MNEM', 'NAME'], kind=0)
```

---

### 5.4 Level 4: Multi-Criteria Filtering

**Q20:** "Find US tech companies with market cap over $10 billion"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"US"), IN(TR.TRBCEconSectorCode,"57"), TR.CompanyMarketCap(Scale=9)>=10, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.TRBCIndustry'])
```

**Q21:** "Show companies with PE ratio under 15 and dividend yield over 3%"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.PE<=15, TR.DividendYield>=3, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.PE', 'TR.DividendYield'])
```

**Q22:** "Find large cap US stocks with ROE above 20%"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"US"), TR.CompanyMarketCap(Scale=9)>=10, TR.ROE>=20, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.ROE'])
```

**Q23:** "Get NYSE stocks with market cap between $1B and $10B"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeMarketIdCode,"XNYS"), TR.CompanyMarketCap(Scale=9)>=1, TR.CompanyMarketCap(Scale=9)<=10, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])
```

**Q24:** "Find tech stocks trading below 52-week high"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), TR.PriceClose<TR.Price52WeekHigh*0.9, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.PriceClose', 'TR.Price52WeekHigh'])
```

---

### 5.5 Level 5: Performance-Based Screening

**Q25:** "What stocks have gained the most this year?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.TotalReturnYTD, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.TotalReturnYTD'])
```

**Q26:** "Show the worst performing stocks in the last month"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"US"), CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.TotalReturn1Mo'])
df.sort_values('TR.TotalReturn1Mo', ascending=True).head(10)
```

**Q27:** "Find US stocks up more than 50% this year"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.ExchangeCountryCode,"US"), TR.TotalReturnYTD>=50, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.TotalReturnYTD'])
```

**Q28:** "Get stocks at 52-week highs"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.PriceClose>=TR.Price52WeekHigh*0.98, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.PriceClose', 'TR.Price52WeekHigh'])
```

**Q29:** "Show stocks that have dropped more than 20% in the last 3 months"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.TotalReturn3Mo<=-20, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.TotalReturn3Mo'])
```

---

### 5.6 Level 6: Fundamental Screening

**Q30:** "Find undervalued stocks: PE under 10 with positive earnings"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.PE>0, TR.PE<=10, TR.NetIncome>0, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.PE', 'TR.NetIncome'])
```

**Q31:** "Show high dividend yield stocks (over 5%)"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.DividendYield>=5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.DividendYield', 'TR.CompanyMarketCap'])
```

**Q32:** "Find profitable companies with low debt"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.NetProfitMargin>=10, TR.DebtToEquity<=0.5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NetProfitMargin', 'TR.DebtToEquity'])
```

**Q33:** "Get stocks with highest ROE"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.ROE, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.ROE'])
```

**Q34:** "Find growth stocks: revenue growth over 20%"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.RevenueGrowth>=20, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.RevenueGrowth', 'TR.Revenue'])
```

---

### 5.7 Level 7: Index-Relative Screening

**Q35:** "Which S&P 500 stocks have the highest dividend yield?"
```python
# Step 1: Get S&P 500 constituents
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
# Step 2: Get dividend yields and sort
df = get_data(constituents, ['TR.CommonName', 'TR.DividendYield'])
df.sort_values('DividendYield', ascending=False).head(10)
```

**Q36:** "What are the cheapest stocks in the FTSE 100 by PE?"
```python
# Step 1: Get FTSE 100 constituents
constituents = get_data(tickers='LFTSE100|L', fields=['RIC'], kind=0)
# Step 2: Get PE ratios and sort
df = get_data(constituents, ['TR.CommonName', 'TR.PE'])
df[df['PE'] > 0].sort_values('PE', ascending=True).head(10)
```

**Q37:** "Show the top performing stocks in the DAX this year"
```python
constituents = get_data(tickers='LDAXINDX|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.TotalReturnYTD'])
df.sort_values('TotalReturnYTD', ascending=False).head(10)
```

**Q38:** "Find the largest S&P 500 companies by revenue"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.Revenue'])
df.sort_values('Revenue', ascending=False).head(10)
```

**Q39:** "Which Dow Jones stocks have the lowest PE ratio?"
```python
constituents = get_data(tickers='LDJINDUS|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.PE'])
df[df['PE'] > 0].sort_values('PE', ascending=True).head(5)
```

---

### 5.8 Level 8: Sector Comparison Within Index

**Q40:** "What are the largest tech stocks in the S&P 500?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.IndexRIC,".SPX"), IN(TR.TRBCEconSectorCode,"57"), TOP(TR.CompanyMarketCap, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap', 'TR.TRBCIndustry'])
```

**Q41:** "Show the top banks in the S&P 500 by assets"
```python
# Get S&P 500 financials
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.TotalAssets', 'TR.TRBCEconSectorCode'])
banks = df[df['TRBCEconSectorCode'] == '55']  # Financials
banks.sort_values('TotalAssets', ascending=False).head(10)
```

**Q42:** "Find the highest yielding REITs in the market"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCBusinessSectorCode,"6010"), TOP(TR.DividendYield, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.DividendYield'])
```

---

### 5.9 Level 9: Earnings & Estimates Screening

**Q43:** "Find stocks that beat earnings estimates last quarter"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.EPSSurprisePct(Period=FQ0)>0, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSSurprisePct(Period=FQ0)'])
```

**Q44:** "Show stocks with the most analyst upgrades"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.EPSNumUp, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSNumUp', 'TR.EPSNumDown'])
```

**Q45:** "Find stocks where analysts are raising estimates"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.MeanPctChg(Period=FY1,WP=30d)>5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
```

**Q46:** "Get stocks with highest expected earnings growth"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.LTGMean, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.LTGMean', 'TR.EPSMean(Period=FY1)'])
```

**Q47:** "Show stocks with most buy ratings"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.NumBuys, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NumBuys', 'TR.RecMean'])
```

---

### 5.10 Level 10: Complex Multi-Factor Screens

**Q48:** "Find quality stocks: high ROE, low debt, consistent dividends"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.ROE>=15,
    TR.DebtToEquity<=1,
    TR.DividendYield>=1,
    TR.CompanyMarketCap(Scale=9)>=5,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.ROE', 'TR.DebtToEquity', 'TR.DividendYield'])
```

**Q49:** "Screen for GARP stocks: reasonable PE with strong growth"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.PE>=5, TR.PE<=25,
    TR.LTGMean>=10,
    TR.PEGRatio<=2,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.PE', 'TR.LTGMean', 'TR.PEGRatio'])
```

**Q50:** "Find momentum stocks: positive returns and rising estimates"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.TotalReturn3Mo>=10,
    TR.TotalReturn1Mo>=0,
    TR.MeanPctChg(Period=FY1,WP=30d)>0,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.TotalReturn3Mo', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
```

---

### 5.11 Level 11: Historical/Point-in-Time Screening

**Q51:** "What were the top 10 companies by market cap in 2020?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active or inactive,public,primary))), TOP(TR.CompanyMarketCap(SDate=2020-12-31), 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap(SDate=2020-12-31)'])
```

**Q52:** "Show the largest US banks at the end of 2019"
```python
screen_exp = '''SCREEN(U(IN(Equity(active or inactive,public,primary))),
    IN(TR.ExchangeCountryCode,"US"),
    IN(TR.TRBCBusinessSectorCode,"5510"),
    TOP(TR.CompanyMarketCap(SDate=2019-12-31), 10, nnumber),
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap(SDate=2019-12-31)'])
```

**Q53:** "What companies were in the S&P 500 in 2015 but aren't now?"
```python
# Step 1: Get 2015 constituents
spx_2015 = get_data(tickers='LS&PCOMP0115|L', fields=['RIC'], kind=0)
# Step 2: Get current constituents
spx_now = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
# Step 3: Find difference (client-side)
removed = set(spx_2015) - set(spx_now)
```

---

### 5.12 Level 12: Aggregate/Statistical Queries

**Q54:** "What is the average PE ratio of S&P 500 stocks?"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.PE'])
df['PE'].mean()  # Calculate average client-side
```

**Q55:** "What's the median dividend yield of FTSE 100 companies?"
```python
constituents = get_data(tickers='LFTSE100|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.DividendYield'])
df['DividendYield'].median()
```

**Q56:** "How many tech companies have market cap over $100B?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), TR.CompanyMarketCap(Scale=9)>=100, CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.CompanyMarketCap'])
len(df)  # Count client-side
```

**Q57:** "What percentage of S&P 500 stocks pay dividends?"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.DividendYield'])
pct = (df['DividendYield'] > 0).sum() / len(df) * 100
```

---

### 5.13 Level 13: Officer/Director Screening

**Q58:** "Which S&P 500 companies have the highest paid CEOs?"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
df.sort_values('ODOfficerTotalComp', ascending=False).head(10)
```

**Q59:** "Find companies where the CEO has been in role for over 10 years"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.OfficerTitleSince'])
# Filter for tenure > 10 years client-side
```

**Q60:** "Show tech companies with female CEOs"
```python
# Would require additional gender field or name-based inference
# This is an example of a query that may need officer detail data
```

---

## 6. Natural Language Mapping

### 6.1 Ranking Keywords

| Expression | Intent | Implementation |
|------------|--------|----------------|
| "top N", "largest N", "biggest N" | Descending sort, take N | `TOP(..., N, nnumber)` or sort desc + head |
| "smallest N", "lowest N" | Ascending sort, take N | Sort asc + head |
| "highest", "most", "greatest" | Descending sort | Sort desc |
| "lowest", "least", "cheapest" | Ascending sort | Sort asc |
| "rank by", "sort by", "order by" | Sort operation | Sort + direction |

### 6.2 Filter Keywords

| Expression | Intent | Implementation |
|------------|--------|----------------|
| "over", "above", "more than", "greater than" | `>=` or `>` | Comparison filter |
| "under", "below", "less than" | `<=` or `<` | Comparison filter |
| "between X and Y" | Range filter | Two comparisons |
| "at least", "minimum" | `>=` | Lower bound |
| "at most", "maximum" | `<=` | Upper bound |

### 6.3 Universe Keywords

| Expression | Maps To |
|------------|---------|
| "in the S&P 500", "S&P 500 stocks" | `LS&PCOMP\|L` constituents |
| "US stocks", "American companies" | `IN(TR.ExchangeCountryCode,"US")` |
| "tech stocks", "technology sector" | `IN(TR.TRBCEconSectorCode,"57")` |
| "banks", "banking sector" | `IN(TR.TRBCBusinessSectorCode,"5510")` |
| "on NYSE", "NYSE listed" | `IN(TR.ExchangeMarketIdCode,"XNYS")` |

---

## 7. Two-Step Query Pattern

Many screening queries require two steps:

### Step 1: Get Universe (via SCREEN or Constituent List)
### Step 2: Get Data + Sort/Filter Client-Side

```python
# Example: Top 5 S&P 500 stocks by revenue last quarter

# Step 1: Get S&P 500 constituents
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
rics = constituents['RIC'].tolist()

# Step 2: Get revenue data
df = get_data(rics, ['TR.CommonName', 'TR.RevenueActValue(Period=FQ0)'])

# Step 3: Sort and take top 5
top_5 = df.sort_values('RevenueActValue', ascending=False).head(5)
```

---

## 8. Complexity Classification

| Level | Criteria | Examples |
|-------|----------|----------|
| 1 | Simple top N by single metric | Top 10 by market cap (Q1-Q8) |
| 2 | Top N within sector/industry | Largest tech companies (Q9-Q13) |
| 3 | Index constituent lists | S&P 500 companies (Q14-Q19) |
| 4 | Multi-criteria filters | PE < 15 AND yield > 3% (Q20-Q24) |
| 5 | Performance-based screens | YTD gainers/losers (Q25-Q29) |
| 6 | Fundamental screens | Value/quality filters (Q30-Q34) |
| 7 | Index-relative ranking | Highest yield in S&P 500 (Q35-Q39) |
| 8 | Sector within index | Top tech in S&P 500 (Q40-Q42) |
| 9 | Estimates-based screens | Earnings beats (Q43-Q47) |
| 10 | Multi-factor screens | Quality + value + momentum (Q48-Q50) |
| 11 | Historical screens | Top 10 in 2020 (Q51-Q53) |
| 12 | Aggregate/statistics | Average PE of index (Q54-Q57) |
| 13 | Officer/director screens | Highest paid CEOs (Q58-Q60) |

---

## 9. Error Handling

| Scenario | Expected Handling |
|----------|-------------------|
| Universe too large | Return warning, suggest narrower criteria |
| No matches | Return empty result with note |
| Invalid field for ranking | Return error with valid alternatives |
| Historical date unavailable | Return closest available date |

---

## Sources

- [LSEG Screener Article](https://developers.lseg.com/en/article-catalog/article/find-your-right-companies-with-screener)
- [LSEG Developer Community - Screener](https://community.developers.refinitiv.com/questions/43537/screener-sort-and-top-results.html)
- [Datastream Constituent Lists](https://developers.lseg.com/content/dam/devportal/api-families/eikon/datastream-web-service/documentation/manuals-and-guides/datastream-refinitiv-dsws-python_0.pdf)
- [LSEG-API-Samples GitHub](https://github.com/LSEG-API-Samples/Article.DataLibrary.Python.Screener)
