# LSEG I/B/E/S Estimates API Reference for Eval Platform

> **Purpose:** Maps natural language queries to expected Estimates API tool calls.
> **Target API:** LSEG I/B/E/S Estimates (via Refinitiv Data Library & Datastream)
> **Version:** 1.0.0

---

## 1. Database Overview

### 1.1 Coverage

| Metric | Value |
|--------|-------|
| Active Companies | 23,000+ |
| Total Historical Companies | 60,000+ |
| Countries | 90+ |
| Contributing Broker Firms | 950+ |
| Individual Analysts | 19,000+ |
| US History | Since 1976 |
| International History | Since 1987 |
| Financial Measures | 360+ |

### 1.2 Data Categories

- **Consensus Estimates** - Aggregated analyst forecasts
- **Detail Estimates** - Individual analyst forecasts
- **Actuals** - Reported values
- **Guidance** - Company-provided forecasts
- **Recommendations** - Buy/Hold/Sell ratings
- **SmartEstimates** - Weighted by analyst accuracy

### 1.3 Key Metrics Covered

| Category | Metrics |
|----------|---------|
| Earnings | EPS, Net Income, EBIT, EBITDA |
| Revenue | Total Revenue, Segment Revenue |
| Cash Flow | Operating CF, Free Cash Flow, CFPS |
| Dividends | DPS, Dividend Yield |
| Valuation | PE Ratio, EV/EBITDA |
| Growth | EPS Growth, Revenue Growth |
| Margins | Gross Margin, Operating Margin |

---

## 2. TR Field Code Reference - Estimates

### 2.1 Consensus Mean Estimates

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| EPS estimate, earnings estimate | `TR.EPSMean` | Mean EPS estimate |
| revenue estimate, sales estimate | `TR.RevenueMean` | Mean revenue estimate |
| EBITDA estimate | `TR.EBITDAMean` | Mean EBITDA estimate |
| EBIT estimate | `TR.EBITMean` | Mean EBIT estimate |
| net income estimate | `TR.NetProfitMean` | Mean net income estimate |
| pretax profit estimate | `TR.PreTaxProfitMean` | Mean pretax income |
| gross income estimate | `TR.GrossIncomeMean` | Mean gross profit |
| operating income estimate | `TR.OperatingIncomeMean` | Mean operating income |
| free cash flow estimate | `TR.FCFMean` | Mean FCF estimate |
| cash flow per share estimate | `TR.CFPSMean` | Mean CFPS estimate |
| DPS estimate, dividend estimate | `TR.DPSMean` | Mean DPS estimate |
| book value estimate | `TR.BVPSMean` | Mean BVPS estimate |
| ROE estimate | `TR.ROEMean` | Mean ROE estimate |
| ROA estimate | `TR.ROAMean` | Mean ROA estimate |
| total assets estimate | `TR.TotalAssetsMean` | Mean total assets |
| enterprise value estimate | `TR.EVMean` | Mean EV estimate |
| net debt estimate | `TR.NetDebtMean` | Mean net debt |

### 2.2 Estimate Counts & Statistics

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| number of EPS estimates | `TR.EPSNumIncEstimates` | Count of EPS estimates |
| number of revenue estimates | `TR.RevenueNumIncEstimates` | Count of revenue estimates |
| number of analysts | `TR.NumOfEst` | Total analyst count |
| EPS high estimate | `TR.EPSHigh` | Highest EPS estimate |
| EPS low estimate | `TR.EPSLow` | Lowest EPS estimate |
| EPS standard deviation | `TR.EPSStdDev` | Estimate dispersion |
| revenue high | `TR.RevenueHigh` | Highest revenue estimate |
| revenue low | `TR.RevenueLow` | Lowest revenue estimate |

### 2.3 Estimate Changes & Revisions

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| EPS revision, estimate change | `TR.EPSMeanChgPct` | % change in mean estimate |
| revenue revision | `TR.RevenueMeanChgPct` | % change in revenue est |
| estimate trend | `TR.MeanPctChg(Period=FY1,WP=60d)` | 60-day % change |
| upgrades | `TR.EPSNumUp` | Number of upward revisions |
| downgrades | `TR.EPSNumDown` | Number of downward revisions |
| revision ratio | Calculated | Up / (Up + Down) |

### 2.4 Actual Reported Values

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| actual EPS, reported EPS | `TR.EPSActValue` | Reported EPS |
| actual revenue, reported revenue | `TR.RevenueActValue` | Reported revenue |
| actual EBITDA | `TR.EBITDAActValue` | Reported EBITDA |
| actual net income | `TR.NetIncomeActValue` | Reported net income |
| actual DPS | `TR.DPSActValue` | Reported dividends |
| actual free cash flow | `TR.FCFActValue` | Reported FCF |

### 2.5 Earnings Surprise

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| earnings surprise, EPS surprise | `TR.EPSSurprise` | Actual - Estimate |
| surprise percentage | `TR.EPSSurprisePct` | % surprise |
| beat/miss | Calculated | Positive = beat |
| revenue surprise | `TR.RevenueSurprise` | Revenue actual vs est |

### 2.6 Forward Valuations

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| forward PE, FY1 PE | `TR.PtoEPSMeanEst(Period=FY1)` | Forward P/E ratio |
| forward EV/EBITDA | `TR.EVToEBITDAMean` | Forward EV/EBITDA |
| PEG ratio | `TR.PEGRatio` | P/E to growth |

### 2.7 Analyst Recommendations

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| analyst rating, recommendation | `TR.RecMean` | Mean recommendation (1-5) |
| buy rating | `TR.NumBuys` | Number of buy ratings |
| hold rating | `TR.NumHolds` | Number of hold ratings |
| sell rating | `TR.NumSells` | Number of sell ratings |
| price target | `TR.PriceTargetMean` | Mean price target |
| price target high | `TR.PriceTargetHigh` | Highest target |
| price target low | `TR.PriceTargetLow` | Lowest target |
| target upside | Calculated | (Target - Price) / Price |

### 2.8 Long-Term Growth

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| long-term growth, LTG | `TR.LTGMean` | Long-term growth estimate |
| 5-year growth | `TR.LTGMean` | Typically 5-year CAGR |
| earnings growth rate | `TR.EPSGrowthRate` | Expected EPS growth |

---

## 3. Period Parameters

### 3.1 Fiscal Year Periods

| Parameter | Description | Example |
|-----------|-------------|---------|
| `FY0` | Current fiscal year (most recent actual) | 2024 (if latest annual) |
| `FY1` | Next fiscal year estimate | 2025 forecast |
| `FY2` | Two years ahead estimate | 2026 forecast |
| `FY3` | Three years ahead estimate | 2027 forecast |
| `FY-1` | Prior fiscal year actual | 2023 actual |
| `FY-2` | Two years ago actual | 2022 actual |

### 3.2 Fiscal Quarter Periods

| Parameter | Description | Example |
|-----------|-------------|---------|
| `FQ0` | Current quarter (most recent actual) | Q4 2024 |
| `FQ1` | Next quarter estimate | Q1 2025 |
| `FQ2` | Two quarters ahead | Q2 2025 |
| `FQ-1` | Prior quarter actual | Q3 2024 |
| `FQ-3` | Same quarter last year | Q4 2023 |

### 3.3 Special Periods

| Parameter | Description |
|-----------|-------------|
| `LTM` | Last twelve months (trailing) |
| `NTM` | Next twelve months (forward) |
| `12M` | Twelve month forward |
| `18M` | Eighteen month forward |

### 3.4 Usage Examples

```python
# Current year estimate
'TR.EPSMean(Period=FY1)'

# Next quarter estimate
'TR.RevenueMean(Period=FQ1)'

# Year-over-year comparison
'TR.EPSActValue(Period=FQ-3)'  # Same quarter last year

# Historical estimates with dates
'TR.EPSMean(Period=FY1,SDate=2020-01-01,EDate=2024-12-31,Frq=FQ)'
```

---

## 4. Datastream IBES Field Codes

### 4.1 Consensus Estimates (Datastream)

| Metric | FY1 Code | FY2 Code | Description |
|--------|----------|----------|-------------|
| EPS Mean | `EPSF1MN` | `EPSF2MN` | Mean EPS estimate |
| EPS Median | `EPSF1MD` | `EPSF2MD` | Median EPS estimate |
| Revenue Mean | `REVF1MN` | `REVF2MN` | Mean revenue estimate |
| DPS Mean | `DPSF1MN` | `DPSF2MN` | Mean DPS estimate |
| CFPS Mean | `CPSF1MN` | `CPSF2MN` | Mean CFPS estimate |
| BPS Mean | `BPSF1MN` | `BPSF2MN` | Mean book value/share |

### 4.2 Actual Values (Datastream)

| Metric | Code Pattern | Example |
|--------|--------------|---------|
| EPS Actual | `A##EPS` | `A24EPS` (2024 actual) |
| Revenue Actual | `A##REV` | `A24REV` |
| DPS Actual | `A##DPS` | `A24DPS` |

### 4.3 Global Aggregate Codes

| Metric | Code | Description |
|--------|------|-------------|
| PE Ratio FY0 | `AF0PE` | Current year PE |
| PE Ratio FY1 | `AF1PE` | Forward PE |
| Earnings Growth FY1 | `AF1GRO` | Expected growth |
| Dividend Yield | `ADVYLD` | Forward dividend yield |

---

## 5. Question-Answer Reference (Test Cases)

### 5.1 Level 1: Basic EPS Estimates

#### Current Year Estimates

**Q1:** "What is the EPS estimate for Apple?"
```python
get_data(['AAPL.O'], ['TR.EPSMean(Period=FY1)'])
```

**Q2:** "Get Microsoft's earnings estimate"
```python
get_data(['MSFT.O'], ['TR.EPSMean(Period=FY1)'])
```

**Q3:** "What are analysts expecting for Amazon's EPS?"
```python
get_data(['AMZN.O'], ['TR.EPSMean(Period=FY1)'])
```

**Q4:** "Get Google's consensus EPS forecast"
```python
get_data(['GOOGL.O'], ['TR.EPSMean(Period=FY1)'])
```

**Q5:** "What's Tesla's expected earnings per share?"
```python
get_data(['TSLA.O'], ['TR.EPSMean(Period=FY1)'])
```

#### Next Quarter Estimates

**Q6:** "What is Apple's next quarter EPS estimate?"
```python
get_data(['AAPL.O'], ['TR.EPSMean(Period=FQ1)'])
```

**Q7:** "Get Microsoft's Q1 earnings estimate"
```python
get_data(['MSFT.O'], ['TR.EPSMean(Period=FQ1)'])
```

**Q8:** "What do analysts expect for Amazon's quarterly EPS?"
```python
get_data(['AMZN.O'], ['TR.EPSMean(Period=FQ1)'])
```

---

### 5.2 Level 2: Revenue Estimates

**Q9:** "What is Apple's revenue estimate?"
```python
get_data(['AAPL.O'], ['TR.RevenueMean(Period=FY1)'])
```

**Q10:** "Get Microsoft's expected sales for next year"
```python
get_data(['MSFT.O'], ['TR.RevenueMean(Period=FY1)'])
```

**Q11:** "What's the revenue forecast for Amazon?"
```python
get_data(['AMZN.O'], ['TR.RevenueMean(Period=FY1)'])
```

**Q12:** "Show Google's quarterly revenue estimate"
```python
get_data(['GOOGL.O'], ['TR.RevenueMean(Period=FQ1)'])
```

**Q13:** "Get Tesla's sales estimate for next quarter"
```python
get_data(['TSLA.O'], ['TR.RevenueMean(Period=FQ1)'])
```

---

### 5.3 Level 3: Multiple Estimates

**Q14:** "Get Apple's EPS and revenue estimates"
```python
get_data(['AAPL.O'], ['TR.EPSMean(Period=FY1)', 'TR.RevenueMean(Period=FY1)'])
```

**Q15:** "Show Microsoft's earnings, revenue, and EBITDA forecasts"
```python
get_data(['MSFT.O'], ['TR.EPSMean(Period=FY1)', 'TR.RevenueMean(Period=FY1)', 'TR.EBITDAMean(Period=FY1)'])
```

**Q16:** "Get full estimate summary for Amazon"
```python
get_data(['AMZN.O'], [
    'TR.EPSMean(Period=FY1)',
    'TR.RevenueMean(Period=FY1)',
    'TR.EBITDAMean(Period=FY1)',
    'TR.NetProfitMean(Period=FY1)',
    'TR.FCFMean(Period=FY1)'
])
```

**Q17:** "What are analysts projecting for Google's key metrics?"
```python
get_data(['GOOGL.O'], [
    'TR.EPSMean(Period=FY1)',
    'TR.RevenueMean(Period=FY1)',
    'TR.EBITDAMean(Period=FY1)',
    'TR.DPSMean(Period=FY1)'
])
```

---

### 5.4 Level 4: Estimate Details & Dispersion

**Q18:** "How many analysts cover Apple?"
```python
get_data(['AAPL.O'], ['TR.EPSNumIncEstimates(Period=FY1)'])
```

**Q19:** "Get the range of EPS estimates for Microsoft"
```python
get_data(['MSFT.O'], ['TR.EPSMean(Period=FY1)', 'TR.EPSHigh(Period=FY1)', 'TR.EPSLow(Period=FY1)'])
```

**Q20:** "Show analyst estimate dispersion for Amazon"
```python
get_data(['AMZN.O'], [
    'TR.EPSMean(Period=FY1)',
    'TR.EPSHigh(Period=FY1)',
    'TR.EPSLow(Period=FY1)',
    'TR.EPSStdDev(Period=FY1)',
    'TR.EPSNumIncEstimates(Period=FY1)'
])
```

**Q21:** "Get revenue estimate range for Tesla"
```python
get_data(['TSLA.O'], [
    'TR.RevenueMean(Period=FY1)',
    'TR.RevenueHigh(Period=FY1)',
    'TR.RevenueLow(Period=FY1)',
    'TR.RevenueNumIncEstimates(Period=FY1)'
])
```

---

### 5.5 Level 5: Actual vs Estimates (Earnings Surprise)

**Q22:** "What was Apple's earnings surprise last quarter?"
```python
get_data(['AAPL.O'], [
    'TR.EPSActValue(Period=FQ0)',
    'TR.EPSMean(Period=FQ0)',
    'TR.EPSSurprise(Period=FQ0)',
    'TR.EPSSurprisePct(Period=FQ0)'
])
```

**Q23:** "Did Microsoft beat or miss estimates?"
```python
get_data(['MSFT.O'], ['TR.EPSSurprisePct(Period=FQ0)'])
```

**Q24:** "Show Amazon's recent earnings surprises"
```python
get_data(['AMZN.O'], [
    'TR.EPSActValue(Period=FQ0)',
    'TR.EPSSurprisePct(Period=FQ0)'
], {'SDate': '-4', 'EDate': '0', 'Period': 'FQ0', 'Frq': 'FQ'})
```

**Q25:** "Get revenue beat/miss for Tesla"
```python
get_data(['TSLA.O'], [
    'TR.RevenueActValue(Period=FQ0)',
    'TR.RevenueMean(Period=FQ0)',
    'TR.RevenueSurprise(Period=FQ0)'
])
```

**Q26:** "Show Google's earnings surprise history"
```python
get_data(['GOOGL.O'], ['TR.EPSSurprisePct.date', 'TR.EPSSurprisePct'],
         {'SDate': '-2Y', 'EDate': '0D', 'Frq': 'FQ'})
```

---

### 5.6 Level 6: Estimate Revisions

**Q27:** "Have analysts raised or lowered Apple estimates recently?"
```python
get_data(['AAPL.O'], [
    'TR.EPSMeanChgPct(Period=FY1)',
    'TR.EPSNumUp(Period=FY1)',
    'TR.EPSNumDown(Period=FY1)'
])
```

**Q28:** "Get estimate revision trend for Microsoft"
```python
get_data(['MSFT.O'], ['TR.MeanPctChg(Period=FY1,WP=30d)', 'TR.MeanPctChg(Period=FY1,WP=60d)', 'TR.MeanPctChg(Period=FY1,WP=90d)'])
```

**Q29:** "Show Amazon's estimate revision history"
```python
get_data(['AMZN.O'], ['TR.EPSMean.calcdate', 'TR.EPSMean(Period=FY1)'],
         {'SDate': '-6M', 'EDate': '0D', 'Frq': 'W'})
```

**Q30:** "Are Tesla estimates trending up or down?"
```python
get_data(['TSLA.O'], [
    'TR.EPSNumUp(Period=FY1)',
    'TR.EPSNumDown(Period=FY1)',
    'TR.RevenueMeanChgPct(Period=FY1)'
])
```

---

### 5.7 Level 7: Analyst Recommendations

**Q31:** "What is the analyst rating for Apple?"
```python
get_data(['AAPL.O'], ['TR.RecMean'])
```

**Q32:** "How many buy/hold/sell ratings does Microsoft have?"
```python
get_data(['MSFT.O'], ['TR.NumBuys', 'TR.NumHolds', 'TR.NumSells'])
```

**Q33:** "Get analyst recommendation breakdown for Amazon"
```python
get_data(['AMZN.O'], [
    'TR.RecMean',
    'TR.NumBuys',
    'TR.NumHolds',
    'TR.NumSells',
    'TR.NumOfEst'
])
```

**Q34:** "What's the consensus rating for Tesla?"
```python
get_data(['TSLA.O'], ['TR.RecMean', 'TR.NumBuys', 'TR.NumHolds', 'TR.NumSells'])
```

**Q35:** "Show Google's analyst recommendation distribution"
```python
get_data(['GOOGL.O'], ['TR.RecMean', 'TR.NumBuys', 'TR.NumHolds', 'TR.NumSells'])
```

---

### 5.8 Level 8: Price Targets

**Q36:** "What is the price target for Apple?"
```python
get_data(['AAPL.O'], ['TR.PriceTargetMean'])
```

**Q37:** "Get Microsoft's price target range"
```python
get_data(['MSFT.O'], ['TR.PriceTargetMean', 'TR.PriceTargetHigh', 'TR.PriceTargetLow'])
```

**Q38:** "What upside do analysts see for Amazon?"
```python
get_data(['AMZN.O'], ['TR.PriceClose', 'TR.PriceTargetMean'])
# Calculate upside: (Target - Price) / Price
```

**Q39:** "Show Tesla's price target summary"
```python
get_data(['TSLA.O'], [
    'TR.PriceClose',
    'TR.PriceTargetMean',
    'TR.PriceTargetHigh',
    'TR.PriceTargetLow',
    'TR.NumOfEst'
])
```

**Q40:** "Get complete analyst view for Google: rating and target"
```python
get_data(['GOOGL.O'], [
    'TR.RecMean',
    'TR.NumBuys',
    'TR.NumHolds',
    'TR.NumSells',
    'TR.PriceTargetMean',
    'TR.PriceTargetHigh',
    'TR.PriceTargetLow'
])
```

---

### 5.9 Level 9: Forward Valuations

**Q41:** "What is Apple's forward PE ratio?"
```python
get_data(['AAPL.O'], ['TR.PtoEPSMeanEst(Period=FY1)'])
```

**Q42:** "Get Microsoft's forward valuation metrics"
```python
get_data(['MSFT.O'], [
    'TR.PtoEPSMeanEst(Period=FY1)',
    'TR.EVToEBITDAMean(Period=FY1)',
    'TR.PEGRatio'
])
```

**Q43:** "Compare current vs forward PE for Amazon"
```python
get_data(['AMZN.O'], ['TR.PE', 'TR.PtoEPSMeanEst(Period=FY1)', 'TR.PtoEPSMeanEst(Period=FY2)'])
```

**Q44:** "Get Tesla's PEG ratio"
```python
get_data(['TSLA.O'], ['TR.PEGRatio', 'TR.PtoEPSMeanEst(Period=FY1)', 'TR.LTGMean'])
```

**Q45:** "Show forward valuation summary for Google"
```python
get_data(['GOOGL.O'], [
    'TR.PtoEPSMeanEst(Period=FY1)',
    'TR.PtoEPSMeanEst(Period=FY2)',
    'TR.EVToEBITDAMean(Period=FY1)',
    'TR.PEGRatio'
])
```

---

### 5.10 Level 10: Multi-Year Forecasts

**Q46:** "Get Apple's EPS estimates for the next 3 years"
```python
get_data(['AAPL.O'], [
    'TR.EPSMean(Period=FY1)',
    'TR.EPSMean(Period=FY2)',
    'TR.EPSMean(Period=FY3)'
])
```

**Q47:** "Show Microsoft's multi-year revenue forecast"
```python
get_data(['MSFT.O'], [
    'TR.RevenueMean(Period=FY1)',
    'TR.RevenueMean(Period=FY2)',
    'TR.RevenueMean(Period=FY3)'
])
```

**Q48:** "Get Amazon's long-term earnings outlook"
```python
get_data(['AMZN.O'], [
    'TR.EPSMean(Period=FY1)',
    'TR.EPSMean(Period=FY2)',
    'TR.EPSMean(Period=FY3)',
    'TR.LTGMean'
])
```

**Q49:** "What's Tesla's expected earnings trajectory?"
```python
get_data(['TSLA.O'], [
    'TR.EPSActValue(Period=FY0)',
    'TR.EPSMean(Period=FY1)',
    'TR.EPSMean(Period=FY2)',
    'TR.EPSMean(Period=FY3)',
    'TR.LTGMean'
])
```

**Q50:** "Show Google's multi-year financial forecast"
```python
get_data(['GOOGL.O'], [
    'TR.EPSMean(Period=FY1)', 'TR.EPSMean(Period=FY2)',
    'TR.RevenueMean(Period=FY1)', 'TR.RevenueMean(Period=FY2)',
    'TR.EBITDAMean(Period=FY1)', 'TR.EBITDAMean(Period=FY2)'
])
```

---

### 5.11 Level 11: Multi-Company Comparisons

**Q51:** "Compare EPS estimates for big tech"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O'],
         ['TR.CommonName', 'TR.EPSMean(Period=FY1)'])
```

**Q52:** "Get revenue forecasts for FAANG stocks"
```python
get_data(['META.O', 'AAPL.O', 'AMZN.O', 'NFLX.O', 'GOOGL.O'],
         ['TR.CommonName', 'TR.RevenueMean(Period=FY1)'])
```

**Q53:** "Compare analyst ratings across tech sector"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O', 'NVDA.O', 'TSLA.O'],
         ['TR.CommonName', 'TR.RecMean', 'TR.PriceTargetMean'])
```

**Q54:** "Which tech stock has the highest earnings growth estimate?"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O', 'NVDA.O'],
         ['TR.CommonName', 'TR.LTGMean', 'TR.EPSMean(Period=FY1)', 'TR.EPSMean(Period=FY2)'])
```

**Q55:** "Compare forward PE ratios for magnificent 7"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O', 'NVDA.O', 'TSLA.O'],
         ['TR.CommonName', 'TR.PtoEPSMeanEst(Period=FY1)', 'TR.PEGRatio'])
```

---

### 5.12 Level 12: Historical Estimates

**Q56:** "Show how Apple EPS estimates have evolved over the year"
```python
get_data(['AAPL.O'], ['TR.EPSMean.calcdate', 'TR.EPSMean(Period=FY1)'],
         {'SDate': '-1Y', 'EDate': '0D', 'Frq': 'W'})
```

**Q57:** "Get Microsoft's historical estimate revisions"
```python
get_data(['MSFT.O'], ['TR.EPSMean.calcdate', 'TR.EPSMean(Period=FY1)', 'TR.RevenueMean(Period=FY1)'],
         {'SDate': '-2Y', 'EDate': '0D', 'Frq': 'M'})
```

**Q58:** "Track Amazon's estimate changes quarterly"
```python
get_data(['AMZN.O'], [
    'TR.EPSMean.calcdate',
    'TR.EPSMean(Period=FY1)',
    'TR.EPSNumIncEstimates(Period=FY1)'
], {'SDate': '-3Y', 'EDate': '0D', 'Frq': 'Q'})
```

---

### 5.13 Level 13: YoY Growth Analysis

**Q59:** "Calculate expected EPS growth for Apple"
```python
get_data(['AAPL.O'], [
    'TR.EPSActValue(Period=FY0)',
    'TR.EPSMean(Period=FY1)',
    'TR.EPSMean(Period=FY2)'
])
# Calculate: (FY1 - FY0) / FY0 * 100
```

**Q60:** "Get year-over-year revenue growth estimate for Microsoft"
```python
get_data(['MSFT.O'], [
    'TR.RevenueActValue(Period=FY0)',
    'TR.RevenueMean(Period=FY1)'
])
# Calculate: (FY1 - FY0) / FY0 * 100
```

**Q61:** "Show quarterly YoY comparisons for Amazon"
```python
get_data(['AMZN.O'], [
    'TR.EPSActValue(Period=FQ-3)',  # Same quarter last year
    'TR.EPSMean(Period=FQ1)',       # Next quarter estimate
    'TR.RevenueActValue(Period=FQ-3)',
    'TR.RevenueMean(Period=FQ1)'
])
```

---

### 5.14 Level 14: Complete Analyst Package

**Q62:** "Get complete analyst summary for Apple"
```python
get_data(['AAPL.O'], [
    # Estimates
    'TR.EPSMean(Period=FY1)',
    'TR.EPSMean(Period=FY2)',
    'TR.RevenueMean(Period=FY1)',
    'TR.EBITDAMean(Period=FY1)',
    # Dispersion
    'TR.EPSHigh(Period=FY1)',
    'TR.EPSLow(Period=FY1)',
    'TR.EPSNumIncEstimates(Period=FY1)',
    # Revisions
    'TR.MeanPctChg(Period=FY1,WP=30d)',
    # Recommendations
    'TR.RecMean',
    'TR.NumBuys',
    'TR.NumHolds',
    'TR.NumSells',
    # Targets
    'TR.PriceTargetMean',
    'TR.PriceTargetHigh',
    'TR.PriceTargetLow',
    # Valuation
    'TR.PtoEPSMeanEst(Period=FY1)',
    'TR.PEGRatio',
    # Growth
    'TR.LTGMean'
])
```

**Q63:** "Full estimate package for earnings analysis"
```python
get_data(['MSFT.O', 'GOOGL.O', 'AMZN.O'], [
    'TR.CommonName',
    # Current estimates
    'TR.EPSMean(Period=FQ1)',
    'TR.EPSMean(Period=FY1)',
    # Actuals for comparison
    'TR.EPSActValue(Period=FQ0)',
    'TR.EPSActValue(Period=FY0)',
    # Surprise history
    'TR.EPSSurprisePct(Period=FQ0)',
    # Analyst sentiment
    'TR.RecMean',
    'TR.EPSNumUp(Period=FY1)',
    'TR.EPSNumDown(Period=FY1)'
])
```

---

### 5.15 Level 15: Sector & Index Analysis

**Q64:** "Compare estimates for semiconductor sector"
```python
get_data(['NVDA.O', 'AMD.O', 'INTC.O', 'QCOM.O', 'AVGO.O', 'TXN.O'], [
    'TR.CommonName',
    'TR.EPSMean(Period=FY1)',
    'TR.RevenueMean(Period=FY1)',
    'TR.LTGMean',
    'TR.RecMean'
])
```

**Q65:** "Get bank sector earnings estimates"
```python
get_data(['JPM.N', 'BAC.N', 'WFC.N', 'C.N', 'GS.N', 'MS.N'], [
    'TR.CommonName',
    'TR.EPSMean(Period=FY1)',
    'TR.RevenueMean(Period=FY1)',
    'TR.PtoEPSMeanEst(Period=FY1)'
])
```

---

## 6. Natural Language Variations

### 6.1 Estimate Synonyms

| Expression | Maps To |
|------------|---------|
| estimate, forecast, projection | Mean estimate fields |
| consensus, analyst consensus | Mean estimate fields |
| expected, expecting, expect | Mean estimate fields |
| what do analysts think | Estimates + recommendations |
| wall street estimate | Mean estimate fields |

### 6.2 Period Expressions

| Expression | Maps To |
|------------|---------|
| this year, current year, FY | `Period=FY1` |
| next year | `Period=FY2` |
| this quarter, next quarter | `Period=FQ1` |
| last quarter | `Period=FQ0` or `FQ-1` |

### 6.3 Comparison Expressions

| Expression | Intent |
|------------|--------|
| beat, exceeded, topped | Positive surprise |
| missed, fell short | Negative surprise |
| raised, upgraded | Positive revision |
| lowered, cut | Negative revision |

---

## 7. Error Handling

### 7.1 Common Issues

| Scenario | Expected Handling |
|----------|-------------------|
| No analyst coverage | Return null with note |
| IPO / new company | Limited estimate history |
| Foreign company | Check ADR vs local listing |
| Estimate date expired | Note period ended |

---

## 8. Complexity Classification

| Level | Criteria | Examples |
|-------|----------|----------|
| 1 | Single estimate, single company | EPS estimate (Q1-Q8) |
| 2 | Revenue estimates | Revenue forecasts (Q9-Q13) |
| 3 | Multiple metrics | Combined estimates (Q14-Q17) |
| 4 | Dispersion analysis | High/low/count (Q18-Q21) |
| 5 | Earnings surprise | Beat/miss analysis (Q22-Q26) |
| 6 | Revisions | Trend analysis (Q27-Q30) |
| 7 | Recommendations | Buy/hold/sell (Q31-Q35) |
| 8 | Price targets | Target analysis (Q36-Q40) |
| 9 | Forward valuations | PE, PEG (Q41-Q45) |
| 10 | Multi-year forecasts | Long-term (Q46-Q50) |
| 11 | Multi-company | Comparisons (Q51-Q55) |
| 12 | Historical estimates | Time series (Q56-Q58) |
| 13 | Growth analysis | YoY calcs (Q59-Q61) |
| 14 | Complete packages | Full summary (Q62-Q63) |
| 15 | Sector analysis | Industry view (Q64-Q65) |

---

## 9. Screening Queries (Find Companies by Estimates)

### 9.1 SCREEN Expression for Estimates

```python
# Basic pattern - filter by estimate metrics
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), [estimate_filters], CURN=USD)'
get_data([screen_exp], [fields])
```

### 9.2 Level 1: Top N by Estimate Metrics

**Q66:** "Which stocks have the most buy ratings?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.NumBuys, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NumBuys', 'TR.NumHolds', 'TR.NumSells'])
```

**Q67:** "Show stocks with highest expected EPS growth"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.LTGMean, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.LTGMean'])
```

**Q68:** "Find stocks with highest price target upside"
```python
# Get stocks and calculate upside client-side
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.PriceTargetMean>0, CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.PriceClose', 'TR.PriceTargetMean'])
df['Upside'] = (df['PriceTargetMean'] - df['PriceClose']) / df['PriceClose'] * 100
df.sort_values('Upside', ascending=False).head(20)
```

**Q69:** "Get stocks with most analyst coverage"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.NumOfEst, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NumOfEst'])
```

### 9.3 Level 2: Earnings Surprise Screening

**Q70:** "Find stocks that beat earnings last quarter"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.EPSSurprisePct(Period=FQ0)>0, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSSurprisePct(Period=FQ0)'])
```

**Q71:** "Show stocks with biggest positive earnings surprises"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.EPSSurprisePct(Period=FQ0)>0, TOP(TR.EPSSurprisePct(Period=FQ0), 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSSurprisePct(Period=FQ0)'])
```

**Q72:** "Find stocks that missed estimates"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.EPSSurprisePct(Period=FQ0)<0, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSSurprisePct(Period=FQ0)'])
```

**Q73:** "Get stocks that beat both EPS and revenue estimates"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.EPSSurprisePct(Period=FQ0)>0,
    TR.RevenueSurprise(Period=FQ0)>0,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.EPSSurprisePct(Period=FQ0)', 'TR.RevenueSurprise(Period=FQ0)'])
```

### 9.4 Level 3: Estimate Revision Screening

**Q74:** "Find stocks where analysts are raising estimates"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.MeanPctChg(Period=FY1,WP=30d)>5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
```

**Q75:** "Show stocks with most analyst upgrades"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.EPSNumUp(Period=FY1), 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSNumUp(Period=FY1)', 'TR.EPSNumDown(Period=FY1)'])
```

**Q76:** "Find stocks where estimates are being cut"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.MeanPctChg(Period=FY1,WP=30d)<-5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
```

**Q77:** "Get stocks with positive revision ratio (more ups than downs)"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.EPSNumUp(Period=FY1)>TR.EPSNumDown(Period=FY1), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EPSNumUp(Period=FY1)', 'TR.EPSNumDown(Period=FY1)'])
```

### 9.5 Level 4: Recommendation Screening

**Q78:** "Find strong buy stocks (consensus rating near 1)"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.RecMean<=1.5, TR.NumOfEst>=5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.RecMean', 'TR.NumBuys', 'TR.NumOfEst'])
```

**Q79:** "Show stocks with unanimous buy ratings"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.NumSells=0, TR.NumHolds=0, TR.NumBuys>=5, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NumBuys'])
```

**Q80:** "Find contrarian opportunities (many sells but improving)"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.RecMean>=3.5,
    TR.MeanPctChg(Period=FY1,WP=30d)>0,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.RecMean', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
```

### 9.6 Level 5: Valuation + Estimates Combined

**Q81:** "Find cheap stocks with strong growth estimates (GARP)"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.PEGRatio>0, TR.PEGRatio<=1.5,
    TR.LTGMean>=10,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.PEGRatio', 'TR.LTGMean', 'TR.PE'])
```

**Q82:** "Show undervalued stocks with positive estimate revisions"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.PE>0, TR.PE<=15,
    TR.MeanPctChg(Period=FY1,WP=30d)>0,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.PE', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
```

**Q83:** "Find stocks with low forward PE and high growth"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.PtoEPSMeanEst(Period=FY1)>0,
    TR.PtoEPSMeanEst(Period=FY1)<=20,
    TR.LTGMean>=15,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.PtoEPSMeanEst(Period=FY1)', 'TR.LTGMean'])
```

### 9.7 Level 6: Index-Based Estimate Screens

**Q84:** "Which S&P 500 stocks have the best analyst ratings?"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.RecMean', 'TR.NumBuys'])
df.sort_values('RecMean', ascending=True).head(10)  # Lower is better (1=Buy, 5=Sell)
```

**Q85:** "Find S&P 500 stocks with rising estimates"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.MeanPctChg(Period=FY1,WP=30d)'])
df.sort_values('MeanPctChg', ascending=False).head(20)
```

---

## Sources

- [LSEG I/B/E/S Estimates](https://www.lseg.com/en/data-analytics/financial-data/company-data/ibes-estimates)
- [LSEG Developer Community - Estimates](https://community.developers.refinitiv.com/questions/73493/get-eps-historical-data-for-stocks.html)
- [Datastream IBES Guide](https://developers.lseg.com/en/article-catalog/article/how-to-collect-datastream-ibes-global-aggregate-earnings-data-with-python-and-codebook)
- [Refinitiv Data Library for Python](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-library-for-python)
