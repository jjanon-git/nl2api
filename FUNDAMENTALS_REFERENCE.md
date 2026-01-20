# LSEG Fundamentals API Reference for Eval Platform

> **Purpose:** Maps natural language queries to expected Fundamentals API tool calls.
> **Target API:** LSEG Worldscope Fundamentals (via Datastream) & Refinitiv Data Library
> **Version:** 1.0.0

---

## 1. API Access Methods

### 1.1 Two Primary Interfaces

| Interface | Field Format | Use Case | Library |
|-----------|-------------|----------|---------|
| Datastream (DSWS) | `WC*****` | Historical fundamentals, time series | `DatastreamPy` |
| Refinitiv Data Library | `TR.*` | Real-time, estimates, broader coverage | `lseg-data` |

### 1.2 Basic Usage

**Datastream (Worldscope):**
```python
import DatastreamPy as dsws
ds = dsws.Datastream(username='****', password='****')
df = ds.get_data(tickers='@AAPL', fields=['WC01001', 'WC01751'], start='-5Y', end='0D', freq='Y')
```

**Refinitiv Data Library:**
```python
import lseg.data as ld
ld.open_session()
df = ld.get_data(['AAPL.O'], ['TR.Revenue', 'TR.NetIncome'], {'SDate': '-5Y', 'EDate': '0D', 'Frq': 'FY'})
```

---

## 2. Worldscope Field Code Reference (WC Codes)

### 2.1 Income Statement

#### Revenue & Sales

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| revenue, sales, net sales, turnover | `WC01001` | Net Sales Or Revenues |
| gross revenue, total revenue | `WC01001` | Net Sales Or Revenues |
| premiums earned (insurance) | `WC01002` | Premiums Earned |
| interest income (banks) | `WC01016` | Interest Income Total |
| net interest income | `WC01076` | Net Interest Income |

#### Cost & Expenses

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| cost of goods sold, COGS, cost of sales | `WC01051` | Cost Of Goods Sold (Excl Depreciation) |
| interest expense | `WC01075` | Interest Expense Total |
| interest expense on debt | `WC01251` | Interest Expense On Debt |
| salaries, employee costs, wages | `WC01084` | Salaries And Benefits Expenses |
| SG&A, selling general admin | `WC01101` | Selling, General & Administrative Expenses |
| R&D, research and development | `WC01201` | Research & Development |
| depreciation, D&A | `WC01151` | Depreciation, Depletion And Amortization |
| amortization of intangibles | `WC01149` | Amortization Of Intangibles |
| operating expenses | `WC01249` | Operating Expenses Total |
| provision for loan losses | `WC01271` | Provision For Loan Losses |

#### Profit Metrics

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| gross profit, gross income | `WC01100` | Gross Income |
| operating income, operating profit, EBIT | `WC01250` | Operating Income |
| pretax income, EBT, income before tax | `WC01401` | Pretax Income |
| income taxes, tax expense | `WC01451` | Income Taxes |
| minority interest (P&L) | `WC01501` | Minority Interest Income Statement |
| net income before extraordinary items | `WC01551` | Net Income Before Extra Items/Preferred Dividends |
| extraordinary items | `WC01601` | Extra Items & Gain/Loss Sale Of Assets |
| net income before preferred dividends | `WC01651` | Net Income Before Preferred Dividends |
| preferred dividends | `WC01701` | Preferred Dividend Requirements |
| net income, earnings, profit | `WC01751` | Net Income Available To Common |

#### Additional Income Statement Items

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| non-operating interest income | `WC01266` | Non-Operating Interest Income |
| extraordinary credit (pretax) | `WC01253` | Extraordinary Credit Pretax |
| extraordinary charge (pretax) | `WC01254` | Extraordinary Charge Pretax |
| capitalized interest | `WC01255` | Interest Capitalized |
| net income for basic EPS | `WC01706` | Net Income After Preferred Dividends (Basic EPS) |
| net income for diluted EPS | `WC01705` | Net Income Used To Calculate Fully Diluted EPS |

---

### 2.2 Balance Sheet - Assets

#### Cash & Investments

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| cash and short-term investments | `WC02001` | Cash & Short Term Investments |
| cash, cash only | `WC02003` | Cash |
| cash and due from banks | `WC02004` | Cash & Due From Banks |
| cash equivalents | `WC02005` | Cash & Equivalents Generic |
| other investments | `WC02250` | Other Investments |
| total investments | `WC02255` | Investments Total |

#### Receivables & Inventory

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| receivables, accounts receivable | `WC02051` | Receivables (Net) |
| inventory, inventories | `WC02101` | Inventories Total |
| loans (net) | `WC02276` | Loans Net |

#### Fixed & Other Assets

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| PP&E, property plant equipment, fixed assets | `WC02501` | Property, Plant And Equipment Net |
| intangibles, intangible assets | `WC02649` | Total Intangible Other Assets Net |
| current assets | `WC02201` | Current Assets Total |
| total assets | `WC02999` | Total Assets |

---

### 2.3 Balance Sheet - Liabilities

#### Current Liabilities

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| accounts payable, payables | `WC03040` | Accounts Payable |
| short-term debt, current portion of LT debt | `WC03051` | Short Term Debt & Current Portion Of Long Term Debt |
| current liabilities | `WC03101` | Current Liabilities Total |
| deposits (banks) | `WC03019` | Deposits Total |
| insurance reserves | `WC03030` | Insurance Reserves Total |

#### Long-Term Liabilities

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| long-term debt, LT debt | `WC03251` | Long Term Debt |
| total debt | `WC03255` | Total Debt |
| deferred taxes | `WC03263` | Deferred Taxes |
| total liabilities | `WC03351` | Total Liabilities |

---

### 2.4 Balance Sheet - Equity

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| minority interest (balance sheet) | `WC03426` | Minority Interest Balance Sheet |
| preferred stock | `WC03451` | Preferred Stock |
| common stock | `WC03480` | Common Stock |
| common equity, shareholders equity | `WC03501` | Common Equity |
| total shareholders equity | `WC03995` | Total Shareholders Equity |
| total capital | `WC03998` | Total Capital |
| total liabilities and equity | `WC03999` | Total Liabilities & Shareholders' Equity |
| working capital | `WC03151` | Working Capital |

---

### 2.5 Cash Flow Statement

#### Operating Activities

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| funds from operations, FFO | `WC04201` | Funds From Operations |
| operating cash flow, CFO | `WC04860` | Net Cash Flow Operating Activities |
| depreciation and depletion | `WC04049` | Depreciation And Depletion |
| amortization of intangibles (CF) | `WC04050` | Amortization Of Intangible Assets |

#### Investing Activities

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| capital expenditures, capex | `WC04601` | Capital Expenditures (Additions To Fixed Assets) |
| investing cash flow, CFI | `WC04870` | Net Cash Flow Investing |

#### Financing Activities

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| dividends paid | `WC04551` | Cash Dividends Paid Total |
| stock issuance, equity issued | `WC04251` | Net Proceeds From Sale/Issue Of Common & Preferred |
| financing cash flow, CFF | `WC04890` | Net Cash Flow Financing |
| change in cash | `WC04851` | Increase/Decrease In Cash & Short Term Investments |

---

### 2.6 Per Share Data

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| EPS, earnings per share | `WC05201` | Earnings Per Share |
| basic EPS | `WC05210` | Earnings Per Share Basic Year |
| diluted EPS, fully diluted EPS | `WC05290` | Fully Diluted Earnings Per Share Year |
| EPS fiscal year end | `WC05202` | Earnings Per Share Fiscal Year End |
| EPS as reported | `WC18193` | Earnings Per Share As Reported |
| EPS including extraordinary | `WC18209` | Earnings Per Share Including Extraordinary Items Fiscal |
| dividends per share, DPS | `WC05101` | Dividends Per Share |
| DPS fiscal | `WC05110` | Dividends Per Share Fiscal |
| book value per share, BVPS | `WC05476` | Book Value Per Share |
| cash flow per share, CFPS | `WC05501` | Cash Flow Per Share (Security) |
| CFPS fiscal | `WC05502` | Cash Flow Per Share Fiscal |
| sales per share, revenue per share | `WC05508` | Sales Per Share |
| shares outstanding | `WC05301` | Common Shares Outstanding |
| shares for basic EPS | `WC05192` | Common Shares Used To Calculate Basic EPS |
| shares for diluted EPS | `WC05194` | Common Shares Used To Calculate Fully Diluted EPS |
| par value | `WC05309` | Par Value |
| common dividends (cash) | `WC05376` | Common Dividends (Cash) |
| stock split ratio | `WC05576` | Stock Split/Dividend Ratio |

---

### 2.7 Financial Ratios

#### Valuation Ratios

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| market cap, market capitalization | `WC08001` | Market Capitalization |
| enterprise value, EV | `WC18100` | Enterprise Value |

#### Profitability Ratios

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| ROE, return on equity | `WC08301` | Return On Equity Total % |
| ROA, return on assets | `WC08326` | Return On Assets |
| ROIC, return on invested capital | `WC08376` | Return On Invested Capital |
| operating margin, operating profit margin | `WC08316` | Operating Profit Margin |
| pretax margin | `WC08321` | Pretax Margin |
| net margin, profit margin | `WC08366` | Net Margin |
| cash flow to sales | `WC08311` | Cash Flow/Sales |

#### Liquidity Ratios

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| current ratio | `WC08106` | Current Ratio |
| quick ratio, acid test | `WC08101` | Quick Ratio |

#### Leverage Ratios

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| debt to capital, debt/capital | `WC08221` | Total Debt % Total Capital |
| debt to equity, D/E | `WC08224` | Total Debt % Common Equity |

#### Efficiency Ratios

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| days receivable, DSO | `WC08131` | Accounts Receivables Days |
| days inventory, DIO | `WC08126` | Inventories Days Held |
| dividend payout | `WC09504` | Dividend Payout Per Share |

---

### 2.8 Additional Data Items

#### EBITDA & Related

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| EBIT | `WC18191` | Earnings Before Interest And Taxes (EBIT) |
| EBITDA | `WC18198` | Earnings Before Interest, Taxes & Depreciation (EBITDA) |
| net debt | `WC18199` | Net Debt |

#### Company Information

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| employees, number of employees | `WC07011` | Employees |
| company name | `WC06001` | Company Name (Security) |
| industry classification | `WC06010` | General Industry Classification |
| business description | `WC06091` | Business Description |
| country | `WC06026` | Nation (Security) |
| fiscal year end | `WC05350` | Date Of Fiscal Year End |
| accounting standards | `WC07536` | Accounting Standards Followed |
| currency | `WC06099` | Currency Of Document |

#### Goodwill & Impairment

| Natural Language | WC Code | Description |
|-----------------|---------|-------------|
| goodwill amortization/impairment | `WC18224` | Amortization & Impairment Of Goodwill |
| goodwill impairment | `WC18225` | Impairment Of Goodwill |
| intangibles impairment | `WC18226` | Impairment Of Other Intangibles |

---

## 3. Refinitiv TR Field Code Reference

### 3.1 Income Statement (TR Codes)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| revenue, sales | `TR.Revenue` | Total Revenue |
| gross profit | `TR.GrossProfit` | Gross Profit |
| COGS, cost of goods sold | `TR.CostofRevenueTotal` | Cost of Revenue Total |
| SG&A expenses | `TR.SGandAExp` | SG&A Expenses |
| R&D | `TR.ResearchAndDevelopment` | Research & Development |
| depreciation & amortization | `TR.DepreciationAmort` | Depreciation & Amortization |
| operating expenses | `TR.TotalOperatingExpense` | Total Operating Expense |
| operating income | `TR.OperatingIncome` | Operating Income |
| EBIT | `TR.EBIT` | Earnings Before Interest & Taxes |
| EBITDA | `TR.EBITDA` | EBITDA |
| interest expense | `TR.InterestExpense` | Interest Expense |
| pretax income | `TR.NetIncomeBeforeTaxes` | Net Income Before Taxes |
| income tax | `TR.IncomeTaxExpense` | Income Tax Expense |
| net income | `TR.NetIncome` | Net Income |
| net income after taxes | `TR.NetIncomeAfterTaxes` | Net Income After Taxes |

### 3.2 Balance Sheet (TR Codes)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| cash | `TR.Cash` | Cash |
| cash and ST investments | `TR.CashAndSTInvestments` | Cash & Short-Term Investments |
| receivables | `TR.TotalReceivablesNet` | Total Receivables Net |
| inventory | `TR.Inventories` | Inventories |
| current assets | `TR.CurrentAssets` | Current Assets |
| total assets | `TR.TotalAssets` | Total Assets |
| total assets (reported) | `TR.TotalAssetsReported` | Total Assets Reported |
| accounts payable | `TR.AccountsPayable` | Accounts Payable |
| current liabilities | `TR.CurrentLiabilities` | Current Liabilities |
| long-term debt | `TR.LongTermDebt` | Long-Term Debt |
| total debt | `TR.TotalDebtOutstanding` | Total Debt Outstanding |
| total liabilities | `TR.TotalLiabilities` | Total Liabilities |
| preferred stock | `TR.PreferredStockNet` | Preferred Stock Net |
| common equity | `TR.CommonEquity` | Common Equity |
| total equity | `TR.TotalEquity` | Total Equity |
| minority interest | `TR.MinorityInterestBSStmt` | Minority Interest (Balance Sheet) |
| total liabilities & equity | `TR.TtlLiabShareholderEqty` | Total Liabilities & Shareholders Equity |

### 3.3 Cash Flow (TR Codes)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| operating cash flow | `TR.OperatingCashFlow` | Operating Cash Flow |
| capex | `TR.CapitalExpenditures` | Capital Expenditures |
| free cash flow | `TR.FreeCashFlow` | Free Cash Flow |
| dividends paid | `TR.DividendsPaid` | Dividends Paid |

### 3.4 Per Share Data (TR Codes)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| EPS | `TR.BasicNormalizedEps` | Basic Normalized EPS |
| EPS basic | `TR.BasicEPS` | Basic EPS |
| EPS diluted | `TR.DilutedNormalizedEps` | Fully Diluted Normalized EPS |
| book value per share | `TR.BookValuePerShare` | Book Value Per Share |
| cash flow per share | `TR.CFPSActValue` | Cash Flow Per Share |
| FCF per share | `TR.FCFPSActValue` | Free Cash Flow Per Share |
| dividends per share | `TR.DpsCommonStock` | Dividends Per Share |
| sales per share | `TR.SalesPerShare` | Sales Per Share |
| shares outstanding | `TR.SharesOutstanding` | Shares Outstanding |

### 3.5 Valuation & Ratios (TR Codes)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| market cap | `TR.CompanyMarketCap` | Market Capitalization |
| enterprise value | `TR.EV` | Enterprise Value |
| PE ratio | `TR.PE` | Price/Earnings Ratio |
| forward PE | `TR.PtoEPSMeanEst(Period=FY1)` | Forward P/E (FY1 Estimate) |
| price to book | `TR.PriceToBVPerShare` | Price to Book Value |
| price to sales | `TR.PriceToSalesPerShare` | Price to Sales |
| price to cash flow | `TR.PricetoCFPerShare` | Price to Cash Flow |
| EV/EBITDA | `TR.EVToEBITDA` | EV to EBITDA |
| dividend yield | `TR.DividendYield` | Dividend Yield |
| ROE | `TR.ROEMean` | Return on Equity |
| ROA | `TR.ROAMean` | Return on Assets |
| current ratio | `TR.CurrentRatio` | Current Ratio |
| quick ratio | `TR.QuickRatio` | Quick Ratio |
| debt to equity | `TR.DebtToEquity` | Debt to Equity |
| net debt to EBITDA | `TR.NetDebtToEBITDA` | Net Debt to EBITDA |
| gross margin | `TR.GrossMargin` | Gross Margin |
| operating margin | `TR.OperatingMargin` | Operating Margin |
| net margin | `TR.NetProfitMargin` | Net Profit Margin |
| beta | `TR.Beta` | Beta |

### 3.6 Company Information (TR Codes)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| company name | `TR.CommonName` | Common Name |
| ticker | `TR.TickerSymbol` | Ticker Symbol |
| exchange | `TR.ExchangeName` | Exchange Name |
| country | `TR.HeadquartersCountry` | Headquarters Country |
| sector | `TR.GICSSector` | GICS Sector |
| industry | `TR.TRBCIndustryGroup` | Industry Group |
| employees | `TR.Employees` | Number of Employees |

---

## 4. Question-Answer Reference (Test Cases)

### 4.1 Level 1: Basic Single-Metric Queries

#### Revenue & Sales

**Q1:** "What is Apple's annual revenue?"
```python
# Datastream
get_data(tickers='@AAPL', fields=['WC01001'], start='0D', end='0D', kind=0)
# Refinitiv
get_data(['AAPL.O'], ['TR.Revenue'])
```

**Q2:** "Get Microsoft's total sales for 2024"
```python
get_data(tickers='U:MSFT', fields=['WC01001'], start='2024-01-01', end='2024-12-31', freq='Y')
```

**Q3:** "Show me Google's revenue"
```python
get_data(tickers='U:GOOGL', fields=['WC01001'], start='0D', end='0D', kind=0)
```

**Q4:** "What are Amazon's net sales?"
```python
get_data(tickers='U:AMZN', fields=['WC01001'], start='0D', end='0D', kind=0)
```

**Q5:** "Get Tesla's turnover"
```python
get_data(tickers='U:TSLA', fields=['WC01001'], start='0D', end='0D', kind=0)
```

#### Profitability

**Q6:** "What is Apple's net income?"
```python
get_data(tickers='@AAPL', fields=['WC01751'], start='0D', end='0D', kind=0)
```

**Q7:** "Get Microsoft's operating income"
```python
get_data(tickers='U:MSFT', fields=['WC01250'], start='0D', end='0D', kind=0)
```

**Q8:** "Show Amazon's gross profit"
```python
get_data(tickers='U:AMZN', fields=['WC01100'], start='0D', end='0D', kind=0)
```

**Q9:** "What is Google's EBITDA?"
```python
get_data(tickers='U:GOOGL', fields=['WC18198'], start='0D', end='0D', kind=0)
```

**Q10:** "Get Tesla's pretax income"
```python
get_data(tickers='U:TSLA', fields=['WC01401'], start='0D', end='0D', kind=0)
```

#### Balance Sheet Basics

**Q11:** "What are Apple's total assets?"
```python
get_data(tickers='@AAPL', fields=['WC02999'], start='0D', end='0D', kind=0)
```

**Q12:** "Get Microsoft's total debt"
```python
get_data(tickers='U:MSFT', fields=['WC03255'], start='0D', end='0D', kind=0)
```

**Q13:** "What is Amazon's cash position?"
```python
get_data(tickers='U:AMZN', fields=['WC02001'], start='0D', end='0D', kind=0)
```

**Q14:** "Show Google's shareholders equity"
```python
get_data(tickers='U:GOOGL', fields=['WC03501'], start='0D', end='0D', kind=0)
```

**Q15:** "Get Tesla's total liabilities"
```python
get_data(tickers='U:TSLA', fields=['WC03351'], start='0D', end='0D', kind=0)
```

---

### 4.2 Level 2: Per Share & Basic Ratios

#### Per Share Metrics

**Q16:** "What is Apple's earnings per share?"
```python
get_data(tickers='@AAPL', fields=['WC05201'], start='0D', end='0D', kind=0)
```

**Q17:** "Get Microsoft's diluted EPS"
```python
get_data(tickers='U:MSFT', fields=['WC05290'], start='0D', end='0D', kind=0)
```

**Q18:** "What is Amazon's book value per share?"
```python
get_data(tickers='U:AMZN', fields=['WC05476'], start='0D', end='0D', kind=0)
```

**Q19:** "Get Tesla's dividends per share"
```python
get_data(tickers='U:TSLA', fields=['WC05101'], start='0D', end='0D', kind=0)
```

**Q20:** "What is Google's cash flow per share?"
```python
get_data(tickers='U:GOOGL', fields=['WC05501'], start='0D', end='0D', kind=0)
```

#### Basic Ratios

**Q21:** "What is Apple's return on equity?"
```python
get_data(tickers='@AAPL', fields=['WC08301'], start='0D', end='0D', kind=0)
```

**Q22:** "Get Microsoft's ROA"
```python
get_data(tickers='U:MSFT', fields=['WC08326'], start='0D', end='0D', kind=0)
```

**Q23:** "What is Amazon's current ratio?"
```python
get_data(tickers='U:AMZN', fields=['WC08106'], start='0D', end='0D', kind=0)
```

**Q24:** "Get Tesla's debt to equity ratio"
```python
get_data(tickers='U:TSLA', fields=['WC08224'], start='0D', end='0D', kind=0)
```

**Q25:** "What is Google's operating margin?"
```python
get_data(tickers='U:GOOGL', fields=['WC08316'], start='0D', end='0D', kind=0)
```

---

### 4.3 Level 3: Historical Time Series

#### Revenue History

**Q26:** "Show Apple's revenue for the last 10 years"
```python
get_data(tickers='@AAPL', fields=['WC01001'], start='-10Y', end='0D', freq='Y')
```

**Q27:** "Get Microsoft's quarterly revenue since 2020"
```python
get_data(tickers='U:MSFT', fields=['WC01001'], start='2020-01-01', end='0D', freq='Q')
```

**Q28:** "Show Amazon's annual sales history"
```python
get_data(tickers='U:AMZN', fields=['WC01001'], start='-15Y', end='0D', freq='Y')
```

#### Earnings History

**Q29:** "Get Apple's EPS for the last 5 years"
```python
get_data(tickers='@AAPL', fields=['WC05201'], start='-5Y', end='0D', freq='Y')
```

**Q30:** "Show Microsoft's net income history since 2015"
```python
get_data(tickers='U:MSFT', fields=['WC01751'], start='2015-01-01', end='0D', freq='Y')
```

**Q31:** "Get Google's quarterly earnings for 2023 and 2024"
```python
get_data(tickers='U:GOOGL', fields=['WC01751'], start='2023-01-01', end='2024-12-31', freq='Q')
```

#### Balance Sheet History

**Q32:** "Show Apple's total assets over the past decade"
```python
get_data(tickers='@AAPL', fields=['WC02999'], start='-10Y', end='0D', freq='Y')
```

**Q33:** "Get Tesla's debt levels annually since IPO"
```python
get_data(tickers='U:TSLA', fields=['WC03255'], start='2010-01-01', end='0D', freq='Y')
```

**Q34:** "Show Amazon's cash position quarterly"
```python
get_data(tickers='U:AMZN', fields=['WC02001'], start='-3Y', end='0D', freq='Q')
```

#### Ratio History

**Q35:** "Get Apple's ROE history for 10 years"
```python
get_data(tickers='@AAPL', fields=['WC08301'], start='-10Y', end='0D', freq='Y')
```

---

### 4.4 Level 4: Multiple Metrics

#### Income Statement Package

**Q36:** "Get Apple's full P&L summary: revenue, gross profit, operating income, and net income"
```python
get_data(tickers='@AAPL', fields=['WC01001', 'WC01100', 'WC01250', 'WC01751'], start='0D', end='0D', kind=0)
```

**Q37:** "Show Microsoft's revenue, EBITDA, and net income for the last 5 years"
```python
get_data(tickers='U:MSFT', fields=['WC01001', 'WC18198', 'WC01751'], start='-5Y', end='0D', freq='Y')
```

**Q38:** "Get Amazon's costs breakdown: COGS, SG&A, R&D, and depreciation"
```python
get_data(tickers='U:AMZN', fields=['WC01051', 'WC01101', 'WC01201', 'WC01151'], start='0D', end='0D', kind=0)
```

#### Balance Sheet Package

**Q39:** "Show Apple's balance sheet summary: assets, liabilities, and equity"
```python
get_data(tickers='@AAPL', fields=['WC02999', 'WC03351', 'WC03501'], start='0D', end='0D', kind=0)
```

**Q40:** "Get Tesla's liquidity metrics: cash, current assets, current liabilities"
```python
get_data(tickers='U:TSLA', fields=['WC02001', 'WC02201', 'WC03101'], start='0D', end='0D', kind=0)
```

**Q41:** "Show Microsoft's capital structure: debt, equity, and total capital"
```python
get_data(tickers='U:MSFT', fields=['WC03255', 'WC03501', 'WC03998'], start='0D', end='0D', kind=0)
```

#### Cash Flow Package

**Q42:** "Get Amazon's cash flow summary: operating, investing, and financing"
```python
get_data(tickers='U:AMZN', fields=['WC04860', 'WC04870', 'WC04890'], start='0D', end='0D', kind=0)
```

**Q43:** "Show Google's free cash flow components: operating cash flow and capex"
```python
get_data(tickers='U:GOOGL', fields=['WC04860', 'WC04601'], start='-5Y', end='0D', freq='Y')
```

#### Profitability Ratios Package

**Q44:** "Get Apple's profitability metrics: ROE, ROA, ROIC, and margins"
```python
get_data(tickers='@AAPL', fields=['WC08301', 'WC08326', 'WC08376', 'WC08316', 'WC08366'], start='0D', end='0D', kind=0)
```

**Q45:** "Show Tesla's margin profile: gross, operating, pretax, and net margins"
```python
get_data(tickers='U:TSLA', fields=['WC08316', 'WC08321', 'WC08366'], start='-5Y', end='0D', freq='Y')
```

---

### 4.5 Level 5: Multi-Company Comparisons

#### Tech Giants Comparison

**Q46:** "Compare revenue for Apple, Microsoft, Google, Amazon, and Meta"
```python
get_data(tickers='@AAPL,U:MSFT,U:GOOGL,U:AMZN,U:META', fields=['WC01001'], start='0D', end='0D', kind=0)
```

**Q47:** "Show net income comparison for FAANG stocks"
```python
get_data(tickers='U:META,@AAPL,U:AMZN,U:NFLX,U:GOOGL', fields=['WC01751'], start='0D', end='0D', kind=0)
```

**Q48:** "Compare EPS for the magnificent seven tech stocks"
```python
get_data(tickers='@AAPL,U:MSFT,U:GOOGL,U:AMZN,U:META,U:NVDA,U:TSLA', fields=['WC05201'], start='0D', end='0D', kind=0)
```

#### Sector Comparisons

**Q49:** "Compare ROE for major US banks: JPM, BAC, WFC, C, GS"
```python
get_data(tickers='U:JPM,U:BAC,U:WFC,U:C,U:GS', fields=['WC08301'], start='0D', end='0D', kind=0)
```

**Q50:** "Show debt to equity ratios for auto manufacturers"
```python
get_data(tickers='U:GM,U:F,U:TSLA,D:BMW,J:7203', fields=['WC08224'], start='0D', end='0D', kind=0)
```

**Q51:** "Compare EBITDA margins for oil majors"
```python
get_data(tickers='U:XOM,U:CVX,SHEL,BP.', fields=['WC18198', 'WC01001'], start='0D', end='0D', kind=0)
```

**Q52:** "Get revenue and net income for pharmaceutical companies"
```python
get_data(tickers='U:JNJ,U:PFE,U:MRK,U:ABBV,U:LLY', fields=['WC01001', 'WC01751'], start='0D', end='0D', kind=0)
```

#### Historical Comparison

**Q53:** "Compare revenue growth for Apple vs Microsoft over 10 years"
```python
get_data(tickers='@AAPL,U:MSFT', fields=['WC01001'], start='-10Y', end='0D', freq='Y')
```

**Q54:** "Show ROE trends for major banks since 2015"
```python
get_data(tickers='U:JPM,U:BAC,U:WFC,U:GS', fields=['WC08301'], start='2015-01-01', end='0D', freq='Y')
```

**Q55:** "Compare cash positions of tech giants quarterly for past 2 years"
```python
get_data(tickers='@AAPL,U:MSFT,U:GOOGL', fields=['WC02001'], start='-2Y', end='0D', freq='Q')
```

---

### 4.6 Level 6: Complete Financial Statements

#### Full Income Statement

**Q56:** "Get Apple's complete income statement"
```python
get_data(tickers='@AAPL', fields=[
    'WC01001',  # Revenue
    'WC01051',  # COGS
    'WC01100',  # Gross Profit
    'WC01101',  # SG&A
    'WC01201',  # R&D
    'WC01151',  # D&A
    'WC01250',  # Operating Income
    'WC01251',  # Interest Expense
    'WC01401',  # Pretax Income
    'WC01451',  # Taxes
    'WC01751'   # Net Income
], start='0D', end='0D', kind=0)
```

**Q57:** "Show Microsoft's detailed P&L for the last 3 years"
```python
get_data(tickers='U:MSFT', fields=[
    'WC01001', 'WC01100', 'WC01101', 'WC01201', 'WC01250',
    'WC18198', 'WC01401', 'WC01451', 'WC01751'
], start='-3Y', end='0D', freq='Y')
```

#### Full Balance Sheet

**Q58:** "Get Amazon's complete balance sheet"
```python
get_data(tickers='U:AMZN', fields=[
    # Assets
    'WC02001',  # Cash
    'WC02051',  # Receivables
    'WC02101',  # Inventory
    'WC02201',  # Current Assets
    'WC02501',  # PP&E
    'WC02999',  # Total Assets
    # Liabilities
    'WC03040',  # Accounts Payable
    'WC03051',  # Short-term Debt
    'WC03101',  # Current Liabilities
    'WC03251',  # Long-term Debt
    'WC03351',  # Total Liabilities
    # Equity
    'WC03501',  # Common Equity
    'WC03999'   # Total Liabilities & Equity
], start='0D', end='0D', kind=0)
```

**Q59:** "Show Tesla's balance sheet trends over 5 years"
```python
get_data(tickers='U:TSLA', fields=[
    'WC02999', 'WC02001', 'WC03255', 'WC03501', 'WC03151'
], start='-5Y', end='0D', freq='Y')
```

#### Full Cash Flow Statement

**Q60:** "Get Google's complete cash flow statement"
```python
get_data(tickers='U:GOOGL', fields=[
    'WC04201',  # Funds from Operations
    'WC04049',  # Depreciation
    'WC04860',  # Operating Cash Flow
    'WC04601',  # Capex
    'WC04870',  # Investing Cash Flow
    'WC04551',  # Dividends Paid
    'WC04890',  # Financing Cash Flow
    'WC04851'   # Change in Cash
], start='0D', end='0D', kind=0)
```

---

### 4.7 Level 7: Industry-Specific Queries

#### Banks & Financial Services

**Q61:** "Get JP Morgan's net interest income"
```python
get_data(tickers='U:JPM', fields=['WC01076'], start='0D', end='0D', kind=0)
```

**Q62:** "Show Bank of America's provision for loan losses"
```python
get_data(tickers='U:BAC', fields=['WC01271'], start='-5Y', end='0D', freq='Y')
```

**Q63:** "Get Goldman Sachs' interest income and expense"
```python
get_data(tickers='U:GS', fields=['WC01016', 'WC01075'], start='0D', end='0D', kind=0)
```

**Q64:** "Show Wells Fargo's deposits and loans"
```python
get_data(tickers='U:WFC', fields=['WC03019', 'WC02276'], start='0D', end='0D', kind=0)
```

#### Insurance Companies

**Q65:** "Get Berkshire Hathaway's premiums earned"
```python
get_data(tickers='U:BRK.A', fields=['WC01002'], start='0D', end='0D', kind=0)
```

**Q66:** "Show Progressive's insurance reserves"
```python
get_data(tickers='U:PGR', fields=['WC03030'], start='-5Y', end='0D', freq='Y')
```

#### Retail & Consumer

**Q67:** "Get Walmart's inventory turnover metrics"
```python
get_data(tickers='U:WMT', fields=['WC02101', 'WC08126'], start='0D', end='0D', kind=0)
```

**Q68:** "Show Costco's days receivable and inventory"
```python
get_data(tickers='U:COST', fields=['WC08131', 'WC08126'], start='0D', end='0D', kind=0)
```

---

### 4.8 Level 8: Financial Analysis Packages

#### DuPont Analysis

**Q69:** "Get components for Apple's DuPont ROE analysis: profit margin, asset turnover, leverage"
```python
get_data(tickers='@AAPL', fields=[
    'WC08366',  # Net Margin
    'WC01001',  # Revenue (for asset turnover calculation)
    'WC02999',  # Total Assets
    'WC03501',  # Equity
    'WC08301'   # ROE
], start='0D', end='0D', kind=0)
```

#### Solvency Analysis

**Q70:** "Analyze Tesla's solvency: debt ratios and interest coverage"
```python
get_data(tickers='U:TSLA', fields=[
    'WC03255',  # Total Debt
    'WC03501',  # Equity
    'WC02999',  # Total Assets
    'WC08221',  # Debt/Capital
    'WC08224',  # Debt/Equity
    'WC01250',  # Operating Income (for interest coverage)
    'WC01251'   # Interest Expense
], start='0D', end='0D', kind=0)
```

#### Liquidity Analysis

**Q71:** "Assess Amazon's liquidity: working capital and quick ratio components"
```python
get_data(tickers='U:AMZN', fields=[
    'WC02001',  # Cash
    'WC02051',  # Receivables
    'WC02101',  # Inventory
    'WC02201',  # Current Assets
    'WC03101',  # Current Liabilities
    'WC03151',  # Working Capital
    'WC08101',  # Quick Ratio
    'WC08106'   # Current Ratio
], start='0D', end='0D', kind=0)
```

#### Capital Efficiency

**Q72:** "Analyze Microsoft's capital efficiency: ROIC, ROE, ROA over time"
```python
get_data(tickers='U:MSFT', fields=['WC08376', 'WC08301', 'WC08326'], start='-10Y', end='0D', freq='Y')
```

#### Free Cash Flow Analysis

**Q73:** "Calculate Apple's free cash flow: OCF minus Capex"
```python
get_data(tickers='@AAPL', fields=['WC04860', 'WC04601'], start='-5Y', end='0D', freq='Y')
```

**Q74:** "Get Google's cash flow conversion: net income vs operating cash flow"
```python
get_data(tickers='U:GOOGL', fields=['WC01751', 'WC04860'], start='-5Y', end='0D', freq='Y')
```

---

### 4.9 Level 9: Growth & Trend Analysis

#### Revenue Growth

**Q75:** "Show Apple's revenue growth trend over 10 years"
```python
get_data(tickers='@AAPL', fields=['WC01001'], start='-10Y', end='0D', freq='Y')
# Calculate YoY growth client-side
```

**Q76:** "Compare quarterly revenue growth for FAANG stocks"
```python
get_data(tickers='U:META,@AAPL,U:AMZN,U:NFLX,U:GOOGL', fields=['WC01001'], start='-3Y', end='0D', freq='Q')
```

#### Earnings Growth

**Q77:** "Track Microsoft's EPS growth over 15 years"
```python
get_data(tickers='U:MSFT', fields=['WC05201'], start='-15Y', end='0D', freq='Y')
```

**Q78:** "Show Tesla's net income trajectory since profitability"
```python
get_data(tickers='U:TSLA', fields=['WC01751'], start='2018-01-01', end='0D', freq='Q')
```

#### Margin Trends

**Q79:** "Analyze Amazon's margin expansion: gross, operating, net margins over time"
```python
get_data(tickers='U:AMZN', fields=['WC08316', 'WC08321', 'WC08366'], start='-10Y', end='0D', freq='Y')
```

**Q80:** "Show how Apple's profitability ratios evolved"
```python
get_data(tickers='@AAPL', fields=['WC08301', 'WC08326', 'WC08376'], start='-15Y', end='0D', freq='Y')
```

---

### 4.10 Level 10: Complex Multi-Step Analysis

**Q81:** "Calculate enterprise value components for Apple"
```python
# Step 1: Get market cap, debt, cash
get_data(tickers='@AAPL', fields=['WC08001', 'WC03255', 'WC02001'], start='0D', end='0D', kind=0)
# Step 2: EV = Market Cap + Total Debt - Cash (calculated client-side)
# Or directly: WC18100 (Enterprise Value)
```

**Q82:** "Perform peer analysis: get all key metrics for semiconductor companies"
```python
get_data(tickers='U:NVDA,U:AMD,U:INTC,U:QCOM,U:AVGO', fields=[
    'WC01001', 'WC01751', 'WC08316', 'WC08301', 'WC08224',
    'WC02001', 'WC03255', 'WC08001'
], start='0D', end='0D', kind=0)
```

**Q83:** "Build a valuation model: get EV/EBITDA inputs for tech sector"
```python
get_data(tickers='@AAPL,U:MSFT,U:GOOGL,U:AMZN,U:META', fields=[
    'WC18100',  # Enterprise Value
    'WC18198',  # EBITDA
    'WC08001',  # Market Cap
    'WC01751'   # Net Income (for P/E)
], start='0D', end='0D', kind=0)
```

**Q84:** "Analyze capital allocation: capex, dividends, buybacks for dividend aristocrats"
```python
get_data(tickers='U:JNJ,U:PG,U:KO,U:PEP,U:MMM', fields=[
    'WC04601',  # Capex
    'WC04551',  # Dividends Paid
    'WC04860',  # Operating Cash Flow
    'WC05101'   # DPS
], start='-5Y', end='0D', freq='Y')
```

**Q85:** "Full financial health check for a company"
```python
get_data(tickers='U:AMZN', fields=[
    # Profitability
    'WC01001', 'WC01751', 'WC08301', 'WC08326', 'WC08366',
    # Liquidity
    'WC08106', 'WC08101', 'WC03151',
    # Solvency
    'WC08224', 'WC08221', 'WC18199',
    # Efficiency
    'WC08131', 'WC08126',
    # Cash Flow
    'WC04860', 'WC04601'
], start='0D', end='0D', kind=0)
```

---

## 5. Query Parameters Reference

### 5.1 Time Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `freq` | `D`, `W`, `M`, `Q`, `Y` | Daily, Weekly, Monthly, Quarterly, Yearly |
| `start` | Date or relative | Start of period (`-5Y`, `2020-01-01`) |
| `end` | Date or relative | End of period (`0D`, `2024-12-31`) |
| `kind` | `0`, `1` | 0 = Static/snapshot, 1 = Time series |

### 5.2 Fiscal Period Parameters (TR Fields)

| Parameter | Values | Description |
|-----------|--------|-------------|
| `Period` | `FY0`, `FY-1`, `FY1` | Current year, prior year, next year |
| `Period` | `FQ0`, `FQ-1` | Current quarter, prior quarter |
| `Period` | `LTM` | Last twelve months |
| `Frq` | `FY`, `FQ`, `FS` | Annual, Quarterly, Semi-annual |
| `SDate`/`EDate` | Date range | Start/end date for time series |

### 5.3 Currency & Scaling

| Parameter | Values | Description |
|-----------|--------|-------------|
| `Curn` | `USD`, `EUR`, `GBP` | Convert to currency |
| `Scale` | `0`, `3`, `6`, `9` | Units, thousands, millions, billions |

---

## 6. Natural Language Variations

### 6.1 Revenue Synonyms

| Expression | Maps To |
|------------|---------|
| revenue, sales, net sales, turnover, top line | `WC01001` / `TR.Revenue` |
| total revenue, gross revenue | `WC01001` / `TR.Revenue` |
| income (when context = revenue) | `WC01001` |

### 6.2 Profit Synonyms

| Expression | Maps To |
|------------|---------|
| net income, earnings, profit, bottom line | `WC01751` / `TR.NetIncome` |
| operating profit, operating income, EBIT | `WC01250` / `TR.OperatingIncome` |
| gross profit, gross margin (absolute) | `WC01100` / `TR.GrossProfit` |
| pretax income, EBT | `WC01401` |

### 6.3 Balance Sheet Synonyms

| Expression | Maps To |
|------------|---------|
| assets, total assets | `WC02999` / `TR.TotalAssets` |
| debt, total debt, borrowings | `WC03255` / `TR.TotalDebtOutstanding` |
| equity, shareholders equity, book value | `WC03501` / `TR.TotalEquity` |
| cash, cash position, liquidity | `WC02001` / `TR.Cash` |

### 6.4 Ratio Synonyms

| Expression | Maps To |
|------------|---------|
| ROE, return on equity | `WC08301` / `TR.ROEMean` |
| ROA, return on assets | `WC08326` / `TR.ROAMean` |
| profit margin, net margin | `WC08366` / `TR.NetProfitMargin` |
| D/E, leverage, debt ratio | `WC08224` / `TR.DebtToEquity` |

---

## 7. Error Handling & Edge Cases

### 7.1 Common Data Issues

| Issue | Expected Handling |
|-------|-------------------|
| No data for period | Return null/NaN with note |
| Different fiscal year end | Align to calendar year or note difference |
| Currency mismatch | Convert using exchange rate or note currency |
| Restated vs original | Default to restated unless specified |
| Private company | Return error: "No public filings available" |

### 7.2 Industry-Specific Fields

| Scenario | Notes |
|----------|-------|
| Bank: no inventory | Return null for inventory fields |
| Insurance: premiums vs revenue | Use `WC01002` instead of `WC01001` |
| REIT: FFO instead of net income | Use funds from operations |
| Utility: rate base metrics | Industry-specific fields required |

---

## 8. Complexity Classification

### 8.1 Scoring Rubric

| Factor | Score |
|--------|-------|
| Single field | +0 |
| 2-5 fields | +1 |
| 6+ fields | +2 |
| Single company | +0 |
| 2-5 companies | +1 |
| 6+ companies | +2 |
| Static data | +0 |
| Time series | +1 |
| Multi-year history | +2 |
| Basic metrics | +0 |
| Ratios/calculated | +1 |
| Full statements | +2 |
| Industry-specific | +1 |
| Multi-step analysis | +3 |

### 8.2 Level Assignment

| Total Score | Level |
|-------------|-------|
| 0-1 | Level 1 (Basic) |
| 2-3 | Level 2 (Simple) |
| 4-5 | Level 3 (Moderate) |
| 6-7 | Level 4 (Complex) |
| 8+ | Level 5 (Advanced) |

---

## Appendix A: WC Code Number Ranges

| Range | Category |
|-------|----------|
| WC00xxx | Company status/indicators |
| WC01xxx | Income statement |
| WC02xxx | Balance sheet - Assets |
| WC03xxx | Balance sheet - Liabilities & Equity |
| WC04xxx | Cash flow statement |
| WC05xxx | Per share data |
| WC06xxx | Company information |
| WC07xxx | Dates and indicators |
| WC08xxx | Financial ratios |
| WC09xxx | Additional ratios |
| WC10xxx | Per share (additional) |
| WC11xxx | Indicators and flags |
| WC18xxx | Derived/calculated items (EBITDA, EV, etc.) |

## Appendix B: TR Field Naming Conventions

| Prefix | Category |
|--------|----------|
| `TR.Revenue`, `TR.NetIncome` | Core financials |
| `TR.Total*` | Aggregated items |
| `TR.*PerShare` | Per share metrics |
| `TR.*Margin` | Margin ratios |
| `TR.*Mean` | Analyst estimates |
| `TR.F.*` | Full statement retrieval |
| `TR.BGS.*` | Business/geographic segments |

---

## 9. Screening Queries (Find Companies by Fundamentals)

### 9.1 SCREEN Expression for Fundamentals

```python
# Basic pattern
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), [filters], CURN=USD)'
get_data([screen_exp], [fields])
```

### 9.2 Level 1: Top N by Fundamental Metric

**Q86:** "What are the top 10 companies by revenue?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.Revenue, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.Revenue'])
```

**Q87:** "Show the largest companies by total assets"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.TotalAssets, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.TotalAssets'])
```

**Q88:** "Get the top 5 most profitable companies by net income"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.NetIncome, 5, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NetIncome'])
```

**Q89:** "Which companies have the highest EBITDA?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.EBITDA, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.EBITDA'])
```

**Q90:** "Show top 10 by free cash flow"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TOP(TR.FreeCashFlow, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.FreeCashFlow'])
```

### 9.3 Level 2: Ratio-Based Screening

**Q91:** "Find companies with ROE above 25%"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.ROE>=25, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.ROE'])
```

**Q92:** "Show stocks with highest ROE"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.ROE>0, TOP(TR.ROE, 20, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.ROE'])
```

**Q93:** "Get companies with profit margin over 20%"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.NetProfitMargin>=20, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NetProfitMargin'])
```

**Q94:** "Find low debt companies (debt/equity under 0.5)"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.DebtToEquity<=0.5, TR.DebtToEquity>=0, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.DebtToEquity'])
```

**Q95:** "Show highly liquid companies (current ratio over 2)"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.CurrentRatio>=2, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.CurrentRatio'])
```

### 9.4 Level 3: Sector-Specific Fundamental Screens

**Q96:** "What are the most profitable tech companies?"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), TOP(TR.NetIncome, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NetIncome', 'TR.TRBCIndustry'])
```

**Q97:** "Show banks with highest ROA"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCBusinessSectorCode,"5510"), TOP(TR.ROA, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.ROA'])
```

**Q98:** "Find healthcare companies with revenue over $10B"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"55"), TR.Revenue(Scale=9)>=10, CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.Revenue'])
```

**Q99:** "Get retail companies with best margins"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCBusinessSectorCode,"5330"), TOP(TR.NetProfitMargin, 10, nnumber), CURN=USD)'
get_data([screen_exp], ['TR.CommonName', 'TR.NetProfitMargin'])
```

### 9.5 Level 4: Multi-Criteria Fundamental Screens

**Q100:** "Find quality stocks: high ROE, low debt, positive cash flow"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.ROE>=15,
    TR.DebtToEquity<=1,
    TR.FreeCashFlow>0,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.ROE', 'TR.DebtToEquity', 'TR.FreeCashFlow'])
```

**Q101:** "Show value stocks: low PE, positive earnings, high dividend"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.PE>0, TR.PE<=15,
    TR.NetIncome>0,
    TR.DividendYield>=3,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.PE', 'TR.DividendYield'])
```

**Q102:** "Find growth companies: revenue growth over 20%, positive margins"
```python
screen_exp = '''SCREEN(U(IN(Equity(active,public,primary))),
    TR.RevenueGrowth>=20,
    TR.NetProfitMargin>0,
    CURN=USD)'''
get_data([screen_exp], ['TR.CommonName', 'TR.RevenueGrowth', 'TR.NetProfitMargin'])
```

### 9.6 Level 5: Index-Based Fundamental Screens

**Q103:** "Which S&P 500 stocks have the highest profit margins?"
```python
# Step 1: Get S&P 500 constituents
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
# Step 2: Get margins and sort
df = get_data(constituents, ['TR.CommonName', 'TR.NetProfitMargin'])
df.sort_values('NetProfitMargin', ascending=False).head(10)
```

**Q104:** "What are the highest ROE stocks in the FTSE 100?"
```python
constituents = get_data(tickers='LFTSE100|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.ROE'])
df.sort_values('ROE', ascending=False).head(10)
```

**Q105:** "Find the most cash-rich companies in the DAX"
```python
constituents = get_data(tickers='LDAXINDX|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.Cash'])
df.sort_values('Cash', ascending=False).head(10)
```

---

## Sources

- [Worldscope Fundamentals - LSEG](https://www.lseg.com/en/data-analytics/financial-data/company-data/fundamentals-data/worldscope-fundamentals)
- [LSEG Developer Portal](https://developers.lseg.com/en/api-catalog/eikon/datastream-web-service)
- [Worldscope Data Definitions Guide](https://www.tilburguniversity.edu/sites/default/files/download/WorldScopeDatatypeDefinitionsGuide_2.pdf)
- [Refinitiv Eikon Reference Card](https://www.mq.edu.au/__data/assets/pdf_file/0019/1064134/refinitiv-eikon-microsoft-office-reference-card.pdf)
- [LSEG API Samples - GitHub](https://github.com/LSEG-API-Samples)
