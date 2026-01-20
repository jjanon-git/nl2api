# Datastream API Reference for Eval Platform

> **Purpose:** Maps natural language queries to expected Datastream API tool calls.
> **Target API:** LSEG Datastream Web Service (DSWS)
> **Version:** 1.0.0

---

## 1. Field Code Reference

### 1.1 Price & Trading Data

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| price, stock price, share price, closing price | `P` | Adjusted closing price |
| open, opening price | `PO` | Opening price |
| high, highest price, intraday high | `PH` | High price |
| low, lowest price, intraday low | `PL` | Low price |
| volume, trading volume, shares traded | `VO` | Trading volume |
| bid price, bid | `PB` | Bid price |
| ask price, offer price, ask | `PA` | Ask price |
| last price, latest price | `LP` | Last traded price |
| settlement price | `PS` | Settlement price (futures) |
| index level, index value | `PI` | Price index (for indices) |

### 1.2 Valuation Metrics

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| market cap, market capitalization, market value | `MV` | Market value |
| PE ratio, price to earnings, P/E | `PE` | Price/Earnings ratio |
| PB ratio, price to book, P/B | `PTBV` | Price to book value |
| EV, enterprise value | `EV` | Enterprise value |
| EV/EBITDA | `EVEBID` | EV to EBITDA |
| price to sales, P/S | `PS` | Price to sales ratio |

### 1.3 Dividend Data

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| dividend yield, yield | `DY` | Dividend yield % |
| dividend, dividend per share, DPS | `DPS` | Dividend per share |
| ex-dividend date, ex-date | `EXDT` | Ex-dividend date |
| dividend payment date | `DPDT` | Payment date |
| payout ratio | `POUT` | Dividend payout ratio |

### 1.4 Financial Fundamentals

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| revenue, sales, turnover | `WC01001` | Net sales/revenue |
| net income, earnings, profit | `WC01751` | Net income |
| EPS, earnings per share | `EPS` | Earnings per share |
| EBITDA | `WC18198` | EBITDA |
| operating income, operating profit | `WC01250` | Operating income |
| gross profit, gross margin | `WC01100` | Gross profit |
| total assets | `WC02999` | Total assets |
| total debt | `WC03255` | Total debt |
| cash, cash and equivalents | `WC02001` | Cash & short-term investments |
| free cash flow, FCF | `WC04860` | Free cash flow |
| ROE, return on equity | `WC08301` | Return on equity |
| ROA, return on assets | `WC08326` | Return on assets |

### 1.5 Company Information

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| name, company name | `NAME` | Security name |
| sector, industry sector | `SECTOR` | Industry sector |
| country, domicile | `GEOG` | Geographic region |
| currency | `CURR` | Trading currency |
| exchange | `EXCH` | Primary exchange |
| ISIN | `ISIN` | ISIN identifier |
| SEDOL | `SEDOL` | SEDOL identifier |
| ticker, symbol | `MNEM` | Mnemonic/ticker |

### 1.6 Economic Indicators

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| GDP, gross domestic product | varies by country | GDP series |
| inflation, CPI | varies by country | Consumer price index |
| unemployment, unemployment rate | varies by country | Unemployment rate |
| interest rate, policy rate | varies by country | Central bank rate |
| PMI, purchasing managers index | varies by country | PMI indicator |

### 1.7 Fixed Income

| Natural Language | Field Code | Description |
|-----------------|------------|-------------|
| yield, bond yield | `RY` | Redemption yield |
| coupon, coupon rate | `COUP` | Coupon rate |
| maturity, maturity date | `MATDT` | Maturity date |
| duration | `DUR` | Modified duration |
| spread, credit spread | `SPREAD` | Yield spread |

---

## 2. Ticker Format Reference

### 2.1 Equity Prefixes

| Market | Prefix | Example | Notes |
|--------|--------|---------|-------|
| UK (default) | none | `BARC`, `VOD` | No prefix needed |
| US | `U:` | `U:AAPL`, `U:MSFT` | NYSE, NASDAQ |
| US (alt) | `@` | `@AAPL`, `@GOOGL` | US equities shorthand |
| Germany | `D:` | `D:SAP`, `D:BMW` | Frankfurt |
| France | `F:` | `F:LVMH`, `F:AIR` | Paris |
| Japan | `J:` | `J:7203`, `J:6758` | Tokyo (numeric codes) |
| Hong Kong | `H:` | `H:0005`, `H:0700` | HKEX |
| Australia | `A:` | `A:BHP`, `A:CBA` | ASX |
| Canada | `C:` | `C:RY`, `C:TD` | TSX |

### 2.2 Index Mnemonics

| Index | Mnemonic | Region |
|-------|----------|--------|
| S&P 500 | `S&PCOMP` | US |
| Dow Jones Industrial | `DJINDUS` | US |
| NASDAQ Composite | `NASCOMP` | US |
| FTSE 100 | `FTSE100` | UK |
| FTSE 250 | `FTSE250` | UK |
| DAX | `DAXINDX` | Germany |
| CAC 40 | `FRCAC40` | France |
| Nikkei 225 | `JAPDOWA` | Japan |
| Hang Seng | `HNGKNGI` | Hong Kong |
| Euro Stoxx 50 | `DJES50I` | Europe |
| MSCI World | `MSWRLD$` | Global |
| MSCI Emerging Markets | `MSEMKF$` | Emerging |

### 2.3 Commodity Futures

| Commodity | Market Code | Example Contract |
|-----------|-------------|------------------|
| Brent Crude Oil | `LLC` | `LLC0125` (Jan 2025) |
| WTI Crude Oil | `CLN` | `CLN0125` |
| Natural Gas | `NGN` | `NGN0125` |
| Gold | `GCN` | `GCN0125` |
| Silver | `SIN` | `SIN0125` |
| Copper | `HGN` | `HGN0125` |
| Corn | `CN` | `CN0125` |
| Wheat | `WN` | `WN0125` |
| Soybeans | `SN` | `SN0125` |

### 2.4 Currency Pairs

| Pair | Mnemonic | Description |
|------|----------|-------------|
| EUR/USD | `EURUSD` | Euro vs Dollar |
| GBP/USD | `GBPUSD` | Sterling vs Dollar |
| USD/JPY | `USDJPY` | Dollar vs Yen |
| USD/CHF | `USDCHF` | Dollar vs Franc |
| AUD/USD | `AUDUSD` | Aussie vs Dollar |
| USD/CAD | `USDCAD` | Dollar vs Loonie |

---

## 3. Date Format Reference

### 3.1 Absolute Dates

| Format | Example | Notes |
|--------|---------|-------|
| ISO format | `2024-01-15` | Recommended |
| US format | `01/15/2024` | Supported |
| UK format | `15/01/2024` | Supported |

### 3.2 Relative Dates

| Natural Language | Code | Description |
|-----------------|------|-------------|
| today | `0D` | Current date |
| yesterday | `-1D` | Previous day |
| last week | `-1W` | 7 days ago |
| last month | `-1M` | 1 month ago |
| last year, year ago | `-1Y` | 1 year ago |
| 5 years ago | `-5Y` | 5 years ago |
| start of year, YTD | `-0Y` | Jan 1 current year |
| beginning of time | `-99Y` | Earliest available |

### 3.3 Frequency Codes

| Natural Language | Code | Description |
|-----------------|------|-------------|
| daily | `D` | Daily data |
| weekly | `W` | Weekly (Friday) |
| monthly | `M` | Monthly (end of month) |
| quarterly | `Q` | Quarterly |
| yearly, annually | `Y` | Annual |

---

## 4. Question-Answer Reference (Test Cases)

### 4.1 Level 1: Simple Single-Ticker Queries

#### Price Queries

**Q1:** "What is Apple's current stock price?"
```python
get_data(tickers='@AAPL', fields=['P'], start='0D', end='0D', kind=0)
```

**Q2:** "Get Microsoft's closing price for today"
```python
get_data(tickers='U:MSFT', fields=['P'], start='0D', end='0D', kind=0)
```

**Q3:** "Show me Barclays share price"
```python
get_data(tickers='BARC', fields=['P'], start='0D', end='0D', kind=0)
```

**Q4:** "What's the latest price of Tesla stock?"
```python
get_data(tickers='U:TSLA', fields=['P'], start='0D', end='0D', kind=0)
```

**Q5:** "Get Amazon's stock price"
```python
get_data(tickers='U:AMZN', fields=['P'], start='0D', end='0D', kind=0)
```

#### Basic Fundamentals

**Q6:** "What is Apple's market cap?"
```python
get_data(tickers='@AAPL', fields=['MV'], start='0D', end='0D', kind=0)
```

**Q7:** "Show me Google's PE ratio"
```python
get_data(tickers='U:GOOGL', fields=['PE'], start='0D', end='0D', kind=0)
```

**Q8:** "What is the dividend yield for Coca-Cola?"
```python
get_data(tickers='U:KO', fields=['DY'], start='0D', end='0D', kind=0)
```

**Q9:** "Get JP Morgan's earnings per share"
```python
get_data(tickers='U:JPM', fields=['EPS'], start='0D', end='0D', kind=0)
```

**Q10:** "What is Shell's enterprise value?"
```python
get_data(tickers='SHEL', fields=['EV'], start='0D', end='0D', kind=0)
```

#### Company Information

**Q11:** "What sector is Netflix in?"
```python
get_data(tickers='U:NFLX', fields=['NAME', 'SECTOR'], start='0D', end='0D', kind=0)
```

**Q12:** "Get the ISIN for Vodafone"
```python
get_data(tickers='VOD', fields=['ISIN'], start='0D', end='0D', kind=0)
```

**Q13:** "What currency does Toyota trade in?"
```python
get_data(tickers='J:7203', fields=['CURR'], start='0D', end='0D', kind=0)
```

**Q14:** "What exchange is HSBC listed on?"
```python
get_data(tickers='HSBA', fields=['EXCH'], start='0D', end='0D', kind=0)
```

**Q15:** "Get the full name for ticker NVDA"
```python
get_data(tickers='U:NVDA', fields=['NAME'], start='0D', end='0D', kind=0)
```

---

### 4.2 Level 2: Time Series & Multiple Fields

#### Historical Price Data

**Q16:** "Get Apple's stock price for all of 2024"
```python
get_data(tickers='@AAPL', fields=['P'], start='2024-01-01', end='2024-12-31', freq='D')
```

**Q17:** "Show me Tesla's daily prices for the last month"
```python
get_data(tickers='U:TSLA', fields=['P'], start='-1M', end='0D', freq='D')
```

**Q18:** "Get Microsoft's weekly closing prices for the past year"
```python
get_data(tickers='U:MSFT', fields=['P'], start='-1Y', end='0D', freq='W')
```

**Q19:** "Show monthly prices for Barclays since 2020"
```python
get_data(tickers='BARC', fields=['P'], start='2020-01-01', end='0D', freq='M')
```

**Q20:** "Get Amazon's annual closing prices for the last 10 years"
```python
get_data(tickers='U:AMZN', fields=['P'], start='-10Y', end='0D', freq='Y')
```

#### OHLCV Data

**Q21:** "Get Apple's OHLC data for January 2024"
```python
get_data(tickers='@AAPL', fields=['PO', 'PH', 'PL', 'P'], start='2024-01-01', end='2024-01-31', freq='D')
```

**Q22:** "Show me daily open, high, low, close, and volume for Tesla last week"
```python
get_data(tickers='U:TSLA', fields=['PO', 'PH', 'PL', 'P', 'VO'], start='-1W', end='0D', freq='D')
```

**Q23:** "Get NVIDIA's daily trading data with volume for the past 3 months"
```python
get_data(tickers='U:NVDA', fields=['PO', 'PH', 'PL', 'P', 'VO'], start='-3M', end='0D', freq='D')
```

**Q24:** "Show weekly OHLC for the S&P 500 index this year"
```python
get_data(tickers='S&PCOMP', fields=['PO', 'PH', 'PL', 'PI'], start='-0Y', end='0D', freq='W')
```

**Q25:** "Get Brent crude oil daily prices and volume for 2024"
```python
get_data(tickers='LCRUDE', fields=['PS', 'VO'], start='2024-01-01', end='2024-12-31', freq='D')
```

#### Multiple Fields

**Q26:** "Get Apple's price and market cap history for 2024"
```python
get_data(tickers='@AAPL', fields=['P', 'MV'], start='2024-01-01', end='2024-12-31', freq='M')
```

**Q27:** "Show me Microsoft's PE ratio and dividend yield over the past 5 years"
```python
get_data(tickers='U:MSFT', fields=['PE', 'DY'], start='-5Y', end='0D', freq='M')
```

**Q28:** "Get quarterly revenue and net income for Amazon since 2020"
```python
get_data(tickers='U:AMZN', fields=['WC01001', 'WC01751'], start='2020-01-01', end='0D', freq='Q')
```

**Q29:** "Show Johnson & Johnson's EPS, dividend, and payout ratio history"
```python
get_data(tickers='U:JNJ', fields=['EPS', 'DPS', 'POUT'], start='-5Y', end='0D', freq='Y')
```

**Q30:** "Get Exxon's ROE and ROA for the last 10 years"
```python
get_data(tickers='U:XOM', fields=['WC08301', 'WC08326'], start='-10Y', end='0D', freq='Y')
```

---

### 4.3 Level 3: Multiple Tickers & Comparisons

#### Multi-Ticker Queries

**Q31:** "Compare Apple, Microsoft, and Google stock prices"
```python
get_data(tickers='@AAPL,U:MSFT,U:GOOGL', fields=['P'], start='0D', end='0D', kind=0)
```

**Q32:** "Get the market caps of the big 5 tech companies"
```python
get_data(tickers='@AAPL,U:MSFT,U:GOOGL,U:AMZN,U:META', fields=['MV'], start='0D', end='0D', kind=0)
```

**Q33:** "Show PE ratios for major US banks"
```python
get_data(tickers='U:JPM,U:BAC,U:WFC,U:C,U:GS', fields=['PE'], start='0D', end='0D', kind=0)
```

**Q34:** "Compare dividend yields of UK oil majors"
```python
get_data(tickers='SHEL,BP.', fields=['DY'], start='0D', end='0D', kind=0)
```

**Q35:** "Get prices for FAANG stocks over the past year"
```python
get_data(tickers='U:META,@AAPL,U:AMZN,U:NFLX,U:GOOGL', fields=['P'], start='-1Y', end='0D', freq='W')
```

#### Cross-Market Comparisons

**Q36:** "Compare S&P 500, FTSE 100, and DAX performance this year"
```python
get_data(tickers='S&PCOMP,FTSE100,DAXINDX', fields=['PI'], start='-0Y', end='0D', freq='D')
```

**Q37:** "Show me EUR/USD, GBP/USD, and USD/JPY exchange rates for 2024"
```python
get_data(tickers='EURUSD,GBPUSD,USDJPY', fields=['P'], start='2024-01-01', end='2024-12-31', freq='D')
```

**Q38:** "Get gold, silver, and copper prices for the past 6 months"
```python
get_data(tickers='GOLDLNPM,SLVRLN,COPSLP', fields=['P'], start='-6M', end='0D', freq='D')
```

**Q39:** "Compare oil prices: Brent vs WTI for 2024"
```python
get_data(tickers='LCRUDE,CRUDOIL', fields=['P'], start='2024-01-01', end='2024-12-31', freq='D')
```

**Q40:** "Show major Asian indices performance: Nikkei, Hang Seng, and Shanghai"
```python
get_data(tickers='JAPDOWA,HNGKNGI,CHSCOMP', fields=['PI'], start='-1Y', end='0D', freq='W')
```

---

### 4.4 Level 4: Index Constituents & Reference Data

#### Index Constituents

**Q41:** "What companies are in the S&P 500?"
```python
get_constituents(index='S&PCOMP', only_list=True)
```

**Q42:** "List all FTSE 100 constituents"
```python
get_constituents(index='FTSE100', only_list=True)
```

**Q43:** "Show me the DAX 40 components"
```python
get_constituents(index='DAXINDX', only_list=True)
```

**Q44:** "Get the constituents of the Dow Jones Industrial Average"
```python
get_constituents(index='DJINDUS', only_list=True)
```

**Q45:** "What stocks are in the NASDAQ 100?"
```python
get_constituents(index='NASCOMP', only_list=True)
```

**Q46:** "List Euro Stoxx 50 members"
```python
get_constituents(index='DJES50I', only_list=True)
```

**Q47:** "Show me the components of the Nikkei 225"
```python
get_constituents(index='JAPDOWA', only_list=True)
```

#### Reference Data Lookups

**Q48:** "Get all exchange listings for Apple stock"
```python
get_all_listings(tickers='@AAPL')
```

**Q49:** "Find the ISIN, SEDOL, and CUSIP for Microsoft"
```python
get_data(tickers='U:MSFT', fields=['ISIN', 'SEDOL', 'CUSIP'], start='0D', end='0D', kind=0)
```

**Q50:** "What type of asset is VFIAX?"
```python
get_asset_types(tickers='U:VFIAX')
```

**Q51:** "Get identifiers and codes for Tesla"
```python
get_codes(tickers='U:TSLA')
```

**Q52:** "Show all available listings for HSBC globally"
```python
get_all_listings(tickers='HSBA')
```

---

### 4.5 Level 5: Futures & Commodities

#### Futures Contracts

**Q53:** "Show me all active Brent crude oil futures contracts"
```python
get_futures_contracts(market_code='LLC', include_dead=False)
```

**Q54:** "Get WTI crude oil futures chain"
```python
get_futures_contracts(market_code='CLN', include_dead=False)
```

**Q55:** "List available gold futures contracts"
```python
get_futures_contracts(market_code='GCN', include_dead=False)
```

**Q56:** "Show natural gas futures contracts including expired"
```python
get_futures_contracts(market_code='NGN', include_dead=True)
```

**Q57:** "Get corn futures contracts for 2025"
```python
get_futures_contracts(market_code='CN', include_dead=False)
```

#### Futures Pricing

**Q58:** "Get the front month Brent crude price"
```python
get_data(tickers='LLC0125', fields=['PS'], start='0D', end='0D', kind=0)
```

**Q59:** "Show gold futures prices for the next 6 contract months"
```python
get_data(tickers='GCN0125,GCN0225,GCN0425,GCN0625,GCN0825,GCN1025', fields=['PS'], start='0D', end='0D', kind=0)
```

**Q60:** "Get the WTI crude oil futures curve"
```python
get_data(tickers='CLN0125,CLN0225,CLN0325,CLN0425,CLN0525,CLN0625', fields=['PS'], start='0D', end='0D', kind=0)
```

---

### 4.6 Level 6: Economic Data

#### GDP & Growth

**Q61:** "Get US GDP growth for the last 10 years"
```python
get_data(tickers='USGDP...D', fields=['P'], start='-10Y', end='0D', freq='Q')
```

**Q62:** "Show UK quarterly GDP since 2015"
```python
get_data(tickers='UKGDP...D', fields=['P'], start='2015-01-01', end='0D', freq='Q')
```

**Q63:** "Compare GDP growth: US, UK, Germany, Japan"
```python
get_data(tickers='USGDP...D,UKGDP...D,WGGDP...D,JPGDP...D', fields=['P'], start='-5Y', end='0D', freq='Q')
```

**Q64:** "Get China's annual GDP for the last 20 years"
```python
get_data(tickers='CHGDP...D', fields=['P'], start='-20Y', end='0D', freq='Y')
```

#### Inflation & CPI

**Q65:** "What is the current US inflation rate?"
```python
get_data(tickers='USCONPRCF', fields=['P'], start='-1M', end='0D', freq='M')
```

**Q66:** "Show UK CPI inflation history for the past 5 years"
```python
get_data(tickers='UKCONPRCF', fields=['P'], start='-5Y', end='0D', freq='M')
```

**Q67:** "Compare inflation rates across G7 countries"
```python
get_data(tickers='USCONPRCF,UKCONPRCF,WGCONPRCF,JPCONPRCF,CNCONPRCF,ITCONPRCF,FRCONPRCF', fields=['P'], start='-2Y', end='0D', freq='M')
```

**Q68:** "Get Eurozone HICP inflation data"
```python
get_data(tickers='EMHICPF', fields=['P'], start='-3Y', end='0D', freq='M')
```

#### Employment & Labor

**Q69:** "What is the US unemployment rate?"
```python
get_data(tickers='USUN%TOTQ', fields=['P'], start='0D', end='0D', kind=0)
```

**Q70:** "Show US nonfarm payrolls for the past 2 years"
```python
get_data(tickers='USNFPAYR', fields=['P'], start='-2Y', end='0D', freq='M')
```

**Q71:** "Compare unemployment rates: US, UK, Germany"
```python
get_data(tickers='USUN%TOTQ,UKUN%TOTQ,WGUN%TOTQ', fields=['P'], start='-5Y', end='0D', freq='M')
```

#### Interest Rates & Central Banks

**Q72:** "What is the current Fed funds rate?"
```python
get_data(tickers='USFEDFUN', fields=['P'], start='0D', end='0D', kind=0)
```

**Q73:** "Show Bank of England base rate history"
```python
get_data(tickers='UKPRATE', fields=['P'], start='-10Y', end='0D', freq='M')
```

**Q74:** "Get ECB main refinancing rate since 2010"
```python
get_data(tickers='ECBMRR', fields=['P'], start='2010-01-01', end='0D', freq='M')
```

**Q75:** "Compare central bank policy rates: Fed, ECB, BoE, BoJ"
```python
get_data(tickers='USFEDFUN,ECBMRR,UKPRATE,JPUNCALL', fields=['P'], start='-5Y', end='0D', freq='M')
```

#### PMI & Sentiment

**Q76:** "Get the latest US manufacturing PMI"
```python
get_data(tickers='USPMIMAN', fields=['P'], start='-1M', end='0D', freq='M')
```

**Q77:** "Show US services PMI history for 2024"
```python
get_data(tickers='USPMISER', fields=['P'], start='2024-01-01', end='2024-12-31', freq='M')
```

**Q78:** "Compare manufacturing PMI: US, China, Germany"
```python
get_data(tickers='USPMIMAN,CHPMIMAN,WGPMIMAN', fields=['P'], start='-2Y', end='0D', freq='M')
```

**Q79:** "Get US consumer confidence index"
```python
get_data(tickers='USCNFCON', fields=['P'], start='-3Y', end='0D', freq='M')
```

**Q80:** "Show University of Michigan consumer sentiment"
```python
get_data(tickers='USUMCSENT', fields=['P'], start='-5Y', end='0D', freq='M')
```

---

### 4.7 Level 7: Fixed Income

#### Government Bonds

**Q81:** "What is the current US 10-year Treasury yield?"
```python
get_data(tickers='USBD10Y', fields=['RY'], start='0D', end='0D', kind=0)
```

**Q82:** "Get UK 10-year gilt yield history"
```python
get_data(tickers='UKBD10Y', fields=['RY'], start='-5Y', end='0D', freq='D')
```

**Q83:** "Show the US Treasury yield curve (2Y, 5Y, 10Y, 30Y)"
```python
get_data(tickers='USBD2Y,USBD5Y,USBD10Y,USBD30Y', fields=['RY'], start='0D', end='0D', kind=0)
```

**Q84:** "Compare 10-year yields: US, UK, Germany, Japan"
```python
get_data(tickers='USBD10Y,UKBD10Y,BDBD10Y,JPBD10Y', fields=['RY'], start='-2Y', end='0D', freq='D')
```

**Q85:** "Get German bund yields across maturities"
```python
get_data(tickers='BDBD2Y,BDBD5Y,BDBD10Y,BDBD30Y', fields=['RY'], start='0D', end='0D', kind=0)
```

#### Credit & Spreads

**Q86:** "Show US investment grade corporate bond spreads"
```python
get_data(tickers='MLCORPM', fields=['P'], start='-3Y', end='0D', freq='D')
```

**Q87:** "Get US high yield bond spreads history"
```python
get_data(tickers='MLHYOAS', fields=['P'], start='-5Y', end='0D', freq='D')
```

**Q88:** "Compare investment grade vs high yield spreads"
```python
get_data(tickers='MLCORPM,MLHYOAS', fields=['P'], start='-3Y', end='0D', freq='D')
```

---

### 4.8 Level 8: Calculated Fields & Functions

#### Moving Averages

**Q89:** "Get Apple's 20-day moving average price"
```python
get_data(tickers='MAV#(@AAPL,20D)', fields=['P'], start='0D', end='0D', kind=0)
```

**Q90:** "Show Tesla's 50-day and 200-day moving averages"
```python
get_data(tickers='MAV#(U:TSLA,50D),MAV#(U:TSLA,200D)', fields=['P'], start='-1Y', end='0D', freq='D')
```

**Q91:** "Get the S&P 500's 200-day moving average"
```python
get_data(tickers='MAV#(S&PCOMP,200D)', fields=['PI'], start='-1Y', end='0D', freq='D')
```

#### Percentage Changes

**Q92:** "What is Apple's year-to-date return?"
```python
get_data(tickers='PCH#(@AAPL,-0Y)', fields=['P'], start='0D', end='0D', kind=0)
```

**Q93:** "Show Microsoft's monthly returns for 2024"
```python
get_data(tickers='PCH#(U:MSFT,1M)', fields=['P'], start='2024-01-01', end='2024-12-31', freq='M')
```

**Q94:** "Get 1-year percentage change for FAANG stocks"
```python
get_data(tickers='PCH#(U:META,1Y),PCH#(@AAPL,1Y),PCH#(U:AMZN,1Y),PCH#(U:NFLX,1Y),PCH#(U:GOOGL,1Y)', fields=['P'], start='0D', end='0D', kind=0)
```

**Q95:** "Show S&P 500's daily returns for the past month"
```python
get_data(tickers='PCH#(S&PCOMP,1D)', fields=['PI'], start='-1M', end='0D', freq='D')
```

#### Combined Functions

**Q96:** "Get the 20-day moving average of Tesla's daily returns"
```python
get_data(tickers='MAV#(PCH#(U:TSLA,1D),20D)', fields=['P'], start='-3M', end='0D', freq='D')
```

**Q97:** "Show percentage change in Apple's 50-day moving average"
```python
get_data(tickers='PCH#(MAV#(@AAPL,50D),1M)', fields=['P'], start='-6M', end='0D', freq='D')
```

---

### 4.9 Level 9: Economic Data Revisions (EPiT)

**Q98:** "Get the revision history for US Q3 2024 GDP"
```python
get_epit_revisions(series='USGDP...D', period='2024-09')
```

**Q99:** "Show how UK inflation estimates for June 2024 were revised"
```python
get_epit_revisions(series='UKCONPRCF', period='2024-06')
```

**Q100:** "Get the vintage matrix for US nonfarm payrolls revisions"
```python
get_epit_vintage_matrix(series='USNFPAYR', date_from='2024-01-01', date_to='2024-12-31')
```

---

### 4.10 Level 10: Complex Multi-Step Queries

These queries require multiple API calls or complex reasoning:

**Q101:** "Get the prices of all S&P 500 constituents"
```python
# Step 1: Get constituents
constituents = get_constituents(index='S&PCOMP', only_list=True)
# Step 2: Get prices for all (batched)
get_data(tickers=','.join(constituents), fields=['P'], start='0D', end='0D', kind=0)
```

**Q102:** "Calculate the average PE ratio of FTSE 100 companies"
```python
# Step 1: Get constituents
constituents = get_constituents(index='FTSE100', only_list=True)
# Step 2: Get PE ratios
get_data(tickers=','.join(constituents), fields=['PE'], start='0D', end='0D', kind=0)
# Step 3: Calculate average (client-side)
```

**Q103:** "Find the best performing stock in the DAX over the past year"
```python
# Step 1: Get constituents
constituents = get_constituents(index='DAXINDX', only_list=True)
# Step 2: Get 1-year returns
get_data(tickers=','.join(['PCH#(' + t + ',1Y)' for t in constituents]), fields=['P'], start='0D', end='0D', kind=0)
# Step 3: Find max (client-side)
```

**Q104:** "Compare sector performance within the S&P 500"
```python
# Requires: sector indices or filtering by sector
get_data(tickers='S5COND,S5CONS,S5ENRS,S5FINL,S5HLTH,S5INDU,S5INFT,S5MATR,S5RLST,S5TELS,S5UTIL', fields=['PI'], start='-1Y', end='0D', freq='W')
```

**Q105:** "Get the top 10 highest dividend yielding stocks in the FTSE 100"
```python
# Step 1: Get constituents
constituents = get_constituents(index='FTSE100', only_list=True)
# Step 2: Get dividend yields
get_data(tickers=','.join(constituents), fields=['DY'], start='0D', end='0D', kind=0)
# Step 3: Sort and take top 10 (client-side)
```

---

## 5. Error Handling Reference

### 5.1 Common Error Scenarios

| Scenario | Expected Behavior |
|----------|-------------------|
| Invalid ticker | Return error: "Instrument not found" |
| Invalid field code | Return error: "Data type not recognized" |
| No data for date range | Return empty dataset with warning |
| Authentication failure | Return error: "Invalid credentials" |
| Rate limit exceeded | Return error: "Request limit exceeded" |
| Permission denied | Return error: "Not authorized for this data" |

### 5.2 Edge Cases for Test Cases

| Test Case | Query | Expected Handling |
|-----------|-------|-------------------|
| Delisted stock | "Get Enron's stock price" | Historical data only, note delisting |
| Future date | "Get Apple's price for 2030" | Error: Date in future |
| Merged company | "Get Time Warner stock price" | Redirect to successor or note merger |
| Weekend date | "Get price for Saturday" | Return Friday's close or nearest trading day |
| Holiday | "Get price for Christmas Day" | Return previous trading day |

---

## 6. Query Complexity Classification

### 6.1 Complexity Scoring Rubric

| Factor | Score |
|--------|-------|
| Single ticker | +0 |
| Multiple tickers (2-5) | +1 |
| Multiple tickers (5+) | +2 |
| Single field | +0 |
| Multiple fields (2-3) | +1 |
| Multiple fields (4+) | +2 |
| Static data (kind=0) | +0 |
| Time series | +1 |
| Long history (5+ years) | +1 |
| Calculated fields (MAV#, PCH#) | +2 |
| Index constituents lookup | +2 |
| Multi-step query | +3 |
| Economic data with revisions | +2 |

### 6.2 Mapping to Complexity Levels

| Total Score | Complexity Level |
|-------------|-----------------|
| 0-1 | Level 1 (Trivial) |
| 2-3 | Level 2 (Simple) |
| 4-5 | Level 3 (Moderate) |
| 6-7 | Level 4 (Complex) |
| 8+ | Level 5 (Advanced) |

---

## 7. Natural Language Variations

### 7.1 Synonyms for Common Terms

| Concept | Variations |
|---------|------------|
| Stock price | share price, equity price, stock quote, current price, trading price |
| Market cap | market capitalization, market value, company value, valuation |
| PE ratio | price-to-earnings, P/E, earnings multiple, PE multiple |
| Dividend yield | yield, dividend %, income yield |
| Get/Show | fetch, retrieve, pull, display, what is, tell me, give me |
| Historical | past, previous, since, from, over the last, for the period |
| Compare | vs, versus, against, relative to, comparison of |
| Current | latest, most recent, today's, now, as of today |

### 7.2 Date Expression Variations

| Expression | Interpretation |
|------------|----------------|
| "last year" | -1Y to 0D |
| "past 12 months" | -1Y to 0D |
| "since January" | -0Y to 0D (YTD) |
| "year to date", "YTD" | -0Y to 0D |
| "over the past 5 years" | -5Y to 0D |
| "from 2020 to 2024" | 2020-01-01 to 2024-12-31 |
| "Q3 2024" | 2024-07-01 to 2024-09-30 |
| "this month" | Start of month to 0D |
| "last week" | -1W to 0D |
| "yesterday" | -1D |

---

## 8. Screening Queries (Find Securities by Market Data Criteria)

### 8.1 Index Constituent Lists (Datastream)

Datastream provides constituent lists with `L` prefix:

```python
# Get current constituents
get_data(tickers='LS&PCOMP|L', fields=['MNEM', 'NAME'], kind=0)

# Get historical constituents (MMYY format)
get_data(tickers='LS&PCOMP0120|L', fields=['MNEM', 'NAME'], kind=0)  # Jan 2020
```

### 8.2 Level 1: Index Constituent Queries

**Q106:** "What companies are in the S&P 500?"
```python
get_data(tickers='LS&PCOMP|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q107:** "List all FTSE 100 constituents"
```python
get_data(tickers='LFTSE100|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q108:** "Get the DAX 40 companies"
```python
get_data(tickers='LDAXINDX|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q109:** "What stocks are in the Nikkei 225?"
```python
get_data(tickers='LSTOKYOSE|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q110:** "Show NASDAQ 100 constituents"
```python
get_data(tickers='LNASDAQ|L', fields=['MNEM', 'NAME'], kind=0)
```

### 8.3 Level 2: Index Ranking Queries

**Q111:** "What are the top 10 S&P 500 stocks by market cap?"
```python
# Step 1: Get constituents
constituents = get_data(tickers='LS&PCOMP|L', fields=['MNEM'], kind=0)
# Step 2: Get market caps
df = get_data(tickers=','.join(constituents), fields=['NAME', 'MV'], kind=0)
# Step 3: Sort
df.sort_values('MV', ascending=False).head(10)
```

**Q112:** "Which FTSE 100 stocks have the highest dividend yield?"
```python
constituents = get_data(tickers='LFTSE100|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'DY'], kind=0)
df.sort_values('DY', ascending=False).head(10)
```

**Q113:** "Find the best performing DAX stocks this year"
```python
constituents = get_data(tickers='LDAXINDX|L', fields=['MNEM'], kind=0)
# Get YTD returns using PCH# function
df = get_data(tickers=','.join(['PCH#(' + t + ',-0Y)' for t in constituents]), fields=['P'], kind=0)
df.sort_values('P', ascending=False).head(10)
```

**Q114:** "Get the lowest PE stocks in the S&P 500"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'PE'], kind=0)
df[df['PE'] > 0].sort_values('PE', ascending=True).head(10)
```

### 8.4 Level 3: Historical Index Composition

**Q115:** "What companies were in the S&P 500 in January 2020?"
```python
get_data(tickers='LS&PCOMP0120|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q116:** "Show FTSE 100 constituents from December 2019"
```python
get_data(tickers='LFTSE1001219|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q117:** "Get DAX composition at end of 2018"
```python
get_data(tickers='LDAXINDX1218|L', fields=['MNEM', 'NAME'], kind=0)
```

**Q118:** "What stocks were added/removed from S&P 500 since 2020?"
```python
# Step 1: Get 2020 constituents
spx_2020 = set(get_data(tickers='LS&PCOMP0120|L', fields=['MNEM'], kind=0)['MNEM'])
# Step 2: Get current constituents
spx_now = set(get_data(tickers='LS&PCOMP|L', fields=['MNEM'], kind=0)['MNEM'])
# Step 3: Calculate changes
added = spx_now - spx_2020
removed = spx_2020 - spx_now
```

### 8.5 Level 4: Cross-Index Comparisons

**Q119:** "Compare top stocks across major indices by market cap"
```python
indices = {
    'S&P 500': 'LS&PCOMP|L',
    'FTSE 100': 'LFTSE100|L',
    'DAX': 'LDAXINDX|L'
}
results = {}
for name, code in indices.items():
    constituents = get_data(tickers=code, fields=['MNEM'], kind=0)
    df = get_data(tickers=','.join(constituents), fields=['NAME', 'MV'], kind=0)
    results[name] = df.sort_values('MV', ascending=False).head(5)
```

**Q120:** "Find stocks that are in both S&P 500 and NASDAQ 100"
```python
spx = set(get_data(tickers='LS&PCOMP|L', fields=['MNEM'], kind=0)['MNEM'])
ndx = set(get_data(tickers='LNASDAQ|L', fields=['MNEM'], kind=0)['MNEM'])
overlap = spx & ndx
```

### 8.6 Level 5: Sector Within Index

**Q121:** "Get all tech stocks in the S&P 500"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'SECTOR', 'MV'], kind=0)
tech = df[df['SECTOR'].str.contains('Technology', na=False)]
tech.sort_values('MV', ascending=False)
```

**Q122:** "Find the largest banks in the FTSE 100"
```python
constituents = get_data(tickers='LFTSE100|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'SECTOR', 'MV'], kind=0)
banks = df[df['SECTOR'].str.contains('Bank', na=False)]
banks.sort_values('MV', ascending=False)
```

**Q123:** "Show energy stocks in the DAX"
```python
constituents = get_data(tickers='LDAXINDX|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'SECTOR', 'MV'], kind=0)
energy = df[df['SECTOR'].str.contains('Energy|Oil|Gas', na=False)]
```

### 8.7 Level 6: Performance Screening Within Index

**Q124:** "Find S&P 500 stocks at 52-week highs"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'P', 'P52WH'], kind=0)
at_highs = df[df['P'] >= df['P52WH'] * 0.98]  # Within 2% of high
```

**Q125:** "Get FTSE 100 stocks down more than 20% from highs"
```python
constituents = get_data(tickers='LFTSE100|L', fields=['MNEM'], kind=0)
df = get_data(tickers=','.join(constituents), fields=['NAME', 'P', 'P52WH'], kind=0)
df['Drawdown'] = (df['P'] - df['P52WH']) / df['P52WH'] * 100
beaten_down = df[df['Drawdown'] <= -20]
```

---

## Appendix A: Full Field Code Catalog

For complete field code reference, see LSEG Datastream Navigator or contact LSEG support.

Common field code patterns:
- `WC*****` - Worldscope fundamentals
- `IBES*` - I/B/E/S estimates
- `TR*` - Thomson Reuters proprietary
- `DS*` - Datastream calculated

## Appendix B: Regional Ticker Prefixes

| Prefix | Country/Region |
|--------|---------------|
| (none) | UK |
| `U:` | United States |
| `@` | US (shorthand) |
| `D:` | Germany |
| `F:` | France |
| `I:` | Italy |
| `E:` | Spain |
| `N:` | Netherlands |
| `S:` | Switzerland |
| `J:` | Japan |
| `H:` | Hong Kong |
| `A:` | Australia |
| `C:` | Canada |
| `K:` | South Korea |
| `T:` | Taiwan |
| `IN:` | India |
| `B:` | Brazil |
| `M:` | Mexico |
