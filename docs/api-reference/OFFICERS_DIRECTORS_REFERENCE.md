# LSEG Officers & Directors API Reference for Eval Platform

> **Purpose:** Maps natural language queries to expected Officers & Directors API tool calls.
> **Target API:** LSEG Officers & Directors Database (via Refinitiv Data Library)
> **Version:** 1.0.0

---

## 1. Database Overview

### 1.1 Coverage

| Metric | Value |
|--------|-------|
| Total Officers/Directors | 3.6+ million |
| Unique Individuals | 2.9+ million |
| Public Companies | 93,000+ |
| Private Companies | 300,000+ |
| Countries | 120+ markets |
| US History | Since 1998 |
| International History | Since 2003 |

### 1.2 Data Categories

- Executive profiles and biographies
- Employment history (24+ years US, 15+ years international)
- Education history
- Compensation data (salary, bonus, stock awards)
- Committee memberships
- Board positions
- Unique person identifiers for career tracking

---

## 2. TR Field Code Reference

### 2.1 Officer Identification

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| officer name, executive name | `TR.OfficerName` | Full name of officer |
| officer full name | `TR.ODOfficerFullName` | Complete name |
| officer title, position, role | `TR.OfficerTitle` | Current title/position |
| position description | `TR.ODOfficerPositionDesc` | Detailed position description |
| officer rank | `TR.ODOfficerRank` | Ranking within company |
| officer age | `TR.OfficerAge` | Current age |
| person age | `TR.ODOfficerPersonAge` | Age of the person |

### 2.2 Employment History

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| title since, start date, tenure | `TR.OfficerTitleSince` | When started current role |
| position start date | `TR.ODOfficerPositionStartDate` | Position commencement date |
| years in position | Calculated | Current year - start date |
| employment history | `TR.ODOfficerEmploymentHistory` | Prior positions |

### 2.3 Biography & Background

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| biography, bio | `TR.ODOfficerBiography` | Full biographical text |
| education, university | `TR.ODOfficerUniversityName` | University attended |
| degree | `TR.ODOfficerGraduationDegree` | Academic degree |
| graduation year | `TR.ODOfficerGraduationYear` | Year of graduation |

### 2.4 Compensation (Executive Pay)

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| salary, base salary | `TR.ODOfficerSalary` | Base salary |
| bonus | `TR.ODOfficerBonus` | Annual bonus |
| stock awards | `TR.ODOfficerStockAwards` | Stock-based compensation |
| option awards | `TR.ODOfficerOptionAwards` | Option grants value |
| total compensation | `TR.ODOfficerTotalComp` | Total annual compensation |
| other compensation | `TR.ODOfficerOtherComp` | Other comp elements |
| compensation year | `TR.ODOfficerCompYear` | Year of compensation data |

### 2.5 Board & Committee Membership

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| board member, director | `TR.ODDirectorName` | Board director name |
| committee membership | `TR.ODCommitteeMembership` | Committee assignments |
| audit committee | `TR.ODAuditCommittee` | Audit committee member |
| compensation committee | `TR.ODCompCommittee` | Comp committee member |
| nominating committee | `TR.ODNominatingCommittee` | Nominating committee |
| independent director | `TR.ODIndependentDirector` | Independence status |
| board tenure | `TR.ODDirectorTenure` | Years on board |

### 2.6 Company & Role Context

| Natural Language | TR Code | Description |
|-----------------|---------|-------------|
| company name | `TR.CommonName` | Company name |
| ticker | `TR.TickerSymbol` | Stock ticker |
| CEO | `TR.CEOName` | Chief Executive Officer |
| CFO | `TR.CFOName` | Chief Financial Officer |
| chairman | `TR.ChairmanName` | Board Chairman |

---

## 3. Query Parameters

### 3.1 Ranking Parameter

Use `RNK` to retrieve multiple officers:

```python
# Get top 10 officers
parameters = {'RNK': 'R1:R10'}

# Get top 30 officers
parameters = {'RNK': 'R1:R30'}

# Get all officers (up to 100)
parameters = {'RNK': 'R1:R100'}
```

### 3.2 Officer Type Filter

```python
# Filter by officer type
parameters = {'ODRnk': 'R1:R10', 'ODType': 'Executive'}
parameters = {'ODRnk': 'R1:R10', 'ODType': 'Director'}
```

### 3.3 Important Notes

- **Static Fields:** Officer data fields are static, not time series
- **No Historical API Access:** Cannot retrieve historical officer data via API
- **Current Snapshot Only:** Returns current officers/directors only

---

## 4. Question-Answer Reference (Test Cases)

### 4.1 Level 1: Basic Officer Queries

#### CEO & Top Executive Queries

**Q1:** "Who is the CEO of Apple?"
```python
get_data(['AAPL.O'], ['TR.CEOName'])
```

**Q2:** "Get Apple's CEO and CFO"
```python
get_data(['AAPL.O'], ['TR.CEOName', 'TR.CFOName'])
```

**Q3:** "Who runs Microsoft?"
```python
get_data(['MSFT.O'], ['TR.CEOName', 'TR.OfficerTitle'], {'RNK': 'R1'})
```

**Q4:** "Who is the CFO of Amazon?"
```python
get_data(['AMZN.O'], ['TR.CFOName'])
```

**Q5:** "Get Tesla's chief executive"
```python
get_data(['TSLA.O'], ['TR.CEOName'])
```

#### Chairman & Board

**Q6:** "Who is the chairman of JP Morgan?"
```python
get_data(['JPM.N'], ['TR.ChairmanName'])
```

**Q7:** "Who chairs Apple's board?"
```python
get_data(['AAPL.O'], ['TR.ChairmanName'])
```

**Q8:** "Get Goldman Sachs board chairman"
```python
get_data(['GS.N'], ['TR.ChairmanName'])
```

---

### 4.2 Level 2: Officer Lists

#### Top Executives

**Q9:** "List Apple's top 5 executives"
```python
get_data(['AAPL.O'], ['TR.OfficerName', 'TR.OfficerTitle'], {'RNK': 'R1:R5'})
```

**Q10:** "Get Microsoft's executive team"
```python
get_data(['MSFT.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.OfficerAge'], {'RNK': 'R1:R10'})
```

**Q11:** "Show Amazon's top 10 officers"
```python
get_data(['AMZN.O'], ['TR.ODOfficerFullName', 'TR.ODOfficerPositionDesc'], {'ODRnk': 'R1:R10'})
```

**Q12:** "List Google's management team"
```python
get_data(['GOOGL.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.OfficerTitleSince'], {'RNK': 'R1:R15'})
```

**Q13:** "Get Tesla's C-suite executives"
```python
get_data(['TSLA.O'], ['TR.OfficerName', 'TR.OfficerTitle'], {'RNK': 'R1:R10'})
```

#### Board of Directors

**Q14:** "List Apple's board of directors"
```python
get_data(['AAPL.O'], ['TR.ODDirectorName', 'TR.ODDirectorTenure'], {'ODRnk': 'R1:R15'})
```

**Q15:** "Get Microsoft's board members"
```python
get_data(['MSFT.O'], ['TR.ODDirectorName', 'TR.ODIndependentDirector'], {'ODRnk': 'R1:R15'})
```

**Q16:** "Show JP Morgan's directors"
```python
get_data(['JPM.N'], ['TR.ODDirectorName', 'TR.ODCommitteeMembership'], {'ODRnk': 'R1:R15'})
```

---

### 4.3 Level 3: Officer Details & Tenure

#### Tenure & History

**Q17:** "How long has Tim Cook been Apple's CEO?"
```python
get_data(['AAPL.O'], ['TR.CEOName', 'TR.OfficerTitleSince'], {'RNK': 'R1'})
```

**Q18:** "When did Satya Nadella become Microsoft CEO?"
```python
get_data(['MSFT.O'], ['TR.CEOName', 'TR.OfficerTitleSince'])
```

**Q19:** "Get officer tenure for Amazon's leadership"
```python
get_data(['AMZN.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.OfficerTitleSince'], {'RNK': 'R1:R10'})
```

**Q20:** "Show how long Google's executives have been in their roles"
```python
get_data(['GOOGL.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.OfficerTitleSince'], {'RNK': 'R1:R10'})
```

#### Age & Demographics

**Q21:** "What is the average age of Apple's executives?"
```python
get_data(['AAPL.O'], ['TR.OfficerName', 'TR.OfficerAge'], {'RNK': 'R1:R10'})
# Calculate average client-side
```

**Q22:** "Get ages of Tesla's top officers"
```python
get_data(['TSLA.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.OfficerAge'], {'RNK': 'R1:R10'})
```

**Q23:** "Show Microsoft board member ages"
```python
get_data(['MSFT.O'], ['TR.ODDirectorName', 'TR.ODOfficerPersonAge'], {'ODRnk': 'R1:R15'})
```

---

### 4.4 Level 4: Biography & Education

#### Education Background

**Q24:** "Where did Apple's CEO go to school?"
```python
get_data(['AAPL.O'], ['TR.CEOName', 'TR.ODOfficerUniversityName', 'TR.ODOfficerGraduationDegree'], {'RNK': 'R1'})
```

**Q25:** "Get education details for Microsoft executives"
```python
get_data(['MSFT.O'], ['TR.OfficerName', 'TR.ODOfficerUniversityName', 'TR.ODOfficerGraduationDegree'], {'RNK': 'R1:R10'})
```

**Q26:** "Show university backgrounds of Amazon's leadership"
```python
get_data(['AMZN.O'], ['TR.OfficerName', 'TR.ODOfficerUniversityName', 'TR.ODOfficerGraduationYear'], {'RNK': 'R1:R10'})
```

#### Biographies

**Q27:** "Get Tim Cook's biography"
```python
get_data(['AAPL.O'], ['TR.CEOName', 'TR.ODOfficerBiography'], {'RNK': 'R1'})
```

**Q28:** "Show biographies for Tesla's top 3 executives"
```python
get_data(['TSLA.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.ODOfficerBiography'], {'RNK': 'R1:R3'})
```

**Q29:** "Get executive bios for JP Morgan"
```python
get_data(['JPM.N'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.ODOfficerBiography'], {'RNK': 'R1:R5'})
```

---

### 4.5 Level 5: Compensation Data

#### CEO Compensation

**Q30:** "What is Tim Cook's total compensation?"
```python
get_data(['AAPL.O'], ['TR.CEOName', 'TR.ODOfficerTotalComp'], {'RNK': 'R1'})
```

**Q31:** "Get Apple CEO's salary breakdown"
```python
get_data(['AAPL.O'], ['TR.CEOName', 'TR.ODOfficerSalary', 'TR.ODOfficerBonus', 'TR.ODOfficerStockAwards', 'TR.ODOfficerTotalComp'], {'RNK': 'R1'})
```

**Q32:** "Show Microsoft CEO compensation"
```python
get_data(['MSFT.O'], ['TR.CEOName', 'TR.ODOfficerSalary', 'TR.ODOfficerBonus', 'TR.ODOfficerStockAwards', 'TR.ODOfficerTotalComp'])
```

**Q33:** "What does Tesla's CEO earn?"
```python
get_data(['TSLA.O'], ['TR.CEOName', 'TR.ODOfficerTotalComp'])
```

#### Executive Team Compensation

**Q34:** "Get compensation for Apple's top 5 executives"
```python
get_data(['AAPL.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.ODOfficerSalary', 'TR.ODOfficerBonus', 'TR.ODOfficerStockAwards', 'TR.ODOfficerTotalComp'], {'RNK': 'R1:R5'})
```

**Q35:** "Show salary and bonus for Amazon executives"
```python
get_data(['AMZN.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.ODOfficerSalary', 'TR.ODOfficerBonus'], {'RNK': 'R1:R10'})
```

**Q36:** "Get stock awards for Google's management"
```python
get_data(['GOOGL.O'], ['TR.OfficerName', 'TR.OfficerTitle', 'TR.ODOfficerStockAwards', 'TR.ODOfficerOptionAwards'], {'RNK': 'R1:R10'})
```

**Q37:** "Compare CEO compensation at big tech companies"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O'], ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
```

---

### 4.6 Level 6: Committee Memberships

#### Audit Committee

**Q38:** "Who is on Apple's audit committee?"
```python
get_data(['AAPL.O'], ['TR.ODDirectorName', 'TR.ODAuditCommittee'], {'ODRnk': 'R1:R15'})
```

**Q39:** "Get audit committee members for JP Morgan"
```python
get_data(['JPM.N'], ['TR.ODDirectorName', 'TR.ODAuditCommittee'], {'ODRnk': 'R1:R15'})
```

#### Compensation Committee

**Q40:** "Who sets executive pay at Microsoft?"
```python
get_data(['MSFT.O'], ['TR.ODDirectorName', 'TR.ODCompCommittee'], {'ODRnk': 'R1:R15'})
```

**Q41:** "Get compensation committee for Goldman Sachs"
```python
get_data(['GS.N'], ['TR.ODDirectorName', 'TR.ODCompCommittee'], {'ODRnk': 'R1:R15'})
```

#### All Committee Assignments

**Q42:** "Show all committee memberships for Amazon directors"
```python
get_data(['AMZN.O'], ['TR.ODDirectorName', 'TR.ODCommitteeMembership', 'TR.ODAuditCommittee', 'TR.ODCompCommittee', 'TR.ODNominatingCommittee'], {'ODRnk': 'R1:R15'})
```

---

### 4.7 Level 7: Board Independence & Governance

**Q43:** "Which Apple directors are independent?"
```python
get_data(['AAPL.O'], ['TR.ODDirectorName', 'TR.ODIndependentDirector'], {'ODRnk': 'R1:R15'})
```

**Q44:** "Show independent vs insider directors at Tesla"
```python
get_data(['TSLA.O'], ['TR.ODDirectorName', 'TR.ODIndependentDirector', 'TR.ODDirectorTenure'], {'ODRnk': 'R1:R15'})
```

**Q45:** "Get board independence ratio for big banks"
```python
get_data(['JPM.N', 'BAC.N', 'WFC.N', 'C.N', 'GS.N'], ['TR.CommonName', 'TR.ODDirectorName', 'TR.ODIndependentDirector'], {'ODRnk': 'R1:R15'})
# Calculate ratio client-side
```

---

### 4.8 Level 8: Multi-Company Comparisons

#### CEO Comparison

**Q46:** "Compare CEOs of FAANG companies"
```python
get_data(['META.O', 'AAPL.O', 'AMZN.O', 'NFLX.O', 'GOOGL.O'], ['TR.CommonName', 'TR.CEOName', 'TR.OfficerTitleSince', 'TR.OfficerAge'])
```

**Q47:** "Get CEO tenure for major tech companies"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O', 'NVDA.O', 'TSLA.O'], ['TR.CommonName', 'TR.CEOName', 'TR.OfficerTitleSince'])
```

**Q48:** "Compare CEO compensation across tech sector"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O'], ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
```

#### Board Comparison

**Q49:** "Compare board sizes of major banks"
```python
get_data(['JPM.N', 'BAC.N', 'WFC.N', 'C.N', 'GS.N'], ['TR.CommonName', 'TR.ODDirectorName'], {'ODRnk': 'R1:R20'})
# Count directors per company client-side
```

**Q50:** "Get average director tenure at tech companies"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O'], ['TR.CommonName', 'TR.ODDirectorName', 'TR.ODDirectorTenure'], {'ODRnk': 'R1:R15'})
# Calculate average client-side
```

---

### 4.9 Level 9: Executive Package Queries

**Q51:** "Get complete profile for Apple's CEO"
```python
get_data(['AAPL.O'], [
    'TR.CEOName',
    'TR.OfficerTitle',
    'TR.OfficerTitleSince',
    'TR.OfficerAge',
    'TR.ODOfficerBiography',
    'TR.ODOfficerUniversityName',
    'TR.ODOfficerGraduationDegree',
    'TR.ODOfficerSalary',
    'TR.ODOfficerBonus',
    'TR.ODOfficerStockAwards',
    'TR.ODOfficerTotalComp'
], {'RNK': 'R1'})
```

**Q52:** "Full executive summary for Microsoft's leadership"
```python
get_data(['MSFT.O'], [
    'TR.OfficerName',
    'TR.OfficerTitle',
    'TR.OfficerTitleSince',
    'TR.OfficerAge',
    'TR.ODOfficerUniversityName',
    'TR.ODOfficerSalary',
    'TR.ODOfficerBonus',
    'TR.ODOfficerTotalComp'
], {'RNK': 'R1:R10'})
```

**Q53:** "Complete board profile for Amazon"
```python
get_data(['AMZN.O'], [
    'TR.ODDirectorName',
    'TR.ODDirectorTenure',
    'TR.ODIndependentDirector',
    'TR.ODCommitteeMembership',
    'TR.ODAuditCommittee',
    'TR.ODCompCommittee',
    'TR.ODNominatingCommittee',
    'TR.ODOfficerPersonAge',
    'TR.ODOfficerUniversityName'
], {'ODRnk': 'R1:R15'})
```

---

### 4.10 Level 10: Complex Governance Analysis

**Q54:** "Analyze governance structure of S&P 500 tech sector"
```python
# Step 1: Get list of tech companies
# Step 2: For each company, get board composition
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'AMZN.O', 'META.O', 'NVDA.O', 'AVGO.O', 'CRM.N', 'ORCL.N', 'ADBE.O'], [
    'TR.CommonName',
    'TR.CEOName',
    'TR.ODOfficerTotalComp',
    'TR.ChairmanName',
    'TR.ODDirectorName',
    'TR.ODIndependentDirector'
], {'ODRnk': 'R1:R15'})
```

**Q55:** "Find companies where CEO is also Chairman"
```python
get_data(['AAPL.O', 'MSFT.O', 'GOOGL.O', 'JPM.N', 'BAC.N'], ['TR.CommonName', 'TR.CEOName', 'TR.ChairmanName'])
# Compare names client-side
```

---

## 5. Natural Language Variations

### 5.1 Executive Title Synonyms

| Expression | Maps To |
|------------|---------|
| CEO, chief executive, head | `TR.CEOName` |
| CFO, chief financial officer, finance chief | `TR.CFOName` |
| chairman, chair, board chair | `TR.ChairmanName` |
| COO, chief operating officer | Search by title |
| CTO, chief technology officer | Search by title |
| president | Search by title |
| executive, officer, leader | `TR.OfficerName` |
| director, board member | `TR.ODDirectorName` |

### 5.2 Compensation Synonyms

| Expression | Maps To |
|------------|---------|
| salary, base salary, base pay | `TR.ODOfficerSalary` |
| bonus, annual bonus, cash bonus | `TR.ODOfficerBonus` |
| stock awards, equity awards, RSUs | `TR.ODOfficerStockAwards` |
| options, stock options | `TR.ODOfficerOptionAwards` |
| total comp, total pay, total compensation | `TR.ODOfficerTotalComp` |
| how much does X earn/make | `TR.ODOfficerTotalComp` |

### 5.3 Query Variations

| Expression | Intent |
|------------|--------|
| "who runs X" | CEO query |
| "leadership team" | Top 10 officers |
| "management" | Executive officers |
| "board" | Directors |
| "how long has X been" | Tenure query |
| "background of" | Biography/education |

---

## 6. Error Handling

### 6.1 Common Issues

| Scenario | Expected Handling |
|----------|-------------------|
| Private company | Limited or no data available |
| Recently appointed | May not yet be in database |
| Historical data requested | Return error: "Static field - current data only" |
| Compensation not disclosed | Return null/not available |
| Non-US company | May have limited compensation data |

### 6.2 Data Limitations

- Compensation data primarily available for US public companies
- Some international markets have limited disclosure requirements
- Historical officer changes not available via API
- Real-time updates may lag press announcements

---

## 7. Complexity Classification

| Level | Criteria | Examples |
|-------|----------|----------|
| 1 | Single executive, single field | CEO name (Q1-Q8) |
| 2 | Multiple officers, basic info | Executive list (Q9-Q16) |
| 3 | Officer details, tenure | Tenure queries (Q17-Q23) |
| 4 | Biography, education | Background queries (Q24-Q29) |
| 5 | Compensation data | Pay queries (Q30-Q37) |
| 6 | Committee membership | Committee queries (Q38-Q42) |
| 7 | Governance analysis | Independence (Q43-Q45) |
| 8 | Multi-company comparison | Comparative (Q46-Q50) |
| 9 | Complete profiles | Full packages (Q51-Q53) |
| 10 | Complex analysis | Governance analysis (Q54-Q55) |

---

## 8. Screening Queries (Find Companies by Officer/Director Criteria)

### 8.1 Two-Step Pattern for Officer Screening

Officer/Director screening typically requires fetching a universe first, then filtering:

```python
# Step 1: Get universe (e.g., S&P 500)
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)

# Step 2: Get officer data
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])

# Step 3: Sort/filter client-side
df.sort_values('ODOfficerTotalComp', ascending=False).head(10)
```

### 8.2 Level 1: Compensation-Based Screening

**Q56:** "Which S&P 500 companies have the highest paid CEOs?"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
df.sort_values('ODOfficerTotalComp', ascending=False).head(10)
```

**Q57:** "Find the lowest paid CEOs among large companies"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), TR.CompanyMarketCap(Scale=9)>=10, CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
df[df['ODOfficerTotalComp'] > 0].sort_values('ODOfficerTotalComp', ascending=True).head(10)
```

**Q58:** "Show tech companies with highest executive compensation"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), TR.CompanyMarketCap(Scale=9)>=10, CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
df.sort_values('ODOfficerTotalComp', ascending=False).head(10)
```

**Q59:** "Compare CEO pay at major banks"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCBusinessSectorCode,"5510"), TR.CompanyMarketCap(Scale=9)>=50, CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerTotalComp'])
df.sort_values('ODOfficerTotalComp', ascending=False)
```

### 8.3 Level 2: Tenure-Based Screening

**Q60:** "Find companies where the CEO has been in role for over 10 years"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.OfficerTitleSince'])
# Filter for tenure > 10 years client-side (compare dates)
```

**Q61:** "Show companies with new CEOs (appointed in last 2 years)"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.OfficerTitleSince'])
# Filter for recent appointments client-side
```

**Q62:** "Get companies with longest-tenured CEOs"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.OfficerTitleSince'])
df.sort_values('OfficerTitleSince', ascending=True).head(10)  # Earliest = longest tenure
```

### 8.4 Level 3: Board Composition Screening

**Q63:** "Find companies with the largest boards"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
# Get director count per company
results = []
for ric in constituents:
    directors = get_data([ric], ['TR.ODDirectorName'], {'ODRnk': 'R1:R20'})
    results.append({'RIC': ric, 'BoardSize': len(directors)})
df = pd.DataFrame(results).sort_values('BoardSize', ascending=False).head(10)
```

**Q64:** "Show companies with mostly independent boards (>80% independent)"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.ODDirectorName', 'TR.ODIndependentDirector'], {'ODRnk': 'R1:R15'})
# Calculate independence ratio client-side
```

**Q65:** "Find companies where CEO is also Chairman"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, ['TR.CommonName', 'TR.CEOName', 'TR.ChairmanName'])
df[df['CEOName'] == df['ChairmanName']]  # Filter where names match
```

### 8.5 Level 4: Combined Governance Screens

**Q66:** "Find well-governed companies: independent board, separate CEO/Chair, reasonable pay"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, [
    'TR.CommonName',
    'TR.CEOName',
    'TR.ChairmanName',
    'TR.ODOfficerTotalComp',
    'TR.ODDirectorName',
    'TR.ODIndependentDirector'
], {'ODRnk': 'R1:R15'})
# Apply governance filters client-side
```

**Q67:** "Show companies with diverse leadership (multiple officer backgrounds)"
```python
constituents = get_data(tickers='LS&PCOMP|L', fields=['RIC'], kind=0)
df = get_data(constituents, [
    'TR.CommonName',
    'TR.OfficerName',
    'TR.OfficerTitle',
    'TR.ODOfficerUniversityName'
], {'RNK': 'R1:R10'})
# Analyze diversity client-side
```

### 8.6 Level 5: Sector Officer Comparisons

**Q68:** "Compare average CEO pay across sectors"
```python
# Get companies by sector
sectors = ['57', '55', '51', '52']  # Tech, Healthcare, Energy, Materials
results = []
for sector in sectors:
    screen_exp = f'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"{sector}"), TR.CompanyMarketCap(Scale=9)>=10, CURN=USD)'
    df = get_data([screen_exp], ['TR.CommonName', 'TR.TRBCEconomicSector', 'TR.ODOfficerTotalComp'])
    avg_pay = df['ODOfficerTotalComp'].mean()
    results.append({'Sector': sector, 'AvgCEOPay': avg_pay})
```

**Q69:** "Find the oldest/youngest executive teams by sector"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), TR.CompanyMarketCap(Scale=9)>=10, CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.OfficerName', 'TR.OfficerAge'], {'RNK': 'R1:R5'})
# Calculate average age per company, then rank
```

**Q70:** "Show tech companies with Ivy League educated CEOs"
```python
screen_exp = 'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"57"), CURN=USD)'
df = get_data([screen_exp], ['TR.CommonName', 'TR.CEOName', 'TR.ODOfficerUniversityName'])
ivy_schools = ['Harvard', 'Yale', 'Princeton', 'Columbia', 'Penn', 'Brown', 'Dartmouth', 'Cornell']
# Filter client-side
```

---

## Sources

- [LSEG Officers & Directors Database](https://www.lseg.com/en/data-analytics/financial-data/company-data/company-profile-information/company-officers-directors-database-search)
- [LSEG Developer Community - Officer Data](https://community.developers.refinitiv.com/questions/84388/officer-and-directors-data.html)
- [Refinitiv Data Library for Python](https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-library-for-python)
