"""
Generate company_mappings.json from S&P 500 and major global companies.

Sources:
- S&P 500 constituents
- FTSE 100
- Major tech, finance, healthcare companies

Run: python scripts/generate_company_mappings.py
"""

import json
from pathlib import Path

# S&P 500 + major global companies
# Format: (primary_name, ric, aliases)
COMPANIES = [
    # Technology
    ("apple", "AAPL.O", ["apple inc", "apple computer"]),
    ("microsoft", "MSFT.O", ["microsoft corp", "msft"]),
    ("alphabet", "GOOGL.O", ["google", "goog"]),
    ("amazon", "AMZN.O", ["amazon.com", "amzn"]),
    ("meta", "META.O", ["facebook", "meta platforms", "fb"]),
    ("nvidia", "NVDA.O", ["nvidia corp"]),
    ("tesla", "TSLA.O", ["tesla motors", "tsla"]),
    ("broadcom", "AVGO.O", []),
    ("oracle", "ORCL.N", ["oracle corp"]),
    ("salesforce", "CRM.N", ["salesforce.com"]),
    ("adobe", "ADBE.O", ["adobe systems"]),
    ("cisco", "CSCO.O", ["cisco systems"]),
    ("intel", "INTC.O", ["intel corp"]),
    ("amd", "AMD.O", ["advanced micro devices"]),
    ("ibm", "IBM.N", ["international business machines"]),
    ("qualcomm", "QCOM.O", []),
    ("texas instruments", "TXN.O", ["ti"]),
    ("applied materials", "AMAT.O", []),
    ("intuit", "INTU.O", []),
    ("servicenow", "NOW.N", []),

    # Finance
    ("jpmorgan", "JPM.N", ["jp morgan", "jpmorgan chase", "jpm", "chase"]),
    ("bank of america", "BAC.N", ["bofa", "boa"]),
    ("wells fargo", "WFC.N", []),
    ("goldman sachs", "GS.N", ["goldman"]),
    ("morgan stanley", "MS.N", []),
    ("citigroup", "C.N", ["citi", "citibank"]),
    ("blackrock", "BLK.N", []),
    ("charles schwab", "SCHW.N", ["schwab"]),
    ("american express", "AXP.N", ["amex"]),
    ("visa", "V.N", []),
    ("mastercard", "MA.N", []),
    ("paypal", "PYPL.O", []),
    ("s&p global", "SPGI.N", ["s&p", "standard and poors"]),
    ("moody's", "MCO.N", ["moodys"]),
    ("cme group", "CME.O", ["cme"]),
    ("intercontinental exchange", "ICE.N", ["ice"]),
    ("nasdaq", "NDAQ.O", []),

    # Healthcare
    ("unitedhealth", "UNH.N", ["unitedhealth group", "united health"]),
    ("johnson & johnson", "JNJ.N", ["j&j", "jnj"]),
    ("eli lilly", "LLY.N", ["lilly"]),
    ("pfizer", "PFE.N", []),
    ("abbvie", "ABBV.N", []),
    ("merck", "MRK.N", ["merck & co"]),
    ("thermo fisher", "TMO.N", ["thermo fisher scientific"]),
    ("abbott", "ABT.N", ["abbott laboratories"]),
    ("danaher", "DHR.N", []),
    ("amgen", "AMGN.O", []),
    ("gilead", "GILD.O", ["gilead sciences"]),
    ("regeneron", "REGN.O", []),
    ("vertex", "VRTX.O", ["vertex pharmaceuticals"]),
    ("moderna", "MRNA.O", []),
    ("intuitive surgical", "ISRG.O", []),
    ("boston scientific", "BSX.N", []),
    ("medtronic", "MDT.N", []),
    ("stryker", "SYK.N", []),
    ("cigna", "CI.N", []),
    ("cvs health", "CVS.N", ["cvs"]),

    # Consumer
    ("walmart", "WMT.N", ["wal-mart"]),
    ("amazon", "AMZN.O", ["amazon.com"]),
    ("costco", "COST.O", []),
    ("home depot", "HD.N", []),
    ("procter & gamble", "PG.N", ["p&g", "procter and gamble"]),
    ("coca-cola", "KO.N", ["coke"]),
    ("pepsico", "PEP.O", ["pepsi"]),
    ("nike", "NKE.N", []),
    ("mcdonald's", "MCD.N", ["mcdonalds"]),
    ("starbucks", "SBUX.O", []),
    ("disney", "DIS.N", ["walt disney"]),
    ("netflix", "NFLX.O", []),
    ("booking holdings", "BKNG.O", ["booking.com", "priceline"]),
    ("lowe's", "LOW.N", ["lowes"]),
    ("target", "TGT.N", []),

    # Industrials
    ("boeing", "BA.N", []),
    ("caterpillar", "CAT.N", ["cat"]),
    ("general electric", "GE.N", ["ge"]),
    ("honeywell", "HON.O", []),
    ("3m", "MMM.N", []),
    ("lockheed martin", "LMT.N", ["lockheed"]),
    ("raytheon", "RTX.N", ["rtx"]),
    ("union pacific", "UNP.N", []),
    ("ups", "UPS.N", ["united parcel service"]),
    ("fedex", "FDX.N", []),
    ("deere", "DE.N", ["john deere"]),

    # Energy
    ("exxon", "XOM.N", ["exxon mobil", "exxonmobil"]),
    ("chevron", "CVX.N", []),
    ("conocophillips", "COP.N", []),
    ("schlumberger", "SLB.N", []),
    ("eog resources", "EOG.N", ["eog"]),
    ("pioneer natural resources", "PXD.N", ["pioneer"]),

    # Telecom/Media
    ("at&t", "T.N", ["att"]),
    ("verizon", "VZ.N", []),
    ("t-mobile", "TMUS.O", ["tmobile"]),
    ("comcast", "CMCSA.O", []),
    ("charter", "CHTR.O", ["charter communications"]),

    # Utilities
    ("nextera energy", "NEE.N", ["nextera"]),
    ("duke energy", "DUK.N", ["duke"]),
    ("southern company", "SO.N", ["southern"]),

    # Real Estate
    ("american tower", "AMT.N", []),
    ("prologis", "PLD.N", []),
    ("equinix", "EQIX.O", []),

    # International
    ("berkshire hathaway", "BRKa.N", ["berkshire", "brk"]),
    ("toyota", "7203.T", ["toyota motor"]),
    ("samsung", "005930.KS", ["samsung electronics"]),
    ("nestle", "NESN.S", []),
    ("lvmh", "LVMH.PA", []),
    ("novo nordisk", "NOVOb.CO", []),
    ("asml", "ASML.AS", []),
    ("shell", "SHEL.L", ["royal dutch shell"]),
    ("bp", "BP.L", ["british petroleum"]),
    ("hsbc", "HSBA.L", []),
]


def generate_mappings():
    """Generate the mappings JSON file."""
    mappings = {}
    tickers = {}

    for primary, ric, aliases in COMPANIES:
        # Extract ticker from RIC
        ticker = ric.split(".")[0]

        mappings[primary] = {
            "ric": ric,
            "aliases": aliases,
        }

        # Add ticker mapping
        tickers[ticker] = ric

        # Also add uppercase ticker as alias
        if ticker.upper() not in [a.upper() for a in aliases]:
            mappings[primary]["aliases"].append(ticker.lower())

    data = {
        "version": "1.0.0",
        "updated": "2026-01-20",
        "company_count": len(mappings),
        "mappings": mappings,
        "tickers": tickers,
    }

    output_path = Path("src/nl2api/resolution/data/company_mappings.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(mappings)} company mappings to {output_path}")


if __name__ == "__main__":
    generate_mappings()
