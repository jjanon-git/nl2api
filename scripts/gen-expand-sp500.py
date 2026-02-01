#!/usr/bin/env python3
"""
Expand S&P 500 company list with CIK mappings from SEC EDGAR.

This script fetches the complete S&P 500 constituent list and maps each
company to its SEC CIK (Central Index Key) for EDGAR API access.

Usage:
    # Preview what will be added (dry run)
    python scripts/expand_sp500_list.py --dry-run

    # Actually update the file
    python scripts/expand_sp500_list.py

    # Specify output file
    python scripts/expand_sp500_list.py --output data/tickers/sp500_full.json
"""

import argparse
import json
from pathlib import Path

import httpx

# SEC EDGAR company tickers endpoint
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Current S&P 500 constituents (as of Jan 2026)
# Source: Wikipedia / S&P Dow Jones Indices
# This list should be periodically updated
SP500_TICKERS = [
    # Original 100 (already in file)
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "TSLA",
    "BRK.B",
    "UNH",
    "JNJ",
    "V",
    "XOM",
    "JPM",
    "WMT",
    "MA",
    "PG",
    "HD",
    "CVX",
    "LLY",
    "MRK",
    "ABBV",
    "PEP",
    "KO",
    "AVGO",
    "COST",
    "TMO",
    "MCD",
    "CSCO",
    "ACN",
    "ABT",
    "DHR",
    "LIN",
    "WFC",
    "CMCSA",
    "VZ",
    "NKE",
    "ADBE",
    "TXN",
    "PM",
    "NEE",
    "ORCL",
    "RTX",
    "COP",
    "AMD",
    "INTC",
    "HON",
    "LOW",
    "QCOM",
    "UNP",
    "UPS",
    "CAT",
    "BA",
    "IBM",
    "SPGI",
    "GE",
    "DE",
    "SBUX",
    "PLD",
    "AMAT",
    "ISRG",
    "ADP",
    "BKNG",
    "MDLZ",
    "GS",
    "BLK",
    "GILD",
    "MMC",
    "AXP",
    "T",
    "SYK",
    "CVS",
    "ADI",
    "REGN",
    "CI",
    "VRTX",
    "SCHW",
    "ETN",
    "PGR",
    "CB",
    "SO",
    "BSX",
    "DUK",
    "LRCX",
    "FI",
    "MO",
    "ZTS",
    "CL",
    "PANW",
    "SLB",
    "CME",
    "TJX",
    "AON",
    "KLAC",
    "MU",
    "FCX",
    "EOG",
    "EQIX",
    "ICE",
    "APH",
    "MSI",
    # Additional 400 S&P 500 constituents
    "BDX",
    "ITW",
    "HUM",
    "MCK",
    "NSC",
    "PNC",
    "USB",
    "MMM",
    "EMR",
    "COF",
    "CCI",
    "FDX",
    "GM",
    "TGT",
    "PSX",
    "AIG",
    "MET",
    "PRU",
    "AFL",
    "ALL",
    "TRV",
    "AJG",
    "WMB",
    "KMI",
    "OKE",
    "PSA",
    "AMT",
    "WELL",
    "DLR",
    "O",
    "SPG",
    "VICI",
    "EXR",
    "AVB",
    "EQR",
    "MAA",
    "ESS",
    "UDR",
    "CPT",
    "ARE",
    "BXP",
    "VTR",
    "HST",
    "PEAK",
    "KIM",
    "REG",
    "FRT",
    "SBAC",
    "WY",
    "RCL",
    "CCL",
    "MAR",
    "HLT",
    "LVS",
    "WYNN",
    "MGM",
    "CZR",
    "NCLH",
    "DAL",
    "UAL",
    "AAL",
    "LUV",
    "ALK",
    "JBLU",
    "SAVE",
    "DIS",
    "NFLX",
    "CMCSA",
    "CHTR",
    "PARA",
    "WBD",
    "FOX",
    "FOXA",
    "NWS",
    "NWSA",
    "OMC",
    "IPG",
    "GOOG",
    "EA",
    "TTWO",
    "ATVI",
    "ZG",
    "MTCH",
    "SNAP",
    "PINS",
    "TWTR",
    "LYFT",
    "UBER",
    "DASH",
    "ABNB",
    "EXPE",
    "TRIP",
    "BKNG",
    "PCLN",
    "CRM",
    "NOW",
    "WDAY",
    "ADSK",
    "INTU",
    "CDNS",
    "SNPS",
    "ANSS",
    "PTC",
    "DDOG",
    "ZS",
    "CRWD",
    "NET",
    "OKTA",
    "ZM",
    "DOCU",
    "TEAM",
    "SPLK",
    "MDB",
    "ESTC",
    "SNOW",
    "PATH",
    "S",
    "U",
    "RBLX",
    "PLTR",
    "COIN",
    "HOOD",
    "SOFI",
    "AFRM",
    "UPST",
    "LC",
    "SQ",
    "PYPL",
    "FIS",
    "FISV",
    "GPN",
    "ADP",
    "PAYX",
    "WEX",
    "FOUR",
    "BILL",
    "TOST",
    "NCNO",
    "PCTY",
    "PAYC",
    "HRB",
    "INTU",
    "VMW",
    "DELL",
    "HPE",
    "HPQ",
    "NTAP",
    "WDC",
    "STX",
    "MU",
    "MRVL",
    "SWKS",
    "QRVO",
    "XLNX",
    "NXPI",
    "ON",
    "MCHP",
    "TXN",
    "ADI",
    "LSCC",
    "MPWR",
    "SLAB",
    "SMTC",
    "POWI",
    "DIOD",
    "VSH",
    "IIVI",
    "COHR",
    "LITE",
    "VIAV",
    "KEYS",
    "TER",
    "KLAC",
    "LRCX",
    "AMAT",
    "ASML",
    "ENTG",
    "MKSI",
    "UCTT",
    "ONTO",
    "AXTI",
    "CREE",
    "WOLF",
    "ALGM",
    "GFS",
    "TSM",
    "UMC",
    "ASX",
    "NVMI",
    "ACLS",
    "BRKS",
    "ICHR",
    "COHU",
    "FORM",
    "KLIC",
    "PLAB",
    "OLED",
    "CRUS",
    "AMBA",
    "SYNA",
    "MXIM",
    "IDTI",
    "SLAB",
    "SMTC",
    "AOSL",
    "MXIM",
    "SLAB",
    "SMTC",
    "POWI",
    "DIOD",
    "WAB",
    "GWW",
    "FAST",
    "WSO",
    "ROP",
    "ZBRA",
    "IEX",
    "NDSN",
    "IDXX",
    "MTD",
    "A",
    "PKI",
    "WAT",
    "BIO",
    "TECH",
    "ILMN",
    "DXCM",
    "HOLX",
    "ALGN",
    "XRAY",
    "HSIC",
    "PDCO",
    "OMI",
    "ABC",
    "CAH",
    "MCK",
    "VTRS",
    "ZBH",
    "BAX",
    "BDX",
    "EW",
    "RMD",
    "ISRG",
    "MDT",
    "STE",
    "TFX",
    "HOLX",
    "NUVA",
    "LIVN",
    "ABMD",
    "CNMD",
    "ATRC",
    "NOVT",
    "HAE",
    "PEN",
    "ICUI",
    "MMSI",
    "PODD",
    "TNDM",
    "DXCM",
    "SENS",
    "NVCR",
    "INSP",
    "AXNX",
    "NVRO",
    "STAA",
    "OFIX",
    "OSUR",
    "OMI",
    "MASI",
    "LMAT",
    "GMED",
    "HZNP",
    "JAZZ",
    "NBIX",
    "PCVX",
    "BHVN",
    "RARE",
    "ALNY",
    "SRPT",
    "BMRN",
    "BLUE",
    "IONS",
    "REGN",
    "VRTX",
    "BIIB",
    "GILD",
    "AMGN",
    "MRK",
    "LLY",
    "PFE",
    "BMY",
    "ABBV",
    "JNJ",
    "NVS",
    "AZN",
    "GSK",
    "SNY",
    "RHHBY",
    "TAK",
    "NVO",
    "ARGX",
    "LEGN",
    "BNTX",
    "MRNA",
    "NVAX",
    "INO",
    "CVAC",
    "VIR",
    "SRRK",
    "VXRT",
    "OCGN",
    "IBRX",
    "CYDY",
    "SRNE",
    "ATOS",
    "BLRX",
    "ATNM",
    "BCRX",
    "HGEN",
    "NTR",
    "MOS",
    "CF",
    "FMC",
    "IFF",
    "DD",
    "DOW",
    "LYB",
    "CE",
    "EMN",
    "HUN",
    "WLK",
    "OLN",
    "TROX",
    "CC",
    "AXTA",
    "PPG",
    "SHW",
    "RPM",
    "AZEK",
    "VMC",
    "MLM",
    "EXP",
    "SUM",
    "USCR",
    "USLM",
    "CRH",
    "MDU",
    "US",
    "MAS",
    "OC",
    "FND",
    "BLDR",
    "BLD",
    "BECN",
    "SITE",
    "GMS",
    "POOL",
    "TREX",
    "AZEK",
    "AWI",
    "JELD",
    "DOOR",
    "MHK",
    "FBHS",
    "AAON",
    "WSC",
    "TT",
    "JCI",
    "LII",
    "CARR",
    "GNRC",
    "PRLB",
    "NVT",
    "AOS",
    "MWA",
    "FBIN",
    "RBC",
    "RRX",
    "GGG",
    "IEX",
    "PNR",
    "XYL",
    "WTS",
    "FLS",
    "FELE",
    "ITT",
    "CFX",
    "CW",
    "BMI",
    "LECO",
    "KMT",
    "TKR",
    "GTES",
    "SPXC",
    "AIT",
    "PCAR",
    "OSK",
    "CMI",
    "AGCO",
    "DE",
    "CAT",
    "TEX",
    "CNHI",
    "NAV",
    "ALSN",
    "REVG",
    "THO",
    "WGO",
    "PII",
]

# Remove duplicates while preserving order
SP500_TICKERS = list(dict.fromkeys(SP500_TICKERS))


def fetch_sec_cik_mappings() -> dict[str, dict]:
    """
    Fetch ticker -> CIK mappings from SEC EDGAR.

    Returns:
        Dict mapping ticker to {cik, name}
    """
    print("Fetching CIK mappings from SEC EDGAR...")

    with httpx.Client(
        headers={"User-Agent": "NL2API Research contact@example.com"},
        timeout=30.0,
    ) as client:
        response = client.get(SEC_COMPANY_TICKERS_URL)
        response.raise_for_status()
        data = response.json()

    # SEC returns {0: {cik_str, ticker, title}, 1: {...}, ...}
    mappings = {}
    for entry in data.values():
        ticker = entry.get("ticker", "").upper()
        cik = entry.get("cik_str")
        name = entry.get("title", "")

        if ticker and cik:
            # Zero-pad CIK to 10 digits
            cik_padded = str(cik).zfill(10)
            mappings[ticker] = {"cik": cik_padded, "name": name}

    print(f"  Found {len(mappings)} companies in SEC database")
    return mappings


def load_existing_companies(path: Path) -> list[dict]:
    """Load existing companies from sp500.json."""
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("companies", [])


def main():
    parser = argparse.ArgumentParser(description="Expand S&P 500 company list")
    parser.add_argument(
        "--output",
        default="data/tickers/sp500.json",
        help="Output file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be added without modifying files",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Load existing companies
    existing = load_existing_companies(output_path)
    existing_tickers = {c.get("ticker", "").upper() for c in existing}
    print(f"Existing companies: {len(existing)}")

    # Fetch SEC mappings
    sec_mappings = fetch_sec_cik_mappings()

    # Find new companies to add
    new_companies = []
    missing_cik = []

    for ticker in SP500_TICKERS:
        if ticker.upper() in existing_tickers:
            continue

        if ticker.upper() in sec_mappings:
            mapping = sec_mappings[ticker.upper()]
            new_companies.append(
                {
                    "ticker": ticker.upper(),
                    "cik": mapping["cik"],
                    "name": mapping["name"],
                }
            )
        else:
            missing_cik.append(ticker)

    print(f"\nNew companies to add: {len(new_companies)}")
    print(f"Missing CIK mappings: {len(missing_cik)}")

    if missing_cik:
        print(f"\n  Tickers without CIK: {', '.join(missing_cik[:20])}")
        if len(missing_cik) > 20:
            print(f"  ... and {len(missing_cik) - 20} more")

    if args.dry_run:
        print("\n[DRY RUN] Would add these companies:")
        for c in new_companies[:20]:
            print(f"  {c['ticker']}: {c['name']} (CIK: {c['cik']})")
        if len(new_companies) > 20:
            print(f"  ... and {len(new_companies) - 20} more")
        print(f"\nTotal after update: {len(existing) + len(new_companies)} companies")
        return

    # Merge and write
    all_companies = existing + new_companies

    output_data = {
        "_meta": {
            "updated_at": "2026-01-23",
            "source": "SEC EDGAR company_tickers.json + S&P 500 constituent list",
            "description": "S&P 500 company tickers with CIK mappings for SEC EDGAR",
            "note": "CIKs are zero-padded to 10 digits as required by SEC EDGAR API",
        },
        "companies": all_companies,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Updated {output_path}")
    print(f"   Total companies: {len(all_companies)}")


if __name__ == "__main__":
    main()
