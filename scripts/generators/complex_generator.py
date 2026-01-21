"""
Complex Generator - Generates multi-step workflow and complex analysis test cases.

Target: ~2000 test cases
- Cross-API workflows (index cascades, screen-then-analyze)
- Multi-step calculation patterns (DCF, DuPont, credit analysis)
- Portfolio analytics workflows
- Event-driven analysis
- Time series analysis
- Edge cases
"""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base_generator import BaseGenerator, TestCase


class ComplexGenerator(BaseGenerator):
    """Generator for complex multi-step test cases."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.category = "complex"

        # Load complex workflow definitions
        self.complex_workflows = self._load_json("complex_workflows.json")
        self.financial_models = self._load_json("financial_models.json")

    # =========================================================================
    # PHASE 1: Cross-API Workflow Templates (~800 tests)
    # =========================================================================

    def _generate_index_cascades(self) -> List[TestCase]:
        """Generate index cascade workflows: get constituents → query all for metrics."""
        test_cases = []

        indices = self.complex_workflows.get("indices_for_cascades", [])
        cascade_metrics = self.complex_workflows.get("cascade_metrics", {})
        templates = self.complex_workflows.get("cascade_nl_templates", [])

        # Metric name mappings
        metric_names = {
            "P": "current prices",
            "MV": "market capitalization",
            "VO": "trading volume",
            "PE": "PE ratios",
            "PTBV": "price-to-book ratios",
            "DY": "dividend yields",
            "EVEBID": "EV/EBITDA ratios",
            "EV": "enterprise values",
            "WC01001": "revenue",
            "WC01751": "net income",
            "WC08301": "ROE",
            "WC08224": "debt-to-equity ratios",
            "WC08366": "net profit margins",
            "WC08316": "ROA",
            "WC08326": "gross margins",
            "WC08131": "asset turnover",
            "WC08126": "inventory turnover",
            "WC04860": "operating cash flow",
            "WC04601": "capital expenditures",
            "WC04851": "free cash flow",
            "WC08106": "current ratios",
            "WC08101": "quick ratios",
            "WC02001": "cash positions"
        }

        # Flatten all metrics
        all_metrics = []
        for category, metrics in cascade_metrics.items():
            all_metrics.extend(metrics)

        for index in indices:
            index_symbol = index["symbol"]
            index_name = index["name"]

            for metric in all_metrics:
                metric_name = metric_names.get(metric, metric)

                # Select a template
                template = random.choice(templates)
                nl_query = template.format(metric_name=metric_name, index_name=index_name)

                # Create 2-step workflow
                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": index_symbol,
                            "fields": ["RIC"],
                            "kind": 0
                        },
                        "description": f"Get {index_name} constituents"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "{{constituents}}",
                            "fields": [metric],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": f"Get {metric_name} for all constituents"
                    }
                ]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory="index_cascade",
                    complexity=8,
                    tags=["complex", "multi_step", "index_cascade", "cross_api"],
                    metadata={
                        "index": index_name,
                        "index_symbol": index_symbol,
                        "metric": metric,
                        "step_count": 2
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_screen_then_analyze(self) -> List[TestCase]:
        """Generate screen-then-analyze workflows: SCREEN → detailed analysis."""
        test_cases = []

        screen_criteria = self.complex_workflows.get("screen_criteria", {})
        analysis_packages = self.complex_workflows.get("analysis_packages", {})
        templates = self.complex_workflows.get("screen_then_analyze_templates", [])

        # Flatten all screens
        all_screens = []
        for category, screens in screen_criteria.items():
            all_screens.extend(screens)

        for screen in all_screens:
            screen_name = screen["name"]
            screen_filter = screen["filter"]

            for pkg_key, package in analysis_packages.items():
                pkg_name = package["name"]
                pkg_fields = package["fields"]

                # Select template
                template = random.choice(templates)
                nl_query = template.format(screen_name=screen_name, package_name=pkg_name)

                # Create 2-step workflow
                steps = [
                    {
                        "tool_name": "screen",
                        "arguments": {
                            "expression": f"SCREEN(U(IN(Equity(active,public,primary))), {screen_filter}, TOP(100))",
                            "output_fields": ["TR.CommonName", "RIC"]
                        },
                        "description": f"Screen for {screen_name}"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": "{{screened_securities}}",
                            "fields": pkg_fields
                        },
                        "description": f"Get {pkg_name} for screened securities"
                    }
                ]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory="screen_then_analyze",
                    complexity=9,
                    tags=["complex", "multi_step", "screen", "cross_api"],
                    metadata={
                        "screen_name": screen_name,
                        "screen_filter": screen_filter,
                        "analysis_package": pkg_name,
                        "step_count": 2
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_company_vs_industry(self) -> List[TestCase]:
        """Generate company vs industry comparison workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=50)
        metrics = self.complex_workflows.get("company_vs_industry_metrics", [])
        templates = self.complex_workflows.get("company_vs_industry_templates", [])

        for ticker_info in tickers:
            ticker = ticker_info.get("ric", ticker_info.get("symbol", ""))
            company = ticker_info.get("name", "Company")
            sector = ticker_info.get("sector", "Unknown")

            for metric_info in metrics:
                metric_field = metric_info["field"]
                metric_name = metric_info["name"]

                # Select template
                template = random.choice(templates)
                nl_query = template.format(company=company, metric_name=metric_name)

                # Create 3-step workflow
                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": [ticker],
                            "fields": [metric_field, "TR.TRBCIndustryCode", "TR.TRBCIndustry"]
                        },
                        "description": f"Get {company}'s {metric_name} and industry code"
                    },
                    {
                        "tool_name": "screen",
                        "arguments": {
                            "expression": "SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCIndustryCode, '{{industry_code}}'), TOP(50))",
                            "output_fields": ["TR.CommonName", "RIC", metric_field]
                        },
                        "description": "Get industry peers with same metric"
                    },
                    {
                        "tool_name": "calculate",
                        "arguments": {
                            "operation": "statistics",
                            "field": metric_field,
                            "stats": ["median", "mean", "percentile_25", "percentile_75"]
                        },
                        "description": "Calculate industry statistics for comparison"
                    }
                ]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory="company_vs_industry",
                    complexity=10,
                    tags=["complex", "multi_step", "comparison", "industry_benchmark"],
                    metadata={
                        "ticker": ticker,
                        "company": company,
                        "metric": metric_name,
                        "sector": sector,
                        "step_count": 3
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_historical_forward_combinations(self) -> List[TestCase]:
        """Generate historical + forward estimate combination workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=50)
        combinations = self.complex_workflows.get("historical_forward_combinations", [])

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            ric = ticker_info.get("ric", ticker)
            company = ticker_info.get("name", "Company")

            for combo in combinations:
                nl_template = combo.get("nl_template", "")
                nl_query = nl_template.format(company=company)

                historical_fields = combo.get("historical_fields", [])
                historical_years = combo.get("historical_years", 5)
                current_fields = combo.get("current_fields", [])
                forward_fields = combo.get("forward_fields", [])

                # Create 3-step workflow
                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": ticker,
                            "fields": historical_fields,
                            "start": f"-{historical_years}Y",
                            "end": "0D",
                            "freq": "Y",
                            "kind": 0
                        },
                        "description": f"Get {historical_years}-year historical data (Datastream)"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": [ric],
                            "fields": current_fields
                        },
                        "description": "Get current values (Refinitiv)"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": [ric],
                            "fields": forward_fields
                        },
                        "description": "Get forward estimates (Refinitiv)"
                    }
                ]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory="historical_forward",
                    complexity=10,
                    tags=["complex", "multi_step", "cross_api", "time_series", "estimates"],
                    metadata={
                        "ticker": ticker,
                        "company": company,
                        "trajectory_type": combo.get("name", ""),
                        "step_count": 3
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    # =========================================================================
    # PHASE 2: Financial Analysis Workflows (~600 tests)
    # =========================================================================

    def _generate_dcf_workflows(self) -> List[TestCase]:
        """Generate DCF valuation model input workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=50)
        dcf_config = self.financial_models.get("dcf_analysis", {})
        templates = dcf_config.get("nl_templates", [])

        step1 = dcf_config.get("step1_historical_cash_flows", {})
        step2 = dcf_config.get("step2_growth_assumptions", {})
        step3 = dcf_config.get("step3_discount_rate_inputs", {})

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            ric = ticker_info.get("ric", ticker)
            company = ticker_info.get("name", "Company")

            template = random.choice(templates)
            nl_query = template.format(company=company)

            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": ticker,
                        "fields": step1.get("fields", []),
                        "start": "-5Y",
                        "end": "0D",
                        "freq": "Y",
                        "kind": 0
                    },
                    "description": "Get 5-year historical cash flows (Datastream)"
                },
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ric],
                        "fields": step2.get("fields", [])
                    },
                    "description": "Get growth assumptions from estimates (Refinitiv)"
                },
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ric],
                        "fields": step3.get("fields", [])
                    },
                    "description": "Get discount rate inputs (Refinitiv)"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="dcf_valuation",
                complexity=12,
                tags=["complex", "multi_step", "cross_api", "valuation", "dcf"],
                metadata={
                    "ticker": ticker,
                    "company": company,
                    "model_type": "DCF",
                    "step_count": 3
                }
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_comparable_analysis(self) -> List[TestCase]:
        """Generate comparable company analysis workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=50)
        comps_config = self.financial_models.get("comparable_company_analysis", {})
        templates = comps_config.get("nl_templates", [])

        step1 = comps_config.get("step1_target_metrics", {})
        step3 = comps_config.get("step3_peer_metrics", {})

        for ticker_info in tickers:
            ticker = ticker_info.get("ric", ticker_info.get("symbol", ""))
            company = ticker_info.get("name", "Company")

            template = random.choice(templates)
            nl_query = template.format(company=company)

            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ticker],
                        "fields": step1.get("fields", [])
                    },
                    "description": f"Get {company}'s valuation multiples and industry"
                },
                {
                    "tool_name": "screen",
                    "arguments": {
                        "expression": "SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCIndustryCode, '{{industry_code}}'), TOP(TR.CompanyMarketCap, 10))",
                        "output_fields": ["TR.CommonName", "RIC"]
                    },
                    "description": "Find top 10 industry peers by market cap"
                },
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": "{{peer_rics}}",
                        "fields": step3.get("fields", [])
                    },
                    "description": "Get peer valuation multiples"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="comparable_analysis",
                complexity=11,
                tags=["complex", "multi_step", "valuation", "comps", "peer_analysis"],
                metadata={
                    "ticker": ticker,
                    "company": company,
                    "model_type": "Comparable Company Analysis",
                    "step_count": 3
                }
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_credit_analysis(self) -> List[TestCase]:
        """Generate credit and financial health analysis workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=25)
        credit_config = self.financial_models.get("credit_analysis", {})
        templates = credit_config.get("nl_templates", [])

        analyses = [
            ("liquidity_analysis", credit_config.get("liquidity_analysis", {})),
            ("leverage_analysis", credit_config.get("leverage_analysis", {})),
            ("coverage_analysis", credit_config.get("coverage_analysis", {})),
            ("altman_z_score", credit_config.get("altman_z_score", {})),
            ("piotroski_f_score", credit_config.get("piotroski_f_score", {}))
        ]

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            for analysis_name, analysis_config in analyses:
                analysis_display_name = analysis_config.get("name", analysis_name)
                fields = analysis_config.get("fields", [])

                # Create specific NL query
                if analysis_name == "altman_z_score":
                    nl_query = f"Calculate Altman Z-Score inputs for {company}"
                elif analysis_name == "piotroski_f_score":
                    nl_query = f"Get Piotroski F-Score components for {company}"
                else:
                    template = random.choice(templates) if templates else f"Analyze {{company}}'s {analysis_display_name.lower()}"
                    nl_query = template.format(company=company)

                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": ticker,
                            "fields": fields,
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": f"Get {analysis_display_name} metrics"
                    }
                ]

                # Add calculation step for Z-score and F-score
                if analysis_name in ["altman_z_score", "piotroski_f_score"]:
                    steps.append({
                        "tool_name": "calculate",
                        "arguments": {
                            "operation": analysis_name,
                            "formula": analysis_config.get("formula", "")
                        },
                        "description": f"Calculate {analysis_display_name}"
                    })

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory=f"credit_{analysis_name}",
                    complexity=8 if len(steps) == 1 else 10,
                    tags=["complex", "analysis", "credit", analysis_name],
                    metadata={
                        "ticker": ticker,
                        "company": company,
                        "analysis_type": analysis_display_name,
                        "step_count": len(steps)
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_portfolio_analytics(self) -> List[TestCase]:
        """Generate portfolio-level analytics workflows."""
        test_cases = []

        portfolio_config = self.financial_models.get("portfolio_analytics", {})
        sample_portfolios = portfolio_config.get("sample_portfolios", [])

        analyses = [
            ("sector_allocation", portfolio_config.get("sector_allocation", {})),
            ("risk_metrics", portfolio_config.get("risk_metrics", {})),
            ("income_analysis", portfolio_config.get("income_analysis", {})),
            ("valuation_analysis", portfolio_config.get("valuation_analysis", {})),
            ("earnings_calendar", portfolio_config.get("earnings_calendar", {})),
            ("correlation_analysis", portfolio_config.get("correlation_analysis", {}))
        ]

        for portfolio in sample_portfolios:
            portfolio_name = portfolio.get("name", "Portfolio")
            holdings = portfolio.get("holdings", [])
            holdings_str = ", ".join(holdings)

            for analysis_name, analysis_config in analyses:
                templates = analysis_config.get("nl_templates", [])
                fields = analysis_config.get("step1_fields", [])

                if not templates:
                    continue

                template = random.choice(templates)
                nl_query = template.format(holdings=holdings_str)

                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": holdings,
                            "fields": fields
                        },
                        "description": f"Get {analysis_config.get('name', analysis_name)} data for portfolio"
                    }
                ]

                # Add calculation step if needed
                calculations = analysis_config.get("calculations", [])
                if calculations:
                    steps.append({
                        "tool_name": "calculate",
                        "arguments": {
                            "operations": calculations
                        },
                        "description": f"Calculate portfolio {analysis_name}"
                    })

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory=f"portfolio_{analysis_name}",
                    complexity=9,
                    tags=["complex", "portfolio", analysis_name],
                    metadata={
                        "portfolio_name": portfolio_name,
                        "holdings": holdings,
                        "analysis_type": analysis_name,
                        "step_count": len(steps)
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_event_driven_analysis(self) -> List[TestCase]:
        """Generate event-driven analysis workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=35)
        event_config = self.financial_models.get("event_driven_analysis", {})

        analyses = [
            ("earnings_surprise_analysis", event_config.get("earnings_surprise_analysis", {})),
            ("estimate_revision_analysis", event_config.get("estimate_revision_analysis", {})),
            ("analyst_rating_changes", event_config.get("analyst_rating_changes", {})),
            ("insider_activity", event_config.get("insider_activity", {})),
            ("institutional_ownership", event_config.get("institutional_ownership", {}))
        ]

        for ticker_info in tickers:
            ticker = ticker_info.get("ric", ticker_info.get("symbol", ""))
            company = ticker_info.get("name", "Company")

            for analysis_name, analysis_config in analyses:
                templates = analysis_config.get("nl_templates", [])
                step1_fields = analysis_config.get("step1_fields", [])
                step2_fields = analysis_config.get("step2_fields", [])

                if not templates:
                    continue

                template = random.choice(templates)
                nl_query = template.format(company=company)

                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": [ticker],
                            "fields": step1_fields
                        },
                        "description": f"Get {analysis_config.get('name', analysis_name)} data"
                    }
                ]

                # Add step 2 if defined (e.g., price data around earnings)
                if step2_fields:
                    step2_params = analysis_config.get("step2_parameters", {})
                    steps.append({
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": ticker,
                            "fields": step2_fields,
                            **step2_params
                        },
                        "description": "Get related price/return data"
                    })

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory=f"event_{analysis_name}",
                    complexity=8 if len(steps) == 1 else 10,
                    tags=["complex", "event_driven", analysis_name],
                    metadata={
                        "ticker": ticker,
                        "company": company,
                        "analysis_type": analysis_name,
                        "step_count": len(steps)
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    # =========================================================================
    # PHASE 3: Multi-Step Calculation Chains (~400 tests)
    # =========================================================================

    def _generate_derived_calculations(self) -> List[TestCase]:
        """Generate derived metric calculation workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=20)
        derived_config = self.financial_models.get("derived_calculations", {})

        for calc_name, calc_config in derived_config.items():
            if calc_name == "description":
                continue

            templates = calc_config.get("nl_templates", [])
            fields = calc_config.get("fields", [])
            formula = calc_config.get("formula", "")
            display_name = calc_config.get("name", calc_name)

            for ticker_info in tickers:
                ticker = ticker_info.get("symbol", "")
                ric = ticker_info.get("ric", ticker)
                company = ticker_info.get("name", "Company")

                template = random.choice(templates) if templates else f"Calculate {{company}}'s {display_name}"
                nl_query = template.format(company=company)

                # Determine API based on field types
                has_wc_fields = any(f.startswith("WC") for f in fields)
                has_tr_fields = any(f.startswith("TR.") for f in fields)
                has_ds_fields = any(f in ["P", "MV", "PE", "PTBV", "DY", "EV", "EVEBID"] for f in fields)

                steps = []

                if has_wc_fields or has_ds_fields:
                    ds_fields = [f for f in fields if f.startswith("WC") or f in ["P", "MV", "PE", "PTBV", "DY", "EV", "EVEBID"]]
                    steps.append({
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": ticker,
                            "fields": ds_fields,
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": f"Get Datastream fields for {display_name}"
                    })

                if has_tr_fields:
                    tr_fields = [f for f in fields if f.startswith("TR.")]
                    steps.append({
                        "tool_name": "get_data",
                        "arguments": {
                            "instruments": [ric],
                            "fields": tr_fields
                        },
                        "description": f"Get Refinitiv fields for {display_name}"
                    })

                # Add calculation step
                steps.append({
                    "tool_name": "calculate",
                    "arguments": {
                        "operation": calc_name,
                        "formula": formula
                    },
                    "description": f"Calculate {display_name}"
                })

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory=f"derived_{calc_name}",
                    complexity=9,
                    tags=["complex", "calculation", "derived_metric", calc_name],
                    metadata={
                        "ticker": ticker,
                        "company": company,
                        "calculation": display_name,
                        "formula": formula,
                        "step_count": len(steps)
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_ranking_operations(self) -> List[TestCase]:
        """Generate ranking and comparison workflows."""
        test_cases = []

        ranking_config = self.complex_workflows.get("ranking_operations", {})
        templates = ranking_config.get("templates", [])
        universes = ranking_config.get("universes", [])
        metrics = ranking_config.get("metrics", [])

        for universe in universes:
            universe_name = universe.get("name", "")
            universe_symbol = universe.get("symbol", "")
            universe_screen = universe.get("screen", "")

            for metric_info in metrics:
                metric_field = metric_info.get("field", "")
                metric_name = metric_info.get("name", "")
                direction = metric_info.get("direction", "desc")

                template = random.choice(templates)
                nl_query = template.format(universe=universe_name, metric=metric_name)

                steps = []

                # Step 1: Get universe constituents
                if universe_symbol:
                    steps.append({
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": universe_symbol,
                            "fields": ["RIC"],
                            "kind": 0
                        },
                        "description": f"Get {universe_name} constituents"
                    })
                elif universe_screen:
                    steps.append({
                        "tool_name": "screen",
                        "arguments": {
                            "expression": f"SCREEN(U(IN(Equity(active,public,primary))), {universe_screen}, TOP(100))",
                            "output_fields": ["TR.CommonName", "RIC"]
                        },
                        "description": f"Screen for {universe_name}"
                    })

                # Step 2: Get metric for all securities
                steps.append({
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": "{{securities}}",
                        "fields": [metric_field, "TR.CommonName"]
                    },
                    "description": f"Get {metric_name} for all securities"
                })

                # Step 3: Sort and rank
                steps.append({
                    "tool_name": "sort_and_select",
                    "arguments": {
                        "field": metric_field,
                        "order": direction,
                        "operation": "rank"
                    },
                    "description": f"Rank by {metric_name}"
                })

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory="ranking",
                    complexity=10,
                    tags=["complex", "ranking", "comparison", "multi_step"],
                    metadata={
                        "universe": universe_name,
                        "metric": metric_name,
                        "direction": direction,
                        "step_count": len(steps)
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_time_series_analysis(self) -> List[TestCase]:
        """Generate time series analysis workflows."""
        test_cases = []

        tickers = self._get_all_tickers(count=10)
        ts_config = self.complex_workflows.get("time_series_analysis", {})

        # CAGR calculations
        cagr_config = ts_config.get("cagr_calculation", {})
        cagr_templates = cagr_config.get("templates", [])
        cagr_metrics = cagr_config.get("metrics", [])
        cagr_periods = cagr_config.get("periods", [])

        metric_to_field = {
            "revenue": "WC01001",
            "earnings": "WC01751",
            "dividends": "WC05101",
            "book value": "WC03501"
        }

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            for metric in cagr_metrics:
                for period in cagr_periods:
                    template = random.choice(cagr_templates) if cagr_templates else "Calculate {company}'s {period}-year {metric} CAGR"
                    nl_query = template.format(company=company, period=period, metric=metric)

                    field = metric_to_field.get(metric, "WC01001")

                    steps = [
                        {
                            "tool_name": "get_data",
                            "arguments": {
                                "tickers": ticker,
                                "fields": [field],
                                "start": f"-{period}Y",
                                "end": "0D",
                                "freq": "Y",
                                "kind": 0
                            },
                            "description": f"Get {period}-year {metric} history"
                        },
                        {
                            "tool_name": "calculate",
                            "arguments": {
                                "operation": "cagr",
                                "period_years": period
                            },
                            "description": "Calculate CAGR"
                        }
                    ]

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=steps,
                        category=self.category,
                        subcategory="time_series_cagr",
                        complexity=8,
                        tags=["complex", "time_series", "cagr", "calculation"],
                        metadata={
                            "ticker": ticker,
                            "company": company,
                            "metric": metric,
                            "period_years": period,
                            "step_count": 2
                        }
                    )

                    if test_case:
                        test_cases.append(test_case)

        # Moving average analysis
        ma_config = ts_config.get("moving_average_analysis", {})
        ma_templates = ma_config.get("templates", [])
        ma_periods = ma_config.get("ma_periods", [])

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            for ma_period in ma_periods:
                template = random.choice(ma_templates) if ma_templates else "Is {company} above its {ma_period}-day moving average?"
                nl_query = template.format(company=company, ma_period=ma_period)

                steps = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": ticker,
                            "fields": ["P"],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": "Get current price"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": f"MAV#({ticker},{ma_period}D)",
                            "fields": ["P"],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": f"Get {ma_period}-day moving average"
                    },
                    {
                        "tool_name": "calculate",
                        "arguments": {
                            "operation": "compare",
                            "comparison": "above_below"
                        },
                        "description": "Compare price to MA"
                    }
                ]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=steps,
                    category=self.category,
                    subcategory="time_series_ma",
                    complexity=9,
                    tags=["complex", "time_series", "moving_average", "technical"],
                    metadata={
                        "ticker": ticker,
                        "company": company,
                        "ma_period": ma_period,
                        "step_count": 3
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        # Relative performance
        perf_config = ts_config.get("relative_performance", {})
        perf_templates = perf_config.get("templates", [])
        perf_periods = perf_config.get("periods", [])
        benchmarks = perf_config.get("benchmarks", [])

        benchmark_symbols = {
            "S&P 500": "S&PCOMP",
            "sector index": "S5INFT",
            "NASDAQ": "NASCOMP"
        }

        for ticker_info in tickers[:5]:  # Limit to avoid explosion
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            for period in perf_periods:
                for benchmark in benchmarks:
                    template = random.choice(perf_templates) if perf_templates else "Compare {company}'s {period} return to {benchmark}"
                    nl_query = template.format(company=company, period=period, benchmark=benchmark)

                    benchmark_sym = benchmark_symbols.get(benchmark, "S&PCOMP")
                    period_map = {"YTD": "-YTD", "1 year": "-1Y", "3 years": "-3Y", "5 years": "-5Y"}
                    start = period_map.get(period, "-1Y")

                    steps = [
                        {
                            "tool_name": "get_data",
                            "arguments": {
                                "tickers": ticker,
                                "fields": ["RI"],
                                "start": start,
                                "end": "0D",
                                "kind": 0
                            },
                            "description": f"Get {company} total return index"
                        },
                        {
                            "tool_name": "get_data",
                            "arguments": {
                                "tickers": benchmark_sym,
                                "fields": ["RI"],
                                "start": start,
                                "end": "0D",
                                "kind": 0
                            },
                            "description": f"Get {benchmark} total return index"
                        },
                        {
                            "tool_name": "calculate",
                            "arguments": {
                                "operation": "relative_return",
                                "metric": "alpha"
                            },
                            "description": "Calculate relative performance"
                        }
                    ]

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=steps,
                        category=self.category,
                        subcategory="time_series_relative",
                        complexity=10,
                        tags=["complex", "time_series", "relative_performance", "benchmark"],
                        metadata={
                            "ticker": ticker,
                            "company": company,
                            "period": period,
                            "benchmark": benchmark,
                            "step_count": 3
                        }
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    # =========================================================================
    # PHASE 4: Edge Cases (~200 tests)
    # =========================================================================

    def _generate_edge_cases(self) -> List[TestCase]:
        """Generate edge case and error handling test cases."""
        test_cases = []

        edge_config = self.financial_models.get("edge_cases", {})

        # Negative earnings companies
        neg_earnings = edge_config.get("negative_earnings", {})
        for ticker in neg_earnings.get("tickers", []):
            nl_query = f"Get PE ratio for {ticker}"
            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ticker],
                        "fields": ["TR.PE", "TR.EVToSales", "TR.PriceToSalesPerShare"]
                    },
                    "description": "Get valuation metrics (PE undefined for negative earnings)"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="edge_negative_earnings",
                complexity=5,
                tags=["edge_case", "negative_earnings", "valuation"],
                metadata={
                    "ticker": ticker,
                    "expected_behavior": neg_earnings.get("expected_behavior", ""),
                    "issue": "PE undefined for unprofitable company"
                }
            )
            if test_case:
                test_cases.append(test_case)

        # Zero debt companies
        zero_debt = edge_config.get("zero_debt", {})
        for ticker in zero_debt.get("tickers", []):
            nl_query = f"Calculate debt-to-equity ratio for {ticker}"
            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ticker],
                        "fields": ["TR.DebtToEquity", "TR.TotalDebtOutstanding", "TR.TotalEquity"]
                    },
                    "description": "Get leverage metrics (D/E = 0 or near zero)"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="edge_zero_debt",
                complexity=4,
                tags=["edge_case", "zero_debt", "leverage"],
                metadata={
                    "ticker": ticker,
                    "expected_behavior": zero_debt.get("expected_behavior", ""),
                    "issue": "Company has minimal or no debt"
                }
            )
            if test_case:
                test_cases.append(test_case)

        # Non-December fiscal year
        non_dec_fy = edge_config.get("non_december_fiscal_year", {})
        for item in non_dec_fy.get("tickers", []):
            ticker = item.get("symbol", "")
            fy_end = item.get("fy_end", "")

            nl_query = f"Get annual revenue for {ticker} aligned with fiscal year"
            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ticker],
                        "fields": ["TR.Revenue", "TR.FiscalYearEnd", "TR.ReportDate"]
                    },
                    "description": f"Get revenue with fiscal year ending in {fy_end}"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="edge_fiscal_year",
                complexity=5,
                tags=["edge_case", "fiscal_year", "timing"],
                metadata={
                    "ticker": ticker,
                    "fiscal_year_end": fy_end,
                    "expected_behavior": non_dec_fy.get("expected_behavior", "")
                }
            )
            if test_case:
                test_cases.append(test_case)

        # ADR vs ordinary
        adr_config = edge_config.get("adr_vs_ordinary", {})
        for pair in adr_config.get("pairs", []):
            adr = pair.get("adr", "")
            ordinary = pair.get("ordinary", "")
            ratio = pair.get("ratio", "1:1")

            nl_query = f"Compare {adr} ADR to {ordinary} ordinary shares valuation"
            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [adr],
                        "fields": ["TR.PriceClose", "TR.SharesOutstanding", "TR.CompanyMarketCap"]
                    },
                    "description": "Get ADR data"
                },
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ordinary],
                        "fields": ["TR.PriceClose", "TR.SharesOutstanding", "TR.CompanyMarketCap"]
                    },
                    "description": "Get ordinary share data"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="edge_adr",
                complexity=7,
                tags=["edge_case", "adr", "share_class", "international"],
                metadata={
                    "adr_ticker": adr,
                    "ordinary_ticker": ordinary,
                    "adr_ratio": ratio,
                    "expected_behavior": adr_config.get("expected_behavior", ""),
                    "step_count": 2
                }
            )
            if test_case:
                test_cases.append(test_case)

        # Recent merger - discontinuous history
        merger_config = edge_config.get("merger_discontinuity", {})
        for example in merger_config.get("examples", []):
            ticker = example.get("symbol", "")
            event = example.get("event", "")
            date = example.get("date", "")

            nl_query = f"Get 10-year revenue history for {ticker}"
            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": ticker,
                        "fields": ["WC01001"],
                        "start": "-10Y",
                        "end": "0D",
                        "freq": "Y",
                        "kind": 0
                    },
                    "description": f"Get 10-year history (note: {event} in {date} may cause discontinuity)"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="edge_merger",
                complexity=6,
                tags=["edge_case", "merger", "discontinuity", "historical"],
                metadata={
                    "ticker": ticker,
                    "event": event,
                    "event_date": date,
                    "expected_behavior": merger_config.get("expected_behavior", "")
                }
            )
            if test_case:
                test_cases.append(test_case)

        # Spinoff - partial history
        spinoff_config = edge_config.get("spinoff", {})
        for example in spinoff_config.get("examples", []):
            ticker = example.get("symbol", "")
            parent = example.get("parent", "")
            date = example.get("date", "")

            nl_query = f"Get complete financial history for {ticker}"
            steps = [
                {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": ticker,
                        "fields": ["WC01001", "WC01751", "WC08301"],
                        "start": "-15Y",
                        "end": "0D",
                        "freq": "Y",
                        "kind": 0
                    },
                    "description": f"Get history (note: spinoff from {parent} in {date}, limited pre-spinoff data)"
                }
            ]

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=steps,
                category=self.category,
                subcategory="edge_spinoff",
                complexity=6,
                tags=["edge_case", "spinoff", "partial_history"],
                metadata={
                    "ticker": ticker,
                    "parent": parent,
                    "spinoff_date": date,
                    "expected_behavior": spinoff_config.get("expected_behavior", "")
                }
            )
            if test_case:
                test_cases.append(test_case)

        # Generate additional edge cases with common patterns
        edge_patterns = [
            {
                "name": "empty_screen",
                "nl": "Find stocks with PE under 1 and revenue over $1 trillion",
                "description": "Screen that returns no results",
                "steps": [{
                    "tool_name": "screen",
                    "arguments": {
                        "expression": "SCREEN(U(IN(Equity(active,public,primary))), TR.PE>0, TR.PE<=1, TR.Revenue(Scale=12)>=1)",
                        "output_fields": ["TR.CommonName", "RIC"]
                    },
                    "description": "Screen with contradictory criteria"
                }]
            },
            {
                "name": "invalid_ticker",
                "nl": "Get price for INVALIDTICKER123",
                "description": "Non-existent ticker",
                "steps": [{
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": "INVALIDTICKER123",
                        "fields": ["P"],
                        "start": "0D",
                        "end": "0D",
                        "kind": 0
                    },
                    "description": "Query invalid ticker"
                }]
            },
            {
                "name": "future_date",
                "nl": "Get Apple's stock price on December 31, 2030",
                "description": "Future date request",
                "steps": [{
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": "@AAPL",
                        "fields": ["P"],
                        "start": "2030-12-31",
                        "end": "2030-12-31",
                        "kind": 0
                    },
                    "description": "Query future date"
                }]
            },
            {
                "name": "very_old_date",
                "nl": "Get Microsoft's revenue in 1975",
                "description": "Date before company existed",
                "steps": [{
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": "U:MSFT",
                        "fields": ["WC01001"],
                        "start": "1975-01-01",
                        "end": "1975-12-31",
                        "freq": "Y",
                        "kind": 0
                    },
                    "description": "Query before company founding (1975, founded 1975 but no public data)"
                }]
            },
            {
                "name": "incompatible_fields",
                "nl": "Get TR.Revenue using Datastream API",
                "description": "Wrong field format for API",
                "steps": [{
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": "@AAPL",
                        "fields": ["TR.Revenue"],
                        "start": "0D",
                        "end": "0D",
                        "kind": 0
                    },
                    "description": "Datastream call with Refinitiv field"
                }]
            }
        ]

        for pattern in edge_patterns:
            test_case = self._create_test_case(
                nl_query=pattern["nl"],
                tool_calls=pattern["steps"],
                category=self.category,
                subcategory=f"edge_{pattern['name']}",
                complexity=5,
                tags=["edge_case", pattern["name"], "error_handling"],
                metadata={
                    "pattern": pattern["name"],
                    "description": pattern["description"]
                }
            )
            if test_case:
                test_cases.append(test_case)

        return test_cases

    # =========================================================================
    # Original Methods (kept for backward compatibility)
    # =========================================================================

    def _generate_multi_step_workflows(self) -> List[TestCase]:
        """Generate multi-step workflow queries (original implementation)."""
        test_cases = []

        workflows = [
            {
                "nl": "Get the prices of all S&P 500 constituents",
                "steps": [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "LS&PCOMP|L",
                            "fields": ["RIC"],
                            "kind": 0
                        },
                        "description": "Get S&P 500 constituents"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "{{constituents}}",
                            "fields": ["P"],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": "Get prices for all constituents"
                    }
                ],
                "complexity": 8
            },
            {
                "nl": "Calculate the average PE ratio of FTSE 100 companies",
                "steps": [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "LFTSE100|L",
                            "fields": ["RIC"],
                            "kind": 0
                        },
                        "description": "Get FTSE 100 constituents"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "{{constituents}}",
                            "fields": ["PE"],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": "Get PE ratios"
                    },
                    {
                        "tool_name": "calculate",
                        "arguments": {
                            "operation": "mean",
                            "field": "PE"
                        },
                        "description": "Calculate average"
                    }
                ],
                "complexity": 10
            },
            {
                "nl": "Find the best performing stock in the DAX over the past year",
                "steps": [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "LDAXINDX|L",
                            "fields": ["RIC"],
                            "kind": 0
                        },
                        "description": "Get DAX constituents"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "{{constituents_pch}}",
                            "fields": ["P"],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": "Get 1-year returns using PCH#"
                    },
                    {
                        "tool_name": "sort_and_select",
                        "arguments": {
                            "operation": "max",
                            "count": 1
                        },
                        "description": "Find maximum"
                    }
                ],
                "complexity": 10
            },
            {
                "nl": "Get the top 10 highest dividend yielding stocks in the FTSE 100",
                "steps": [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "LFTSE100|L",
                            "fields": ["RIC"],
                            "kind": 0
                        },
                        "description": "Get FTSE 100 constituents"
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "{{constituents}}",
                            "fields": ["DY"],
                            "start": "0D",
                            "end": "0D",
                            "kind": 0
                        },
                        "description": "Get dividend yields"
                    },
                    {
                        "tool_name": "sort_and_select",
                        "arguments": {
                            "operation": "top",
                            "count": 10,
                            "order": "desc"
                        },
                        "description": "Sort and take top 10"
                    }
                ],
                "complexity": 10
            }
        ]

        for workflow in workflows:
            test_case = self._create_test_case(
                nl_query=workflow["nl"],
                tool_calls=workflow["steps"],
                category=self.category,
                subcategory="multi_step",
                complexity=workflow["complexity"],
                tags=["complex", "multi_step", "workflow"],
                metadata={
                    "step_count": len(workflow["steps"])
                }
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_dupont_analysis(self) -> List[TestCase]:
        """Generate DuPont analysis queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=20)

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            nl_query = f"Get components for {company}'s DuPont ROE analysis"

            tool_call = {
                "tool_name": "get_data",
                "arguments": {
                    "tickers": ticker,
                    "fields": [
                        "WC08366",  # Net Margin
                        "WC01001",  # Revenue
                        "WC02999",  # Total Assets
                        "WC03501",  # Equity
                        "WC08301"   # ROE
                    ],
                    "start": "0D",
                    "end": "0D",
                    "kind": 0
                }
            }

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=[tool_call],
                category=self.category,
                subcategory="dupont_analysis",
                complexity=6,
                tags=["complex", "analysis", "dupont"],
                metadata={
                    "ticker": ticker,
                    "analysis_type": "DuPont"
                }
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_valuation_model_inputs(self) -> List[TestCase]:
        """Generate valuation model input queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=15)

        models = [
            {
                "name": "EV/EBITDA valuation",
                "fields": ["WC18100", "WC18198", "WC08001", "WC01751"]
            },
            {
                "name": "DCF input gathering",
                "fields": ["WC04860", "WC04601", "WC03255", "WC02001", "WC03501"]
            },
            {
                "name": "comparable company analysis",
                "fields": ["PE", "PTBV", "EVEBID", "DY", "WC08301"]
            }
        ]

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            for model in models:
                nl_query = f"Get {model['name']} inputs for {company}"

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": ticker,
                        "fields": model["fields"],
                        "start": "0D",
                        "end": "0D",
                        "kind": 0
                    }
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="valuation_model",
                    complexity=7,
                    tags=["complex", "valuation", model["name"].replace(" ", "_")],
                    metadata={
                        "ticker": ticker,
                        "model": model["name"]
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_analyst_estimate_packages(self) -> List[TestCase]:
        """Generate comprehensive analyst estimate queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=20)

        estimate_packages = [
            {
                "name": "complete analyst summary",
                "fields": [
                    "TR.EPSMean(Period=FY1)",
                    "TR.EPSMean(Period=FY2)",
                    "TR.RevenueMean(Period=FY1)",
                    "TR.EBITDAMean(Period=FY1)",
                    "TR.EPSHigh(Period=FY1)",
                    "TR.EPSLow(Period=FY1)",
                    "TR.EPSNumIncEstimates(Period=FY1)",
                    "TR.RecMean",
                    "TR.NumBuys",
                    "TR.NumHolds",
                    "TR.NumSells",
                    "TR.PriceTargetMean",
                    "TR.PriceTargetHigh",
                    "TR.PriceTargetLow",
                    "TR.PtoEPSMeanEst(Period=FY1)",
                    "TR.PEGRatio",
                    "TR.LTGMean"
                ]
            },
            {
                "name": "earnings surprise history",
                "fields": [
                    "TR.EPSActValue(Period=FQ0)",
                    "TR.EPSMean(Period=FQ0)",
                    "TR.EPSSurprise(Period=FQ0)",
                    "TR.EPSSurprisePct(Period=FQ0)"
                ]
            },
            {
                "name": "estimate revision trend",
                "fields": [
                    "TR.EPSMeanChgPct(Period=FY1)",
                    "TR.EPSNumUp(Period=FY1)",
                    "TR.EPSNumDown(Period=FY1)",
                    "TR.RevenueMeanChgPct(Period=FY1)"
                ]
            }
        ]

        for ticker_info in tickers:
            ticker = ticker_info.get("ric", ticker_info.get("symbol", ""))
            company = ticker_info.get("name", "Company")

            for package in estimate_packages:
                nl_query = f"Get {package['name']} for {company}"

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ticker],
                        "fields": package["fields"]
                    }
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="estimates_package",
                    complexity=9,
                    tags=["complex", "estimates", package["name"].replace(" ", "_")],
                    metadata={
                        "ticker": ticker,
                        "package": package["name"]
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_officer_director_queries(self) -> List[TestCase]:
        """Generate officer and director queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=15)

        officer_queries = [
            {
                "nl_template": "Who is the CEO of {company}?",
                "fields": ["TR.CEOName"],
                "subcategory": "ceo"
            },
            {
                "nl_template": "Get {company}'s executive team",
                "fields": ["TR.OfficerName", "TR.OfficerTitle", "TR.OfficerAge"],
                "parameters": {"RNK": "R1:R10"},
                "subcategory": "exec_team"
            },
            {
                "nl_template": "What is {company} CEO's compensation?",
                "fields": ["TR.CEOName", "TR.ODOfficerSalary", "TR.ODOfficerBonus", "TR.ODOfficerStockAwards", "TR.ODOfficerTotalComp"],
                "parameters": {"RNK": "R1"},
                "subcategory": "ceo_comp"
            },
            {
                "nl_template": "List {company}'s board of directors",
                "fields": ["TR.ODDirectorName", "TR.ODDirectorTenure", "TR.ODIndependentDirector"],
                "parameters": {"ODRnk": "R1:R15"},
                "subcategory": "board"
            },
            {
                "nl_template": "Get complete profile for {company}'s CEO",
                "fields": [
                    "TR.CEOName",
                    "TR.OfficerTitle",
                    "TR.OfficerTitleSince",
                    "TR.OfficerAge",
                    "TR.ODOfficerBiography",
                    "TR.ODOfficerUniversityName",
                    "TR.ODOfficerGraduationDegree",
                    "TR.ODOfficerSalary",
                    "TR.ODOfficerBonus",
                    "TR.ODOfficerStockAwards",
                    "TR.ODOfficerTotalComp"
                ],
                "parameters": {"RNK": "R1"},
                "subcategory": "ceo_profile"
            }
        ]

        for ticker_info in tickers:
            ticker = ticker_info.get("ric", ticker_info.get("symbol", ""))
            company = ticker_info.get("name", "Company")

            for query in officer_queries:
                nl_query = query["nl_template"].format(company=company)

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "instruments": [ticker],
                        "fields": query["fields"]
                    }
                }

                if "parameters" in query:
                    tool_call["arguments"]["parameters"] = query["parameters"]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory=f"officers_{query['subcategory']}",
                    complexity=5,
                    tags=["complex", "officers", query["subcategory"]],
                    metadata={
                        "ticker": ticker,
                        "query_type": query["subcategory"]
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_financial_health_checks(self) -> List[TestCase]:
        """Generate comprehensive financial health check queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=15)

        for ticker_info in tickers:
            ticker = ticker_info.get("symbol", "")
            company = ticker_info.get("name", "Company")

            nl_query = f"Full financial health check for {company}"

            tool_call = {
                "tool_name": "get_data",
                "arguments": {
                    "tickers": ticker,
                    "fields": [
                        # Profitability
                        "WC01001", "WC01751", "WC08301", "WC08326", "WC08366",
                        # Liquidity
                        "WC08106", "WC08101", "WC03151",
                        # Solvency
                        "WC08224", "WC08221", "WC18199",
                        # Efficiency
                        "WC08131", "WC08126",
                        # Cash Flow
                        "WC04860", "WC04601"
                    ],
                    "start": "0D",
                    "end": "0D",
                    "kind": 0
                }
            }

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=[tool_call],
                category=self.category,
                subcategory="health_check",
                complexity=8,
                tags=["complex", "analysis", "health_check"],
                metadata={
                    "ticker": ticker
                }
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_peer_analysis(self) -> List[TestCase]:
        """Generate peer analysis queries."""
        test_cases = []

        peer_groups = [
            {
                "name": "semiconductor companies",
                "tickers": ["U:NVDA", "U:AMD", "U:INTC", "U:QCOM", "U:AVGO"]
            },
            {
                "name": "big tech",
                "tickers": ["@AAPL", "U:MSFT", "U:GOOGL", "U:AMZN", "U:META"]
            },
            {
                "name": "major US banks",
                "tickers": ["U:JPM", "U:BAC", "U:WFC", "U:C", "U:GS"]
            }
        ]

        for group in peer_groups:
            nl_query = f"Perform peer analysis: get all key metrics for {group['name']}"

            tool_call = {
                "tool_name": "get_data",
                "arguments": {
                    "tickers": ",".join(group["tickers"]),
                    "fields": [
                        "WC01001", "WC01751", "WC08316", "WC08301", "WC08224",
                        "WC02001", "WC03255", "WC08001"
                    ],
                    "start": "0D",
                    "end": "0D",
                    "kind": 0
                }
            }

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=[tool_call],
                category=self.category,
                subcategory="peer_analysis",
                complexity=9,
                tags=["complex", "analysis", "peer_comparison"],
                metadata={
                    "peer_group": group["name"],
                    "tickers": group["tickers"]
                }
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def generate(self) -> List[TestCase]:
        """Generate all complex test cases."""
        test_cases = []

        # Phase 1: Cross-API Workflow Templates (~800 tests)
        test_cases.extend(self._generate_index_cascades())
        test_cases.extend(self._generate_screen_then_analyze())
        test_cases.extend(self._generate_company_vs_industry())
        test_cases.extend(self._generate_historical_forward_combinations())

        # Phase 2: Financial Analysis Workflows (~600 tests)
        test_cases.extend(self._generate_dcf_workflows())
        test_cases.extend(self._generate_comparable_analysis())
        test_cases.extend(self._generate_credit_analysis())
        test_cases.extend(self._generate_portfolio_analytics())
        test_cases.extend(self._generate_event_driven_analysis())

        # Phase 3: Multi-Step Calculation Chains (~400 tests)
        test_cases.extend(self._generate_derived_calculations())
        test_cases.extend(self._generate_ranking_operations())
        test_cases.extend(self._generate_time_series_analysis())

        # Phase 4: Edge Cases (~200 tests)
        test_cases.extend(self._generate_edge_cases())

        # Original methods (backward compatibility)
        test_cases.extend(self._generate_multi_step_workflows())
        test_cases.extend(self._generate_dupont_analysis())
        test_cases.extend(self._generate_valuation_model_inputs())
        test_cases.extend(self._generate_analyst_estimate_packages())
        test_cases.extend(self._generate_officer_director_queries())
        test_cases.extend(self._generate_financial_health_checks())
        test_cases.extend(self._generate_peer_analysis())

        return test_cases
