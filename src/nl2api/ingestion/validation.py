"""
Entity Validation

Validates entity records before import to ensure data quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Valid ISO 3166-1 alpha-2 country codes (subset of most common)
VALID_COUNTRY_CODES = {
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT", "AU", "AW", "AX", "AZ",
    "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS",
    "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE",
    "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF",
    "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
    "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT", "JE", "JM",
    "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC",
    "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
    "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA",
    "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG",
    "PH", "PK", "PL", "PM", "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS",
    "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO",
    "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW",
}

# LEI format: 20 alphanumeric characters
LEI_PATTERN = re.compile(r"^[A-Z0-9]{20}$")

# CIK format: up to 10 digits
CIK_PATTERN = re.compile(r"^\d{1,10}$")

# FIGI format: 12 alphanumeric starting with BBG
FIGI_PATTERN = re.compile(r"^BBG[A-Z0-9]{9}$")

# Ticker format: 1-10 alphanumeric, may include dots and hyphens
TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


@dataclass
class ValidationResult:
    """Result of entity validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


class EntityValidator:
    """Validates entity records before import."""

    def __init__(
        self,
        strict_mode: bool = False,
        min_name_length: int = 2,
        max_name_length: int = 500,
    ):
        """
        Initialize validator.

        Args:
            strict_mode: If True, warnings become errors
            min_name_length: Minimum length for primary_name
            max_name_length: Maximum length for primary_name
        """
        self.strict_mode = strict_mode
        self.min_name_length = min_name_length
        self.max_name_length = max_name_length

    def validate(self, record: dict) -> ValidationResult:
        """
        Validate an entity record.

        Args:
            record: Dictionary with entity fields

        Returns:
            ValidationResult with is_valid, errors, and warnings
        """
        result = ValidationResult(is_valid=True)

        # Required fields
        self._validate_required_fields(record, result)

        # Format validation
        self._validate_lei(record.get("lei"), result)
        self._validate_cik(record.get("cik"), result)
        self._validate_figi(record.get("figi"), result)
        self._validate_ticker(record.get("ticker"), result)
        self._validate_country_code(record.get("country_code"), result)
        self._validate_name(record.get("primary_name"), result)

        # In strict mode, warnings become errors
        if self.strict_mode and result.warnings:
            result.errors.extend(result.warnings)
            result.is_valid = False

        return result

    def _validate_required_fields(self, record: dict, result: ValidationResult) -> None:
        """Check required fields are present."""
        if not record.get("primary_name"):
            result.add_error("Missing required field: primary_name")

        if not record.get("data_source"):
            result.add_error("Missing required field: data_source")

    def _validate_lei(self, lei: str | None, result: ValidationResult) -> None:
        """Validate LEI format (20 alphanumeric chars)."""
        if not lei:
            return

        lei = lei.upper().strip()
        if not LEI_PATTERN.match(lei):
            result.add_error(f"Invalid LEI format: {lei} (expected 20 alphanumeric chars)")

    def _validate_cik(self, cik: str | None, result: ValidationResult) -> None:
        """Validate CIK format (up to 10 digits)."""
        if not cik:
            return

        cik = cik.strip()
        if not CIK_PATTERN.match(cik):
            result.add_error(f"Invalid CIK format: {cik} (expected up to 10 digits)")

    def _validate_figi(self, figi: str | None, result: ValidationResult) -> None:
        """Validate FIGI format (12 chars starting with BBG)."""
        if not figi:
            return

        figi = figi.upper().strip()
        if not FIGI_PATTERN.match(figi):
            result.add_warning(f"Invalid FIGI format: {figi} (expected BBG + 9 alphanumeric)")

    def _validate_ticker(self, ticker: str | None, result: ValidationResult) -> None:
        """Validate ticker format."""
        if not ticker:
            return

        ticker = ticker.upper().strip()
        if not TICKER_PATTERN.match(ticker):
            result.add_warning(f"Unusual ticker format: {ticker}")

    def _validate_country_code(self, country_code: str | None, result: ValidationResult) -> None:
        """Validate ISO 3166-1 alpha-2 country code."""
        if not country_code:
            return

        country_code = country_code.upper().strip()
        if len(country_code) != 2:
            result.add_warning(f"Invalid country code length: {country_code}")
        elif country_code not in VALID_COUNTRY_CODES:
            result.add_warning(f"Unknown country code: {country_code}")

    def _validate_name(self, name: str | None, result: ValidationResult) -> None:
        """Validate entity name."""
        if not name:
            return

        name = name.strip()
        if len(name) < self.min_name_length:
            result.add_error(f"Name too short: '{name}' (min {self.min_name_length} chars)")
        elif len(name) > self.max_name_length:
            result.add_error(f"Name too long: {len(name)} chars (max {self.max_name_length})")

        # Check for placeholder/test names
        placeholder_patterns = [
            r"^test\s*$",
            r"^xxx+$",
            r"^n/?a$",
            r"^unknown$",
            r"^\?+$",
            r"^-+$",
        ]
        name_lower = name.lower()
        for pattern in placeholder_patterns:
            if re.match(pattern, name_lower):
                result.add_warning(f"Name appears to be placeholder: '{name}'")
                break


class IngestionErrorHandler:
    """Handles errors during ingestion without killing the job."""

    def __init__(
        self,
        max_errors: int = 1000,
        error_log_path: str | None = None,
    ):
        """
        Initialize error handler.

        Args:
            max_errors: Maximum errors before aborting
            error_log_path: Path to error log file (optional)
        """
        self.max_errors = max_errors
        self.error_count = 0
        self.error_log = None
        if error_log_path:
            self.error_log = open(error_log_path, "w")

    def handle_error(self, record_id: str | None, error: Exception | str) -> bool:
        """
        Handle a record error.

        Args:
            record_id: Identifier of the failed record
            error: Exception or error message

        Returns:
            True to continue processing, False to abort

        Raises:
            IngestionAbortError: If max_errors exceeded
        """
        self.error_count += 1

        if self.error_log:
            import json
            self.error_log.write(
                json.dumps({
                    "record_id": record_id,
                    "error": str(error),
                    "error_number": self.error_count,
                })
                + "\n"
            )
            self.error_log.flush()

        if self.error_count >= self.max_errors:
            raise IngestionAbortError(
                f"Too many errors: {self.error_count} (max: {self.max_errors})"
            )

        return True

    def close(self) -> None:
        """Close error log file."""
        if self.error_log:
            self.error_log.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class IngestionAbortError(Exception):
    """Raised when ingestion must abort due to too many errors."""

    pass
