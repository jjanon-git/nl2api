"""
Continuous Evaluation Module

Provides scheduled batch evaluations with regression detection and alerting.

Components:
- Scheduler: Cron-based evaluation scheduling
- RegressionDetector: Statistical regression detection
- AlertHandler: Multi-channel alerting (webhook, email)
"""

from src.evalkit.continuous.alerts import AlertHandler, RegressionAlert
from src.evalkit.continuous.config import ScheduleConfig
from src.evalkit.continuous.regression import RegressionDetector, RegressionResult
from src.evalkit.continuous.scheduler import EvalScheduler

__all__ = [
    "ScheduleConfig",
    "EvalScheduler",
    "RegressionDetector",
    "RegressionResult",
    "AlertHandler",
    "RegressionAlert",
]
