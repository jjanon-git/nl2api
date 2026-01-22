"""
Continuous Evaluation Module

Provides scheduled batch evaluations with regression detection and alerting.

Components:
- Scheduler: Cron-based evaluation scheduling
- RegressionDetector: Statistical regression detection
- AlertHandler: Multi-channel alerting (webhook, email)
"""

from src.evaluation.continuous.alerts import AlertHandler, RegressionAlert
from src.evaluation.continuous.config import ScheduleConfig
from src.evaluation.continuous.regression import RegressionDetector, RegressionResult
from src.evaluation.continuous.scheduler import EvalScheduler

__all__ = [
    "ScheduleConfig",
    "EvalScheduler",
    "RegressionDetector",
    "RegressionResult",
    "AlertHandler",
    "RegressionAlert",
]
