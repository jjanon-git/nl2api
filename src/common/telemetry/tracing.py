"""
Tracing Utilities.

Provides decorators and helpers for distributed tracing.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from src.common.telemetry.setup import get_tracer, is_telemetry_enabled

logger = logging.getLogger(__name__)

# Try to import trace status for error recording
_status_available = False
try:
    from opentelemetry.trace import Status, StatusCode

    _status_available = True
except ImportError:
    pass

F = TypeVar("F", bound=Callable[..., Any])


def trace_async(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing async functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to span

    Example:
        @trace_async("process_query")
        async def process(query: str) -> Result:
            ...
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_telemetry_enabled():
                return await func(*args, **kwargs)

            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)

                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    record_exception(e, span)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_sync(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    Decorator for tracing sync functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Static attributes to add to span

    Example:
        @trace_sync("parse_response")
        def parse(response: str) -> dict:
            ...
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_telemetry_enabled():
                return func(*args, **kwargs)

            tracer = get_tracer()
            with tracer.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    record_exception(e, span)
                    raise

        return wrapper  # type: ignore

    return decorator


def add_span_attributes(attributes: dict[str, Any]) -> None:
    """
    Add attributes to the current span.

    Args:
        attributes: Key-value pairs to add

    Example:
        add_span_attributes({
            "query": query,
            "domain": domain,
            "confidence": confidence,
        })
    """
    if not is_telemetry_enabled():
        return

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attributes(attributes)
    except Exception as e:
        logger.debug(f"Failed to add span attributes: {e}")


def record_exception(exception: Exception, span: Any = None) -> None:
    """
    Record an exception on the current or specified span.

    Args:
        exception: The exception to record
        span: Optional span (uses current span if not provided)
    """
    if not is_telemetry_enabled():
        return

    try:
        if span is None:
            from opentelemetry import trace

            span = trace.get_current_span()

        if span and span.is_recording():
            span.record_exception(exception)

            if _status_available:
                span.set_status(Status(StatusCode.ERROR, str(exception)))

    except Exception as e:
        logger.debug(f"Failed to record exception: {e}")


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """
    Add an event to the current span.

    Args:
        name: Event name
        attributes: Optional event attributes

    Example:
        add_span_event("cache_hit", {"key": cache_key})
    """
    if not is_telemetry_enabled():
        return

    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes or {})
    except Exception as e:
        logger.debug(f"Failed to add span event: {e}")
