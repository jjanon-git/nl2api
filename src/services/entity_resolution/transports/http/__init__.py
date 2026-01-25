"""
HTTP transport for entity resolution service.
"""

from .app import create_app, run_http_server

__all__ = ["create_app", "run_http_server"]
