"""
MCP Tool Definitions and Handlers for Entity Resolution

Defines the tools exposed by the Entity Resolution MCP Server:
- resolve_entity: Resolve a single entity to RIC
- resolve_entities_batch: Batch resolve multiple entities
- extract_and_resolve: Extract entities from query and resolve them
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from src.evalkit.common.telemetry import get_tracer

if TYPE_CHECKING:
    from src.nl2api.resolution.resolver import ExternalEntityResolver

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


# =============================================================================
# Tool Definitions (MCP Schema)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "resolve_entity",
        "description": (
            "Resolve a company name, ticker symbol, or identifier to a Reuters RIC "
            "(Reuters Instrument Code). Supports company names (e.g., 'Apple Inc.'), "
            "tickers (e.g., 'AAPL'), LEIs, and CIKs. Returns the primary RIC and "
            "confidence score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": (
                        "The entity to resolve. Can be a company name, ticker, LEI, "
                        "or other identifier. Examples: 'Apple', 'AAPL', 'Microsoft Corporation'"
                    ),
                },
                "entity_type": {
                    "type": "string",
                    "enum": ["company", "ticker", "ric", "lei", "cik"],
                    "description": (
                        "Optional hint about the entity type. If not provided, "
                        "the resolver will attempt to detect the type automatically."
                    ),
                },
            },
            "required": ["entity"],
        },
    },
    {
        "name": "resolve_entities_batch",
        "description": (
            "Resolve multiple entities to RICs in a single batch operation. "
            "More efficient than calling resolve_entity multiple times. "
            "Returns results for all entities that could be resolved."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of entities to resolve. Each can be a company name, "
                        "ticker, or identifier."
                    ),
                    "minItems": 1,
                    "maxItems": 100,
                },
            },
            "required": ["entities"],
        },
    },
    {
        "name": "extract_and_resolve",
        "description": (
            "Extract entity mentions from a natural language query and resolve "
            "them to RICs. Useful for processing user queries that mention "
            "multiple companies. Returns both the extracted entities and their "
            "resolved RICs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query containing entity references. "
                        "Example: 'Compare Apple and Microsoft revenue growth'"
                    ),
                },
            },
            "required": ["query"],
        },
    },
]


# =============================================================================
# Response Models
# =============================================================================


@dataclass(frozen=True)
class ResolveEntityResult:
    """Response for resolve_entity tool."""

    found: bool
    original: str
    identifier: str | None = None
    confidence: float = 0.0
    entity_type: str = "unknown"
    alternatives: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["alternatives"] = list(self.alternatives)
        return result


@dataclass(frozen=True)
class BatchResolveResult:
    """Response for resolve_entities_batch tool."""

    results: tuple[ResolveEntityResult, ...]
    total_requested: int
    total_resolved: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_requested": self.total_requested,
            "total_resolved": self.total_resolved,
        }


@dataclass(frozen=True)
class ExtractAndResolveResult:
    """Response for extract_and_resolve tool."""

    extracted_entities: tuple[str, ...]
    resolved: dict[str, str]  # entity -> RIC
    unresolved: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "extracted_entities": list(self.extracted_entities),
            "resolved": self.resolved,
            "unresolved": list(self.unresolved),
        }


# =============================================================================
# Tool Handlers
# =============================================================================


class ToolHandlers:
    """
    Handlers for MCP tool invocations.

    Wraps the ExternalEntityResolver and provides MCP-compatible interfaces.
    """

    def __init__(self, resolver: ExternalEntityResolver):
        """
        Initialize tool handlers.

        Args:
            resolver: The entity resolver instance to use
        """
        self._resolver = resolver

    async def handle_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Route a tool call to the appropriate handler.

        Args:
            tool_name: Name of the tool to invoke
            arguments: Tool arguments

        Returns:
            Tool result as a dictionary

        Raises:
            ValueError: If tool name is unknown
        """
        with tracer.start_as_current_span(f"mcp.tool.{tool_name}") as span:
            span.set_attribute("mcp.tool.name", tool_name)
            span.set_attribute("mcp.tool.arguments", str(arguments)[:200])

            if tool_name == "resolve_entity":
                result = await self._resolve_entity(
                    entity=arguments["entity"],
                    entity_type=arguments.get("entity_type"),
                )
            elif tool_name == "resolve_entities_batch":
                result = await self._resolve_entities_batch(
                    entities=arguments["entities"],
                )
            elif tool_name == "extract_and_resolve":
                result = await self._extract_and_resolve(
                    query=arguments["query"],
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            span.set_attribute("mcp.tool.success", True)
            return result

    async def _resolve_entity(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Handle resolve_entity tool call.

        Args:
            entity: Entity to resolve
            entity_type: Optional type hint

        Returns:
            ResolveEntityResult as dictionary
        """
        with tracer.start_as_current_span("mcp.resolve_entity") as span:
            span.set_attribute("entity.name", entity[:50])
            if entity_type:
                span.set_attribute("entity.type_hint", entity_type)

            resolved = await self._resolver.resolve_single(entity, entity_type)

            if resolved:
                result = ResolveEntityResult(
                    found=True,
                    original=resolved.original,
                    identifier=resolved.identifier,
                    confidence=resolved.confidence,
                    entity_type=resolved.entity_type,
                    alternatives=resolved.alternatives,
                    metadata=resolved.metadata,
                )
                span.set_attribute("entity.found", True)
                span.set_attribute("entity.identifier", resolved.identifier)
                span.set_attribute("entity.confidence", resolved.confidence)
            else:
                result = ResolveEntityResult(
                    found=False,
                    original=entity,
                )
                span.set_attribute("entity.found", False)

            return result.to_dict()

    async def _resolve_entities_batch(
        self,
        entities: list[str],
    ) -> dict[str, Any]:
        """
        Handle resolve_entities_batch tool call.

        Args:
            entities: List of entities to resolve

        Returns:
            BatchResolveResult as dictionary
        """
        with tracer.start_as_current_span("mcp.resolve_entities_batch") as span:
            span.set_attribute("batch.size", len(entities))

            resolved_list = await self._resolver.resolve_batch(entities)

            # Build results - include both resolved and unresolved
            results = []
            resolved_set = {r.original.lower() for r in resolved_list}

            for resolved in resolved_list:
                results.append(
                    ResolveEntityResult(
                        found=True,
                        original=resolved.original,
                        identifier=resolved.identifier,
                        confidence=resolved.confidence,
                        entity_type=resolved.entity_type,
                        alternatives=resolved.alternatives,
                        metadata=resolved.metadata,
                    )
                )

            # Add unresolved entities
            for entity in entities:
                if entity.lower() not in resolved_set:
                    results.append(
                        ResolveEntityResult(
                            found=False,
                            original=entity,
                        )
                    )

            batch_result = BatchResolveResult(
                results=tuple(results),
                total_requested=len(entities),
                total_resolved=len(resolved_list),
            )

            span.set_attribute("batch.resolved", len(resolved_list))
            span.set_attribute(
                "batch.resolution_rate",
                len(resolved_list) / len(entities) if entities else 0,
            )

            return batch_result.to_dict()

    async def _extract_and_resolve(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Handle extract_and_resolve tool call.

        Args:
            query: Natural language query

        Returns:
            ExtractAndResolveResult as dictionary
        """
        with tracer.start_as_current_span("mcp.extract_and_resolve") as span:
            span.set_attribute("query.length", len(query))

            # Use the resolver's extract and resolve method
            resolved_map = await self._resolver.resolve(query)

            # Get the extracted entities (keys from resolved map)
            # Note: We need to also find unresolved entities
            # The resolver._extract_entities is private, but we can derive from the result
            extracted = list(resolved_map.keys())

            # For now, we only have the resolved entities from the resolver
            # The resolver internally handles extraction, so we return what was resolved
            result = ExtractAndResolveResult(
                extracted_entities=tuple(extracted),
                resolved=resolved_map,
                unresolved=(),  # The resolver only returns resolved entities
            )

            span.set_attribute("query.extracted_count", len(extracted))
            span.set_attribute("query.resolved_count", len(resolved_map))

            return result.to_dict()
