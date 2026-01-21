"""
Conversation Management for Multi-turn Support

This module provides conversation tracking and context management
for multi-turn NL2API interactions.
"""

from src.nl2api.conversation.manager import ConversationManager
from src.nl2api.conversation.models import (
    ConversationSession,
    ConversationTurn,
    ConversationContext,
)
from src.nl2api.conversation.expander import QueryExpander

__all__ = [
    "ConversationManager",
    "ConversationSession",
    "ConversationTurn",
    "ConversationContext",
    "QueryExpander",
]
