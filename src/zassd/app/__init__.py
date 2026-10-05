"""Application layer for ZASSD (thin wrapper for thesis demo).

This package sits ON TOP of the research core and adds no new
decoding science. It only packages existing engines
(vanilla / prompt_lookup / zassd / routed) into end-user tasks
suitable for a capstone thesis on Generative AI for intelligent
computing systems:

  - chat       : general assistant
  - summarize  : long-context summarization (PLD-friendly)
  - qa         : retrieval-augmented QA over user documents
  - code       : code explanation / generation
"""

from zassd.app.assistant import Conversation, IntelligentAssistant, AppResult
from zassd.app.store import SimpleDocStore

__all__ = ["IntelligentAssistant", "Conversation", "AppResult", "SimpleDocStore"]
