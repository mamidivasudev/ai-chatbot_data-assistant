"""
Spoken commands in any language → one of a few allowed actions.

The browser's own matcher only knows English keywords. A command spoken in Hindi,
Telugu, Tamil, Sinhala… ("रफनेस वाले चार्ट का सारांश बताओ") names a chart whose
title is in English, so it is handed here: the model sees the command and the
numbered items on screen and picks an action and an item. The reply is checked
against that list, so the model can only choose something that is really there;
anything else becomes "none".
"""

import re
from typing import Optional

from summarizer import _parse_json_object

ACTIONS = ("summarize_all", "summarize", "read", "stop", "none")
MAX_TARGETS = 60
MAX_COMMAND_CHARS = 300

VOICE_SYSTEM = """You understand one spoken command for a road and bridge dashboard app. The command may be in {language}, in English, or a mix, and speech recognition may have misheard some words.

Pick exactly one action:
- "summarize_all": summarize the whole dashboard, page or screen.
- "summarize": summarize one item from the numbered list. The user may say the item's name translated, transliterated, shortened or misheard; choose the item it most plausibly means.
- "read": read the current summary aloud again.
- "stop": stop speaking.
- "none": anything else, or when you are not reasonably sure.

Reply with a JSON object only:
{"action": "...", "target": null, "english": "..."}
"target" is the item number for "summarize", otherwise null. "english" is the command translated to English."""


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


async def resolve_intent(text: str, targets: list[str], provider, language: str = "English",
                         can_summarize_all: bool = True) -> dict:
    """{"action", "target_index" (0-based or None), "target", "english"}; never an unlisted item.

    Raises LLMError when the model cannot be reached.
    """
    command = _clean(text)[:MAX_COMMAND_CHARS]
    items = [_clean(t) for t in targets][:MAX_TARGETS]
    none = {"action": "none", "target_index": None, "target": None, "english": ""}
    if not command:
        return none

    listing = "\n".join(f"{i}. {label}" for i, label in enumerate(items, 1)) or "(nothing on screen)"
    prompt = f"Command: {command}\n\nItems on screen:\n{listing}"
    if not can_summarize_all:
        prompt += "\n\nThis screen has no dashboard, so \"summarize_all\" is not available."

    reply = _parse_json_object(await provider.complete(
        VOICE_SYSTEM.replace("{language}", language), prompt, json_mode=True, temperature=0.0, timeout=60.0))
    if not reply:
        return none

    action = str(reply.get("action") or "").strip().lower()
    english = _clean(str(reply.get("english") or ""))[:MAX_COMMAND_CHARS]
    if action not in ACTIONS or (action == "summarize_all" and not can_summarize_all):
        return {**none, "english": english}

    index: Optional[int] = None
    if action == "summarize":
        try:
            number = int(reply.get("target"))
        except (TypeError, ValueError):
            number = 0
        if not 1 <= number <= len(items):
            return {**none, "english": english}
        index = number - 1

    return {"action": action, "target_index": index, "target": items[index] if index is not None else None,
            "english": english}
