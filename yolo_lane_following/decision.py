"""Pure traffic-rule decision logic; no perception or motor side effects."""
from typing import Dict, Iterable, Optional


def choose_branch(available: Dict[str, bool], sign: Optional[str],
                  preferred: Iterable[str]) -> Optional[str]:
    legal = {name for name, exists in available.items() if exists}
    normalized = (sign or "").upper()
    forbidden = {"NO_LEFT": "left", "NO_RIGHT": "right", "NO_STRAIGHT": "straight"}
    mandatory = {"MUST_LEFT": "left", "MUST_RIGHT": "right", "MUST_STRAIGHT": "straight"}
    if normalized in forbidden:
        legal.discard(forbidden[normalized])
    elif normalized in mandatory:
        legal &= {mandatory[normalized]}
    # NO_ENTRY must first be associated with a branch by sign perception; an
    # unassociated instance is intentionally uncertain, not a guessed turn.
    elif normalized in ("NO_ENTRY", "UNKNOWN"):
        return None
    for direction in preferred:
        if direction in legal:
            return direction
    return None
