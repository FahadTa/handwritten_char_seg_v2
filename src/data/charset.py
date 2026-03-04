# =============================================================================
# charset.py
# =============================================================================
# Single source of truth for character-to-class-index mapping.
#
# Design decisions:
#   - Index 0 is always BACKGROUND (no character).
#   - Indices 1-26:  uppercase A-Z
#   - Indices 27-52: lowercase a-z
#   - Indices 53-62: digits 0-9
#   - Indices 63-79: punctuation and special characters
#   - Total: 80 classes (1 background + 79 character classes)
#
# Every module in the project imports from here. No other file should
# hardcode character mappings.
# =============================================================================

from typing import Dict, List, Optional, Tuple


# Background class label
BACKGROUND_INDEX: int = 0
BACKGROUND_LABEL: str = "<BG>"

# Ordered character groups
UPPERCASE: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE: str = "abcdefghijklmnopqrstuvwxyz"
DIGITS: str = "0123456789"
SPECIAL: str = " .,;:!?'-\"()&/@#+"

# Full character vocabulary (excluding background)
CHARSET: str = UPPERCASE + LOWERCASE + DIGITS + SPECIAL

# Total number of classes including background
NUM_CLASSES: int = len(CHARSET) + 1  # +1 for background


def _build_char_to_index() -> Dict[str, int]:
    """Build mapping from character to class index.

    Returns:
        Dictionary mapping each character to its integer class index.
        Index 0 is reserved for background and is not included here.
    """
    mapping = {}
    for i, char in enumerate(CHARSET):
        mapping[char] = i + 1  # offset by 1 because 0 = background
    return mapping


def _build_index_to_char() -> Dict[int, str]:
    """Build mapping from class index to character.

    Returns:
        Dictionary mapping each integer class index to its character.
        Includes index 0 mapped to the background label.
    """
    mapping = {BACKGROUND_INDEX: BACKGROUND_LABEL}
    for i, char in enumerate(CHARSET):
        mapping[i + 1] = char
    return mapping


# Module-level lookup tables (built once at import time)
CHAR_TO_INDEX: Dict[str, int] = _build_char_to_index()
INDEX_TO_CHAR: Dict[int, str] = _build_index_to_char()


def char_to_index(char: str) -> int:
    """Convert a single character to its class index.

    Args:
        char: A single character string.

    Returns:
        Integer class index. Returns BACKGROUND_INDEX if the character
        is not in the vocabulary.
    """
    return CHAR_TO_INDEX.get(char, BACKGROUND_INDEX)


def index_to_char(index: int) -> str:
    """Convert a class index to its character.

    Args:
        index: Integer class index.

    Returns:
        The corresponding character string, or the background label
        if the index is out of range.
    """
    return INDEX_TO_CHAR.get(index, BACKGROUND_LABEL)


def encode_string(text: str) -> List[int]:
    """Convert a string to a list of class indices.

    Characters not in the vocabulary are silently mapped to BACKGROUND_INDEX.

    Args:
        text: Input string.

    Returns:
        List of integer class indices.
    """
    return [char_to_index(c) for c in text]


def decode_indices(indices: List[int]) -> str:
    """Convert a list of class indices back to a string.

    Background indices are skipped in the output.

    Args:
        indices: List of integer class indices.

    Returns:
        Decoded string with background tokens removed.
    """
    chars = []
    for idx in indices:
        label = index_to_char(idx)
        if label != BACKGROUND_LABEL:
            chars.append(label)
    return "".join(chars)


def get_class_names() -> List[str]:
    """Get ordered list of class names for display and logging.

    Returns:
        List of length NUM_CLASSES where element i is the display name
        for class index i.
    """
    names = [BACKGROUND_LABEL]
    for char in CHARSET:
        if char == " ":
            names.append("<SP>")
        elif char == "\t":
            names.append("<TAB>")
        else:
            names.append(char)
    return names


def get_group_indices() -> Dict[str, Tuple[int, int]]:
    """Get index ranges for each character group.

    Useful for per-group metric computation (e.g., uppercase accuracy
    vs. digit accuracy in evaluation).

    Returns:
        Dictionary mapping group name to (start_index, end_index) inclusive.
    """
    offset = 1  # background is 0
    groups = {}

    groups["uppercase"] = (offset, offset + len(UPPERCASE) - 1)
    offset += len(UPPERCASE)

    groups["lowercase"] = (offset, offset + len(LOWERCASE) - 1)
    offset += len(LOWERCASE)

    groups["digits"] = (offset, offset + len(DIGITS) - 1)
    offset += len(DIGITS)

    groups["special"] = (offset, offset + len(SPECIAL) - 1)

    return groups


def validate_charset() -> None:
    """Run sanity checks on the charset configuration.

    Raises:
        AssertionError: If any invariant is violated.
    """
    assert NUM_CLASSES == 80, (
        f"Expected 80 classes, got {NUM_CLASSES}. "
        f"CHARSET has {len(CHARSET)} characters + 1 background."
    )

    # Verify no duplicate characters
    assert len(CHARSET) == len(set(CHARSET)), (
        "Duplicate characters found in CHARSET."
    )

    # Verify bijectivity of mappings
    assert len(CHAR_TO_INDEX) == len(CHARSET), (
        "CHAR_TO_INDEX size mismatch."
    )
    assert len(INDEX_TO_CHAR) == NUM_CLASSES, (
        "INDEX_TO_CHAR size mismatch."
    )

    # Verify round-trip consistency
    for char in CHARSET:
        idx = char_to_index(char)
        recovered = index_to_char(idx)
        assert recovered == char or (char == " " and recovered == " "), (
            f"Round-trip failed for '{char}': index={idx}, recovered='{recovered}'"
        )

    # Verify background mapping
    assert char_to_index("\x00") == BACKGROUND_INDEX, (
        "Unknown characters should map to BACKGROUND_INDEX."
    )
    assert index_to_char(BACKGROUND_INDEX) == BACKGROUND_LABEL, (
        "Index 0 should map to BACKGROUND_LABEL."
    )


# Run validation at import time to catch configuration errors early
validate_charset()