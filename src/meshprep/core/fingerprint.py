# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Model fingerprinting for filter script discovery and sharing.

The fingerprint system enables community sharing of filter scripts by providing
a unique, searchable identifier for each model file. When you open a model in
MeshPrep, it computes and displays the fingerprint, which you can then search
on Reddit, Google, or other platforms to find filter scripts others have shared.

Key Design Decisions:
- Fingerprint is computed from the ORIGINAL FILE BYTES (not loaded mesh data)
- CTM files are fingerprinted as CTM (compressed), not as decompressed geometry
- This ensures exact matching: same download = same fingerprint
- Format: MP:xxxxxxxxxxxx (12 hex characters from SHA256)

Example workflow:
1. Download "spaceship.ctm" from CGTrader
2. Open in MeshPrep, see fingerprint: MP:a3f8c2d1e5b7
3. Search Reddit for "MP:a3f8c2d1e5b7"
4. Find a community filter script that works for this exact model
5. Import and apply the filter script

When sharing filter scripts:
- Post title: "Filter script for MP:a3f8c2d1e5b7 (spaceship.ctm) - fixes holes and normals"
- Include the fingerprint in the filter script JSON
- Others can search and find your solution
"""

import hashlib
from pathlib import Path
from typing import Union


# Fingerprint prefix for MeshPrep
FINGERPRINT_PREFIX = "MP"

# Number of hex characters to use from SHA256 (12 = 48 bits = 281 trillion combinations)
FINGERPRINT_LENGTH = 12

# MeshPrep GitHub URL - included in filter scripts for discoverability
MESHPREP_URL = "https://github.com/DragonAceNL/MeshPrep"


def compute_file_fingerprint(path: Union[str, Path]) -> str:
    """
    Compute a searchable fingerprint for a model file.
    
    The fingerprint is computed from the raw file bytes, ensuring that:
    - CTM files are fingerprinted as CTM (not decompressed mesh)
    - Same file download = same fingerprint
    - Fingerprint can be searched on Reddit/Google to find filter scripts
    
    Args:
        path: Path to the model file (STL, CTM, OBJ, etc.)
    
    Returns:
        Fingerprint string in format "MP:xxxxxxxxxxxx" (12 hex chars)
    
    Example:
        >>> fingerprint = compute_file_fingerprint("model.ctm")
        >>> print(fingerprint)
        MP:a3f8c2d1e5b7
        >>> # Search "MP:a3f8c2d1e5b7" on Reddit to find filter scripts
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Compute SHA256 of file contents
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    # Take first N hex characters
    hex_digest = sha256_hash.hexdigest()[:FINGERPRINT_LENGTH]
    
    return f"{FINGERPRINT_PREFIX}:{hex_digest}"


def compute_full_file_hash(path: Union[str, Path]) -> str:
    """
    Compute the full SHA256 hash of a file.
    
    This is useful for exact matching in databases or for verification.
    
    Args:
        path: Path to the file
    
    Returns:
        Full SHA256 hex digest (64 characters)
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def is_valid_fingerprint(fingerprint: str) -> bool:
    """
    Check if a string is a valid MeshPrep fingerprint.
    
    Args:
        fingerprint: String to validate
    
    Returns:
        True if valid fingerprint format
    """
    if not fingerprint:
        return False
    
    parts = fingerprint.split(":")
    if len(parts) != 2:
        return False
    
    prefix, hex_part = parts
    
    if prefix != FINGERPRINT_PREFIX:
        return False
    
    if len(hex_part) != FINGERPRINT_LENGTH:
        return False
    
    # Check if hex_part is valid hexadecimal
    try:
        int(hex_part, 16)
        return True
    except ValueError:
        return False


def format_fingerprint_for_search(fingerprint: str) -> str:
    """
    Format a fingerprint for searching on platforms like Reddit.
    
    Returns search-friendly formats for different platforms.
    
    Args:
        fingerprint: The fingerprint string (e.g., "MP:a3f8c2d1e5b7")
    
    Returns:
        Search query string
    """
    return fingerprint  # The MP:xxx format is already search-friendly


def format_fingerprint_for_reddit(fingerprint: str, filename: str = "", description: str = "") -> str:
    """
    Format a fingerprint for posting on Reddit.
    
    Creates a formatted string suitable for Reddit post titles or comments.
    
    Args:
        fingerprint: The fingerprint string
        filename: Optional original filename
        description: Optional description of the filter script
    
    Returns:
        Reddit-formatted string
    
    Example:
        >>> format_fingerprint_for_reddit("MP:a3f8c2d1e5b7", "spaceship.ctm", "fixes holes")
        '[MeshPrep Filter] MP:a3f8c2d1e5b7 (spaceship.ctm) - fixes holes'
    """
    parts = ["[MeshPrep Filter]", fingerprint]
    
    if filename:
        parts.append(f"({filename})")
    
    if description:
        parts.append(f"- {description}")
    
    return " ".join(parts)


def get_fingerprint_search_urls(fingerprint: str) -> dict[str, str]:
    """
    Get URLs to search for a fingerprint on various platforms.
    
    Args:
        fingerprint: The fingerprint string
    
    Returns:
        Dictionary of platform name -> search URL
    """
    from urllib.parse import quote_plus
    
    encoded = quote_plus(fingerprint)
    
    return {
        "google": f"https://www.google.com/search?q={encoded}",
        "reddit": f"https://www.reddit.com/search/?q={encoded}",
        "github": f"https://github.com/search?q={encoded}&type=code",
    }
