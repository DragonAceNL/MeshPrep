# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Model fingerprinting for filter script discovery and sharing.

The fingerprint is computed from the ORIGINAL FILE BYTES, so:
- CTM files are fingerprinted as CTM (compressed bytes)
- Same download = same fingerprint
- Format: MP:xxxxxxxxxxxx (12 hex chars from SHA256)

Search "MP:xxxxxxxxxxxx" on Reddit/Google to find filter scripts.
"""

import hashlib
from pathlib import Path
from typing import Union


FINGERPRINT_PREFIX = "MP"
FINGERPRINT_LENGTH = 12
MESHPREP_URL = "https://github.com/DragonAceNL/MeshPrep"


def compute_file_fingerprint(path: Union[str, Path]) -> str:
    """
    Compute a searchable fingerprint for a model file.
    
    Args:
        path: Path to the model file (STL, CTM, OBJ, etc.)
    
    Returns:
        Fingerprint string in format "MP:xxxxxxxxxxxx"
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    hex_digest = sha256_hash.hexdigest()[:FINGERPRINT_LENGTH]
    return f"{FINGERPRINT_PREFIX}:{hex_digest}"


def compute_full_file_hash(path: Union[str, Path]) -> str:
    """Compute full SHA256 hash of a file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def is_valid_fingerprint(fingerprint: str) -> bool:
    """Check if a string is a valid MeshPrep fingerprint."""
    if not fingerprint:
        return False
    
    parts = fingerprint.split(":")
    if len(parts) != 2:
        return False
    
    prefix, hex_part = parts
    if prefix != FINGERPRINT_PREFIX or len(hex_part) != FINGERPRINT_LENGTH:
        return False
    
    try:
        int(hex_part, 16)
        return True
    except ValueError:
        return False
