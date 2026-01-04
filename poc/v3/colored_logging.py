# Copyright 2025 Allard Peper (Dragon Ace / DragonAceNL)
# Licensed under the Apache License, Version 2.0 (see LICENSE).
# This file is part of MeshPrep — https://github.com/DragonAceNL/MeshPrep

"""
Colored logging formatter for terminal output.

Provides ANSI color codes for different log levels to make
terminal output more readable during batch processing.
"""

import logging


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels in terminal output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        """Initialize the formatter.
        
        Args:
            fmt: Log message format string
            datefmt: Date format string
            use_colors: If True, add ANSI color codes to output
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        """Format a log record with optional color codes.
        
        Args:
            record: LogRecord to format
            
        Returns:
            Formatted log message string
        """
        # Save original levelname
        original_levelname = record.levelname
        
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            # Color the entire line for ERROR and WARNING, just levelname for others
            if record.levelname in ('ERROR', 'CRITICAL'):
                record.levelname = f"{self.BOLD}{color}{record.levelname}{self.RESET}"
                record.msg = f"{self.BOLD}{color}{record.msg}{self.RESET}"
            elif record.levelname == 'WARNING':
                record.levelname = f"{color}{record.levelname}{self.RESET}"
                record.msg = f"{color}{record.msg}{self.RESET}"
            else:
                record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        result = super().format(record)
        
        # Restore original levelname for other handlers (like file handler)
        record.levelname = original_levelname
        
        return result


def setup_logging(log_dir, log_filename: str = None):
    """Set up logging with console and file handlers.
    
    Args:
        log_dir: Directory for log files
        log_filename: Optional custom log filename (defaults to timestamped name)
        
    Returns:
        Configured logger instance
    """
    import sys
    from datetime import datetime
    from pathlib import Path
    
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    if log_filename is None:
        log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        use_colors=True
    ))
    
    file_handler = logging.FileHandler(log_dir / log_filename)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True
    )
    
    return logging.getLogger(__name__)
