#!/bin/bash
# MeshPrep v5 - Setup Script
# Automates virtual environment creation and installation

echo ""
echo "============================================================"
echo "MeshPrep v5 - Automated Setup"
echo "============================================================"
echo ""

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "To activate: source venv/bin/activate"
        exit 0
    fi
    echo "Removing old venv..."
    rm -rf venv
fi

# Try Python 3.12 first
echo "Checking for Python 3.12..."
if command -v python3.12 &> /dev/null; then
    echo "Found Python 3.12! Creating venv..."
    python3.12 -m venv venv
elif command -v python3.11 &> /dev/null; then
    echo "Found Python 3.11! Creating venv..."
    python3.11 -m venv venv
else
    echo ""
    echo "ERROR: Python 3.11 or 3.12 not found!"
    echo ""
    echo "MeshPrep requires Python 3.11 or 3.12 (Open3D limitation)"
    echo ""
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo "  macOS: brew install python@3.12"
    echo ""
    exit 1
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

echo ""
echo "Installing MeshPrep with all dependencies..."
echo "This may take 5-10 minutes (PyTorch is large)..."
pip install -e ".[all]"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "SUCCESS! MeshPrep v5 installed!"
    echo "============================================================"
    echo ""
    echo "To activate: source venv/bin/activate"
    echo "To test: python test_runner_simple.py"
    echo "To use: meshprep repair model.stl"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "ERROR: Installation failed!"
    echo "============================================================"
    echo ""
    echo "See INSTALL.md for troubleshooting steps."
    echo ""
    exit 1
fi
