#!/bin/bash

# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Starting Morphometric Statistics Pipeline..."

# Check that the virtual environment exists
if [ ! -f "$DIR/env/bin/activate" ]; then
    echo ""
    echo "Error: Python virtual environment not found."
    echo ""
    echo "Create it first with:"
    echo "  python3 -m venv env"
    echo "  source env/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

source "$DIR/env/bin/activate"

echo "Launching Streamlit application..."
python -m streamlit run app.py
