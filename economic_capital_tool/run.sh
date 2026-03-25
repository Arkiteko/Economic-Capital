#!/bin/bash
# Economic Capital Calculator - Launch Script
# Usage: bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Economic Capital Calculator"
echo "  Monte Carlo Credit Portfolio Simulator"
echo "========================================"
echo ""

# Install dependencies
echo "Checking dependencies..."
pip install -q streamlit numpy pandas scipy openpyxl plotly numba 2>/dev/null || \
pip install -q --break-system-packages streamlit numpy pandas scipy openpyxl plotly numba 2>/dev/null

# Generate dummy data Excel file if it doesn't exist
if [ ! -f "../Bank_Portfolio_Dummy_Data.xlsx" ]; then
    echo "Generating dummy portfolio data..."
    python generate_excel_data.py
fi

echo ""
echo "Starting Streamlit application..."
echo "Open your browser to: http://localhost:8501"
echo ""

streamlit run app.py --server.port 8501 --server.headless true
