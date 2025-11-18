#!/bin/bash
# Simple test coverage checker
# Usage: ./check_coverage.sh [--html]

set -e

echo "=== PAR2-RS Test Coverage Report ==="
echo ""

# Run coverage analysis
if [ "$1" = "--html" ]; then
    echo "Generating HTML report..."
    cargo tarpaulin --out Html --output-dir target/coverage --exclude-files 'tmp/*' --exclude-files 'benches/*' --exclude-files 'src/bin/*'
    echo ""
    echo "HTML report: target/coverage/index.html"
    echo "Opening in browser..."
    open target/coverage/index.html 2>/dev/null || xdg-open target/coverage/index.html 2>/dev/null || echo "(Open manually: target/coverage/index.html)"
else
    echo "Running coverage analysis..."
    cargo tarpaulin --out Stdout --exclude-files 'tmp/*' --exclude-files 'benches/*' --exclude-files 'src/bin/*' 2>&1 | tail -20
fi

echo ""
echo "=== Coverage Targets ==="
echo "✓ Excellent:  95-100%"
echo "✓ Good:       75-94%"
echo "⚠ Fair:       50-74%"
echo "✗ Poor:       <50%"
echo ""
echo "Run with --html for detailed line-by-line coverage report"
