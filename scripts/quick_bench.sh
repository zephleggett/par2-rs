#!/bin/bash
# Quick repair benchmark - 100MB file for fast iteration
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd /tmp
rm -rf quick_par2_bench && mkdir quick_par2_bench && cd quick_par2_bench

echo "Creating 100MB test file..."
dd if=/dev/urandom of=test.bin bs=1M count=100 2>/dev/null

# Store original hash
ORIG_MD5=$(md5sum test.bin | cut -d' ' -f1)
echo "Original MD5: $ORIG_MD5"

# Download turbo if needed
if [ ! -f "$HOME/par2cmdline-turbo/par2" ]; then
    echo "Downloading par2cmdline-turbo..."
    mkdir -p "$HOME/par2cmdline-turbo"
    curl -sL https://github.com/animetosho/par2cmdline-turbo/releases/download/v1.2.0/par2cmdline-turbo-v1.2.0-linux-amd64.tar.xz | tar -xJ -C "$HOME/par2cmdline-turbo" --strip-components=1
fi

# Create PAR2 with turbo (10% redundancy)
echo "Creating PAR2 files..."
"$HOME/par2cmdline-turbo/par2" create -r10 -n1 test.par2 test.bin >/dev/null 2>&1

# Setup test directories with CLEAN copies first
echo "Setting up test directories..."
mkdir turbo_test rs_test
cp test.par2 test.vol*.par2 turbo_test/
cp test.par2 test.vol*.par2 rs_test/
cp test.bin turbo_test/
cp test.bin rs_test/

# Now corrupt both copies identically
echo "Corrupting files (10MB)..."
dd if=/dev/urandom of=corrupt_data bs=1M count=10 2>/dev/null
dd if=corrupt_data of=turbo_test/test.bin conv=notrunc 2>/dev/null
dd if=corrupt_data of=rs_test/test.bin conv=notrunc 2>/dev/null

# Benchmark par2cmdline-turbo
echo "Running par2cmdline-turbo repair..."
cd turbo_test
TURBO_START=$(date +%s.%N)
"$HOME/par2cmdline-turbo/par2" repair test.par2 >/dev/null 2>&1
TURBO_END=$(date +%s.%N)
TURBO_TIME=$(echo "$TURBO_END - $TURBO_START" | bc)
cd ..

# Verify turbo result
TURBO_MD5=$(md5sum turbo_test/test.bin | cut -d' ' -f1)
if [ "$TURBO_MD5" != "$ORIG_MD5" ]; then
    echo "ERROR: turbo repair verification failed!"
    echo "Expected: $ORIG_MD5"
    echo "Got:      $TURBO_MD5"
    exit 1
fi
echo "turbo verification OK"

# Benchmark par2-rs
echo "Running par2-rs repair..."
cd rs_test
RS_START=$(date +%s.%N)
"$PROJECT_ROOT/target/release/repair" test.par2 >/dev/null 2>&1
RS_END=$(date +%s.%N)
RS_TIME=$(echo "$RS_END - $RS_START" | bc)
cd ..

# Verify par2-rs result
RS_MD5=$(md5sum rs_test/test.bin | cut -d' ' -f1)
if [ "$RS_MD5" != "$ORIG_MD5" ]; then
    echo "ERROR: par2-rs repair verification failed!"
    echo "Expected: $ORIG_MD5"
    echo "Got:      $RS_MD5"
    exit 1
fi
echo "par2-rs verification OK"

# Results
SPEEDUP=$(echo "scale=2; $TURBO_TIME / $RS_TIME" | bc)
echo ""
echo "=== Results ==="
echo "turbo: ${TURBO_TIME}s | par2-rs: ${RS_TIME}s | speedup: ${SPEEDUP}x"
