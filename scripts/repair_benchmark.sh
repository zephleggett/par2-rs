#!/bin/bash
# Clean benchmark: compare par2cmdline, par2cmdline-turbo, and par2-rs
# Simple test: delete a file and time full repair
#
# Usage:
#   ./clean_benchmark.sh              # Run only par2-rs
#   ./clean_benchmark.sh --all        # Run all three implementations
#   ./clean_benchmark.sh --with-turbo # Run par2-rs and par2cmdline-turbo

set -e

PAR2_STANDARD="${PAR2_STANDARD:-/opt/homebrew/bin/par2}"
PAR2_TURBO="${PAR2_TURBO:-$(pwd)/tmp/par2-turbo}"
TEST_DATA="/private/tmp/orig-test-data"
WORK_DIR="/tmp/par2-clean-bench-$$"

# Parse arguments
RUN_STANDARD=false
RUN_TURBO=false
RUN_RS=true

if [ "$1" = "--all" ]; then
    RUN_STANDARD=true
    RUN_TURBO=true
    RUN_RS=true
elif [ "$1" = "--with-turbo" ]; then
    RUN_TURBO=true
    RUN_RS=true
fi

echo "=== Clean PAR2 Benchmark ==="
echo ""

# Build par2-rs
echo "Building par2-rs..."
cargo build --release --bin repair 2>&1 | grep -E "Finished|Compiling par2-rs"
echo ""

PAR2_RS="/Users/zeph/Developer/experiments/par2-rs/target/release/repair"

# Create work directories
mkdir -p "$WORK_DIR/standard"
mkdir -p "$WORK_DIR/turbo"
mkdir -p "$WORK_DIR/rs"

echo "Test: Delete file and repair"
echo "----------------------------"
echo ""

#==========================================
# Test par2cmdline (standard)
#==========================================
if [ "$RUN_STANDARD" = true ]; then
    echo "par2cmdline (standard):"
    cd "$WORK_DIR/standard"
    cp -r "$TEST_DATA"/* .

    # Find a file to delete (should be recoverable from PAR2)
    FILE_TO_DELETE=$(ls *.r00 2>/dev/null | head -1 || ls *.rar 2>/dev/null | head -1)

    if [ -z "$FILE_TO_DELETE" ]; then
        echo "  ✗ No recoverable files found to delete"
        exit 1
    fi

    # Save original MD5 before deleting
    echo "  Computing original MD5..."
    ORIGINAL_MD5=$(md5 -q "$FILE_TO_DELETE")

    echo "  Deleting: $FILE_TO_DELETE"
    rm -f "$FILE_TO_DELETE"

    # Time the repair
    echo "  Repairing..."
    /usr/bin/time -l "$PAR2_STANDARD" r bZd2VVKcrSYWJREe17t8NJ1TROif1D7p.par2 2>&1 | grep -E "real|user|sys|peak memory"

    # Verify file was restored and matches original
    if [ -f "$FILE_TO_DELETE" ]; then
        REPAIRED_MD5=$(md5 -q "$FILE_TO_DELETE")
        if [ "$ORIGINAL_MD5" = "$REPAIRED_MD5" ]; then
            echo "  ✓ File restored successfully (MD5 verified)"
        else
            echo "  ✗ File restored but MD5 MISMATCH!"
            echo "    Original:  $ORIGINAL_MD5"
            echo "    Repaired:  $REPAIRED_MD5"
            exit 1
        fi
    else
        echo "  ✗ File NOT restored"
        exit 1
    fi

    echo ""
fi

#==========================================
# Test par2cmdline-turbo
#==========================================
if [ "$RUN_TURBO" = true ]; then
    echo "par2cmdline-turbo:"
    cd "$WORK_DIR/turbo"
    cp -r "$TEST_DATA"/* .

    # Find the same file to delete
    FILE_TO_DELETE=$(ls *.r00 2>/dev/null | head -1 || ls *.rar 2>/dev/null | head -1)

    # Save original MD5 before deleting
    echo "  Computing original MD5..."
    ORIGINAL_MD5=$(md5 -q "$FILE_TO_DELETE")

    echo "  Deleting: $FILE_TO_DELETE"
    rm -f "$FILE_TO_DELETE"

    # Time the repair
    echo "  Repairing..."
    /usr/bin/time -l "$PAR2_TURBO" r bZd2VVKcrSYWJREe17t8NJ1TROif1D7p.par2 2>&1 | grep -E "real|user|sys|peak memory"

    # Verify file was restored and matches original
    if [ -f "$FILE_TO_DELETE" ]; then
        REPAIRED_MD5=$(md5 -q "$FILE_TO_DELETE")
        if [ "$ORIGINAL_MD5" = "$REPAIRED_MD5" ]; then
            echo "  ✓ File restored successfully (MD5 verified)"
        else
            echo "  ✗ File restored but MD5 MISMATCH!"
            echo "    Original:  $ORIGINAL_MD5"
            echo "    Repaired:  $REPAIRED_MD5"
            exit 1
        fi
    else
        echo "  ✗ File NOT restored"
        exit 1
    fi

    echo ""
fi

#==========================================
# Test par2-rs
#==========================================
if [ "$RUN_RS" = true ]; then
    echo "par2-rs:"
    cd "$WORK_DIR/rs"
    cp -r "$TEST_DATA"/* .

    # Delete the same file
    FILE_TO_DELETE=$(ls *.r00 2>/dev/null | head -1 || ls *.rar 2>/dev/null | head -1)

    # Save original MD5 before deleting
    echo "  Computing original MD5..."
    ORIGINAL_MD5=$(md5 -q "$FILE_TO_DELETE")

    echo "  Deleting: $FILE_TO_DELETE"
    rm -f "$FILE_TO_DELETE"

    # Time the repair
    echo "  Repairing..."
    /usr/bin/time -l "$PAR2_RS" bZd2VVKcrSYWJREe17t8NJ1TROif1D7p.par2 2>&1 | grep -E "real|user|sys|peak memory"

    # Verify file was restored and matches original
    if [ -f "$FILE_TO_DELETE" ]; then
        REPAIRED_MD5=$(md5 -q "$FILE_TO_DELETE")
        if [ "$ORIGINAL_MD5" = "$REPAIRED_MD5" ]; then
            echo "  ✓ File restored successfully (MD5 verified)"
        else
            echo "  ✗ File restored but MD5 MISMATCH!"
            echo "    Original:  $ORIGINAL_MD5"
            echo "    Repaired:  $REPAIRED_MD5"
            exit 1
        fi
    else
        echo "  ✗ File NOT restored"
        exit 1
    fi

    echo ""
fi

#==========================================
# Cleanup
#==========================================
cd /tmp
rm -rf "$WORK_DIR"

echo "=== Benchmark Complete ==="
