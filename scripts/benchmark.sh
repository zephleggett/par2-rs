#!/bin/bash
# Benchmark par2-rs against par2cmdline-turbo
# Tests creation and repair performance on a 1GB test file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== PAR2 Performance Benchmark ===${NC}"
echo ""

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Darwin" ]; then
    PLATFORM="macos"
    if [ "$ARCH" = "arm64" ]; then
        TURBO_URL="https://github.com/animetosho/par2cmdline-turbo/releases/download/v1.3.0/par2cmdline-turbo-1.3.0-macos-arm64.zip"
    else
        echo -e "${RED}Unsupported macOS architecture: $ARCH${NC}"
        exit 1
    fi
elif [ "$OS" = "Linux" ]; then
    PLATFORM="linux"
    if [ "$ARCH" = "x86_64" ]; then
        TURBO_URL="https://github.com/animetosho/par2cmdline-turbo/releases/download/v1.3.0/par2cmdline-turbo-1.3.0-linux-amd64.zip"
    else
        echo -e "${RED}Unsupported Linux architecture: $ARCH${NC}"
        exit 1
    fi
else
    echo -e "${RED}Unsupported OS: $OS${NC}"
    exit 1
fi

echo -e "${BLUE}Platform: $PLATFORM $ARCH${NC}"
echo ""

# Find par2-rs repair binary BEFORE changing to temp directory
if [ -n "$PAR2_RS_PATH" ]; then
    # User-specified path (for remote testing)
    PAR2_RS_REPAIR="$PAR2_RS_PATH"
    PAR2_RS_CREATE="$(dirname "$PAR2_RS_PATH")/create"
elif [ -n "$GITHUB_WORKSPACE" ]; then
    # Running in GitHub Actions
    PAR2_RS_REPAIR="$GITHUB_WORKSPACE/target/release/repair"
    PAR2_RS_CREATE="$GITHUB_WORKSPACE/target/release/create"
else
    # Running locally - try to find in project
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    PAR2_RS_REPAIR="$PROJECT_ROOT/target/release/repair"
    PAR2_RS_CREATE="$PROJECT_ROOT/target/release/create"
fi

# Create temporary directory
BENCH_DIR=$(mktemp -d)
trap "rm -rf $BENCH_DIR" EXIT

cd "$BENCH_DIR"
echo -e "${BLUE}Working directory: $BENCH_DIR${NC}"
echo ""

# Download and extract par2cmdline-turbo (or use provided path)
if [ -n "$PAR2_TURBO_PATH" ] && [ -f "$PAR2_TURBO_PATH" ]; then
    echo -e "${GREEN}Using provided par2cmdline-turbo at: $PAR2_TURBO_PATH${NC}"
    cp "$PAR2_TURBO_PATH" ./par2
    chmod +x par2
else
    echo -e "${YELLOW}Downloading par2cmdline-turbo...${NC}"
    curl -L -o turbo.zip "$TURBO_URL"
    unzip -q turbo.zip
    chmod +x par2
fi

if [ ! -f "$PAR2_RS_REPAIR" ] || [ ! -f "$PAR2_RS_CREATE" ]; then
    echo -e "${RED}Error: par2-rs binaries not found at $PAR2_RS_REPAIR${NC}"
    echo "Please run: cargo build --release --bins"
    exit 1
fi

echo -e "${GREEN}Found par2-rs at: $PAR2_RS_REPAIR${NC}"
echo ""

# Create 1GB test file efficiently (use /dev/zero for speed)
echo -e "${YELLOW}Creating 1GB test file...${NC}"
dd if=/dev/zero of=testfile.bin bs=1M count=1024 status=none
echo -e "${GREEN}Created testfile.bin (1GB)${NC}"
echo ""

# Calculate MD5 hash of original file for verification
echo -e "${YELLOW}Calculating MD5 hash of original file...${NC}"
if [ "$OS" = "Darwin" ]; then
    ORIGINAL_MD5=$(md5 -q testfile.bin)
else
    ORIGINAL_MD5=$(md5sum testfile.bin | cut -d' ' -f1)
fi
echo -e "${GREEN}Original MD5: $ORIGINAL_MD5${NC}"
echo ""

# Function to format time
format_time() {
    local seconds=$1
    printf "%.2fs" "$seconds"
}

# Function to format memory (input in KB)
format_mem() {
    local kb=$1
    if [ "$kb" -ge 1048576 ]; then
        printf "%.1fGB" "$(echo "scale=1; $kb / 1048576" | bc)"
    elif [ "$kb" -ge 1024 ]; then
        printf "%.1fMB" "$(echo "scale=1; $kb / 1024" | bc)"
    else
        printf "%dKB" "$kb"
    fi
}

# Function to run command and capture time + peak memory
# Usage: run_timed CMD_ARRAY -> sets TIME_RESULT and MEM_RESULT (KB)
run_timed() {
    local time_file=$(mktemp)
    local start end
    start=$(date +%s.%N)

    if [ "$OS" = "Darwin" ]; then
        # macOS: /usr/bin/time -l reports bytes
        /usr/bin/time -l "$@" 2>"$time_file" >/dev/null
        end=$(date +%s.%N)
        # Extract "maximum resident set size" (in bytes on macOS)
        local mem_bytes=$(grep "maximum resident set size" "$time_file" | awk '{print $1}')
        MEM_RESULT=$((mem_bytes / 1024))
    else
        # Linux: /usr/bin/time -v reports KB
        /usr/bin/time -v "$@" 2>"$time_file" >/dev/null
        end=$(date +%s.%N)
        MEM_RESULT=$(grep "Maximum resident set size" "$time_file" | awk '{print $NF}')
    fi

    TIME_RESULT=$(echo "$end - $start" | bc)
    rm -f "$time_file"
}

# Benchmark creation with both tools
echo -e "${BLUE}=== Creation Benchmark ===${NC}"
echo ""

# Create working copies for each test
cp testfile.bin testfile.bin.turbo
cp testfile.bin testfile.bin.par2rs

# Benchmark creation with par2cmdline-turbo
echo -e "${YELLOW}par2cmdline-turbo: Creating PAR2 files (10% redundancy)...${NC}"
run_timed ./par2 c -r10 -q testfile.bin.turbo.par2 testfile.bin.turbo
TURBO_CREATE_TIME=$TIME_RESULT
TURBO_CREATE_MEM=$MEM_RESULT

TURBO_PAR2_SIZE=$(du -sh testfile.bin.turbo.par2 | cut -f1)
echo -e "${GREEN}✓ $(format_time $TURBO_CREATE_TIME), $(format_mem $TURBO_CREATE_MEM) peak (PAR2: $TURBO_PAR2_SIZE)${NC}"
echo ""

# Benchmark creation with par2-rs
echo -e "${YELLOW}par2-rs: Creating PAR2 files (10% redundancy)...${NC}"
run_timed "$PAR2_RS_CREATE" --redundancy 10 --output testfile.bin.par2rs.par2 testfile.bin.par2rs
PAR2RS_CREATE_TIME=$TIME_RESULT
PAR2RS_CREATE_MEM=$MEM_RESULT

PAR2RS_PAR2_SIZE=$(du -sh testfile.bin.par2rs.par2 | cut -f1)
echo -e "${GREEN}✓ $(format_time $PAR2RS_CREATE_TIME), $(format_mem $PAR2RS_CREATE_MEM) peak (PAR2: $PAR2RS_PAR2_SIZE)${NC}"
echo ""

# Calculate speedup
CREATE_SPEEDUP=$(echo "scale=2; $TURBO_CREATE_TIME / $PAR2RS_CREATE_TIME" | bc)
if (( $(echo "$CREATE_SPEEDUP > 1" | bc -l) )); then
    echo -e "${GREEN}par2-rs is ${CREATE_SPEEDUP}x faster at creation${NC}"
else
    CREATE_SLOWDOWN=$(echo "scale=2; $PAR2RS_CREATE_TIME / $TURBO_CREATE_TIME" | bc)
    echo -e "${YELLOW}par2-rs is ${CREATE_SLOWDOWN}x slower at creation${NC}"
fi
echo ""

# Benchmark verification (no corruption)
echo -e "${BLUE}=== Verification Benchmark (clean file) ===${NC}"
echo ""

echo -e "${YELLOW}par2cmdline-turbo: Verifying file...${NC}"
run_timed ./par2 v -q testfile.bin.turbo.par2
TURBO_VERIFY_TIME=$TIME_RESULT
TURBO_VERIFY_MEM=$MEM_RESULT
echo -e "${GREEN}✓ $(format_time $TURBO_VERIFY_TIME), $(format_mem $TURBO_VERIFY_MEM) peak${NC}"
echo ""

echo -e "${YELLOW}par2-rs: Verifying file...${NC}"
run_timed "$PAR2_RS_REPAIR" testfile.bin.par2rs.par2
PAR2RS_VERIFY_TIME=$TIME_RESULT
PAR2RS_VERIFY_MEM=$MEM_RESULT
echo -e "${GREEN}✓ $(format_time $PAR2RS_VERIFY_TIME), $(format_mem $PAR2RS_VERIFY_MEM) peak${NC}"
echo ""

# Calculate speedup
VERIFY_SPEEDUP=$(echo "scale=2; $TURBO_VERIFY_TIME / $PAR2RS_VERIFY_TIME" | bc)
if (( $(echo "$VERIFY_SPEEDUP > 1" | bc -l) )); then
    echo -e "${GREEN}par2-rs is ${VERIFY_SPEEDUP}x faster at verification${NC}"
else
    VERIFY_SLOWDOWN=$(echo "scale=2; $PAR2RS_VERIFY_TIME / $TURBO_VERIFY_TIME" | bc)
    echo -e "${YELLOW}par2-rs is ${VERIFY_SLOWDOWN}x slower at verification${NC}"
fi
echo ""

# Benchmark repair
echo -e "${BLUE}=== Repair Benchmark (100MB corrupted) ===${NC}"
echo ""

# Corrupt both files the same way (100MB at offset 500MB)
echo -e "${YELLOW}Corrupting 100MB of both test files...${NC}"
# Save random data to ensure both files get identical corruption
dd if=/dev/urandom of=corruption.dat bs=1M count=100 status=none
dd if=corruption.dat of=testfile.bin.turbo bs=1M count=100 conv=notrunc seek=500 status=none
dd if=corruption.dat of=testfile.bin.par2rs bs=1M count=100 conv=notrunc seek=500 status=none
rm corruption.dat
echo -e "${GREEN}Corrupted 100MB (identical corruption for both)${NC}"
echo ""

# Benchmark repair with par2cmdline-turbo
echo -e "${YELLOW}par2cmdline-turbo: Repairing file...${NC}"
run_timed ./par2 r -q testfile.bin.turbo.par2
TURBO_REPAIR_TIME=$TIME_RESULT
TURBO_REPAIR_MEM=$MEM_RESULT
echo -e "${GREEN}✓ $(format_time $TURBO_REPAIR_TIME), $(format_mem $TURBO_REPAIR_MEM) peak${NC}"

# Verify MD5 hash after turbo repair
if [ "$OS" = "Darwin" ]; then
    TURBO_REPAIRED_MD5=$(md5 -q testfile.bin.turbo)
else
    TURBO_REPAIRED_MD5=$(md5sum testfile.bin.turbo | cut -d' ' -f1)
fi

if [ "$TURBO_REPAIRED_MD5" = "$ORIGINAL_MD5" ]; then
    echo -e "${GREEN}✓ MD5 verification passed${NC}"
else
    echo -e "${RED}✗ MD5 verification FAILED! Repaired: $TURBO_REPAIRED_MD5 != Original: $ORIGINAL_MD5${NC}"
    exit 1
fi
echo ""

# Benchmark repair with par2-rs
echo -e "${YELLOW}par2-rs: Repairing file...${NC}"
run_timed "$PAR2_RS_REPAIR" testfile.bin.par2rs.par2
PAR2RS_REPAIR_TIME=$TIME_RESULT
PAR2RS_REPAIR_MEM=$MEM_RESULT
echo -e "${GREEN}✓ $(format_time $PAR2RS_REPAIR_TIME), $(format_mem $PAR2RS_REPAIR_MEM) peak${NC}"

# Verify MD5 hash after par2-rs repair
if [ "$OS" = "Darwin" ]; then
    PAR2RS_REPAIRED_MD5=$(md5 -q testfile.bin.par2rs)
else
    PAR2RS_REPAIRED_MD5=$(md5sum testfile.bin.par2rs | cut -d' ' -f1)
fi

if [ "$PAR2RS_REPAIRED_MD5" = "$ORIGINAL_MD5" ]; then
    echo -e "${GREEN}✓ MD5 verification passed${NC}"
else
    echo -e "${RED}✗ MD5 verification FAILED! Repaired: $PAR2RS_REPAIRED_MD5 != Original: $ORIGINAL_MD5${NC}"
    exit 1
fi
echo ""

# Calculate speedup
REPAIR_SPEEDUP=$(echo "scale=2; $TURBO_REPAIR_TIME / $PAR2RS_REPAIR_TIME" | bc)
if (( $(echo "$REPAIR_SPEEDUP > 1" | bc -l) )); then
    echo -e "${GREEN}par2-rs is ${REPAIR_SPEEDUP}x faster at repair${NC}"
else
    REPAIR_SLOWDOWN=$(echo "scale=2; $PAR2RS_REPAIR_TIME / $TURBO_REPAIR_TIME" | bc)
    echo -e "${YELLOW}par2-rs is ${REPAIR_SLOWDOWN}x slower at repair${NC}"
fi
echo ""

# Summary table
echo -e "${BLUE}=== Summary ===${NC}"
echo ""
printf "%-22s %-18s %-18s %-10s\n" "Operation" "par2cmdline-turbo" "par2-rs" "Speedup"
printf "%-22s %-18s %-18s %-10s\n" "----------------------" "------------------" "------------------" "----------"
printf "%-22s %-18s %-18s %-10s\n" "Create (time)" "$(format_time $TURBO_CREATE_TIME)" "$(format_time $PAR2RS_CREATE_TIME)" "${CREATE_SPEEDUP}x"
printf "%-22s %-18s %-18s %-10s\n" "Create (memory)" "$(format_mem $TURBO_CREATE_MEM)" "$(format_mem $PAR2RS_CREATE_MEM)" ""
printf "%-22s %-18s %-18s %-10s\n" "Verify (time)" "$(format_time $TURBO_VERIFY_TIME)" "$(format_time $PAR2RS_VERIFY_TIME)" "${VERIFY_SPEEDUP}x"
printf "%-22s %-18s %-18s %-10s\n" "Verify (memory)" "$(format_mem $TURBO_VERIFY_MEM)" "$(format_mem $PAR2RS_VERIFY_MEM)" ""
printf "%-22s %-18s %-18s %-10s\n" "Repair (time)" "$(format_time $TURBO_REPAIR_TIME)" "$(format_time $PAR2RS_REPAIR_TIME)" "${REPAIR_SPEEDUP}x"
printf "%-22s %-18s %-18s %-10s\n" "Repair (memory)" "$(format_mem $TURBO_REPAIR_MEM)" "$(format_mem $PAR2RS_REPAIR_MEM)" ""
echo ""

# Save results to file for artifact upload
RESULTS_FILE="benchmark-results-${PLATFORM}-${ARCH}.txt"
cat > "$RESULTS_FILE" <<EOF
PAR2 Performance Benchmark Results
==================================
Platform: $PLATFORM $ARCH
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Original File MD5: $ORIGINAL_MD5

Creation (1GB file, 10% redundancy):
  par2cmdline-turbo: $(format_time $TURBO_CREATE_TIME), $(format_mem $TURBO_CREATE_MEM) peak
  par2-rs:           $(format_time $PAR2RS_CREATE_TIME), $(format_mem $PAR2RS_CREATE_MEM) peak
  Speedup:           ${CREATE_SPEEDUP}x

Verification (1GB file, clean):
  par2cmdline-turbo: $(format_time $TURBO_VERIFY_TIME), $(format_mem $TURBO_VERIFY_MEM) peak
  par2-rs:           $(format_time $PAR2RS_VERIFY_TIME), $(format_mem $PAR2RS_VERIFY_MEM) peak
  Speedup:           ${VERIFY_SPEEDUP}x

Repair (100MB corrupted):
  par2cmdline-turbo: $(format_time $TURBO_REPAIR_TIME), $(format_mem $TURBO_REPAIR_MEM) peak
  par2-rs:           $(format_time $PAR2RS_REPAIR_TIME), $(format_mem $PAR2RS_REPAIR_MEM) peak
  Speedup:           ${REPAIR_SPEEDUP}x

MD5 Verification:
  par2cmdline-turbo: ✓ PASSED
  par2-rs:           ✓ PASSED
EOF

echo -e "${GREEN}Saved results to: $RESULTS_FILE${NC}"
echo ""

# Export results for GitHub Actions
if [ -n "$GITHUB_OUTPUT" ]; then
    echo "turbo_create_time=$TURBO_CREATE_TIME" >> "$GITHUB_OUTPUT"
    echo "par2rs_create_time=$PAR2RS_CREATE_TIME" >> "$GITHUB_OUTPUT"
    echo "turbo_create_mem=$TURBO_CREATE_MEM" >> "$GITHUB_OUTPUT"
    echo "par2rs_create_mem=$PAR2RS_CREATE_MEM" >> "$GITHUB_OUTPUT"
    echo "create_speedup=$CREATE_SPEEDUP" >> "$GITHUB_OUTPUT"
    echo "turbo_verify_time=$TURBO_VERIFY_TIME" >> "$GITHUB_OUTPUT"
    echo "par2rs_verify_time=$PAR2RS_VERIFY_TIME" >> "$GITHUB_OUTPUT"
    echo "turbo_verify_mem=$TURBO_VERIFY_MEM" >> "$GITHUB_OUTPUT"
    echo "par2rs_verify_mem=$PAR2RS_VERIFY_MEM" >> "$GITHUB_OUTPUT"
    echo "verify_speedup=$VERIFY_SPEEDUP" >> "$GITHUB_OUTPUT"
    echo "turbo_repair_time=$TURBO_REPAIR_TIME" >> "$GITHUB_OUTPUT"
    echo "par2rs_repair_time=$PAR2RS_REPAIR_TIME" >> "$GITHUB_OUTPUT"
    echo "turbo_repair_mem=$TURBO_REPAIR_MEM" >> "$GITHUB_OUTPUT"
    echo "par2rs_repair_mem=$PAR2RS_REPAIR_MEM" >> "$GITHUB_OUTPUT"
    echo "repair_speedup=$REPAIR_SPEEDUP" >> "$GITHUB_OUTPUT"
    echo "original_md5=$ORIGINAL_MD5" >> "$GITHUB_OUTPUT"
    echo "results_file=$RESULTS_FILE" >> "$GITHUB_OUTPUT"
    echo "platform=${PLATFORM}-${ARCH}" >> "$GITHUB_OUTPUT"
fi

# Copy results file to GITHUB_WORKSPACE if in CI
if [ -n "$GITHUB_WORKSPACE" ]; then
    cp "$RESULTS_FILE" "$GITHUB_WORKSPACE/"
    echo -e "${GREEN}Copied results to workspace${NC}"
fi

echo -e "${GREEN}Benchmark complete!${NC}"
