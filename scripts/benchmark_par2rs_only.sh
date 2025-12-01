#!/usr/bin/env bash
# Benchmarks par2-rs only (no turbo) on a 1GB test file.
# Measures: create (10% redundancy), verify (clean), repair (100MB corruption).
# Always cleans up temporary files.

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BLUE}$*${NC}"; }
ok()    { echo -e "${GREEN}$*${NC}"; }
warn()  { echo -e "${YELLOW}$*${NC}"; }
err()   { echo -e "${RED}$*${NC}"; }

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT INT TERM
cd "$TMPDIR"

info "par2-rs only benchmark (1GB)"
echo

# Locate par2-rs binaries
PAR2_RS_BIN_DIR="${PAR2_RS_BIN_DIR:-$HOME/par2-rs/target/release}"
PAR2_RS_CREATE="$PAR2_RS_BIN_DIR/create"
PAR2_RS_REPAIR="$PAR2_RS_BIN_DIR/repair"

if [[ ! -x "$PAR2_RS_CREATE" || ! -x "$PAR2_RS_REPAIR" ]]; then
  warn "par2-rs binaries not found; building release..."
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"
  fi
  (cd "$HOME/par2-rs" && cargo build --release --bins)
fi

if [[ ! -x "$PAR2_RS_CREATE" || ! -x "$PAR2_RS_REPAIR" ]]; then
  err "Failed to locate par2-rs binaries at $PAR2_RS_BIN_DIR"
  exit 1
fi

ok "Using par2-rs from: $PAR2_RS_BIN_DIR"
echo

# md5 command
if command -v md5sum >/dev/null 2>&1; then
  md5_calc() { md5sum "$1" | awk '{print $1}'; }
else
  md5_calc() { md5 -q "$1"; }
fi

# Helper to time a command in seconds
time_cmd() {
  local out_file=$1; shift
  local start end elapsed

  # Portable timing using bash built-in SECONDS
  SECONDS=0
  if "$@"; then
    elapsed=$SECONDS
    printf "%.2f\n" "$elapsed" > "$out_file"
    return 0
  else
    local exit_code=$?
    err "Command failed with exit code $exit_code: $*"
    return $exit_code
  fi
}

warn "Creating 1GB test file..."
dd if=/dev/urandom of=testfile.bin bs=1M count=1024 status=none
ok "Created testfile.bin (1GB)"
echo

warn "Computing original MD5..."
ORIG_MD5=$(md5_calc testfile.bin)
ok "Original MD5: $ORIG_MD5"
echo

# Prepare separate working copy for par2-rs
cp testfile.bin testfile.bin.par2rs

info "=== Create (par2-rs, 10% redundancy) ==="
time_cmd create.time "$PAR2_RS_CREATE" --redundancy 10 --output testfile.bin.par2rs.par2 testfile.bin.par2rs
CREATE_TIME=$(cat create.time)
PAR2_SIZE=$(du -sh testfile.bin.par2rs.par2 | awk '{print $1}')
ok "Create done in ${CREATE_TIME}s (PAR2 size: $PAR2_SIZE)"
echo

info "=== Verify (clean) ==="
time_cmd verify.time "$PAR2_RS_REPAIR" testfile.bin.par2rs.par2
VERIFY_TIME=$(cat verify.time)
ok "Verify done in ${VERIFY_TIME}s"
echo

info "=== Repair (100MB corrupted) ==="
warn "Corrupting 100MB at 500MB offset..."
dd if=/dev/urandom of=corruption.dat bs=1M count=100 status=none
dd if=corruption.dat of=testfile.bin.par2rs bs=1M count=100 conv=notrunc seek=500 status=none
rm -f corruption.dat
ok "Corrupted testfile.bin.par2rs"

time_cmd repair.time "$PAR2_RS_REPAIR" testfile.bin.par2rs.par2
REPAIR_TIME=$(cat repair.time)
ok "Repair done in ${REPAIR_TIME}s"

warn "Verifying MD5 after repair..."
REPAIRED_MD5=$(md5_calc testfile.bin.par2rs)
if [[ "$REPAIRED_MD5" == "$ORIG_MD5" ]]; then
  ok "MD5 match: $REPAIRED_MD5"
else
  err "MD5 mismatch! repaired=$REPAIRED_MD5 original=$ORIG_MD5"
  exit 2
fi
echo

info "=== Summary ==="
printf "Create (1GB, 10%% red.)  %s\n" "${CREATE_TIME}s"
printf "Verify (1GB, clean)     %s\n" "${VERIFY_TIME}s"
printf "Repair (100MB corrupt)  %s\n" "${REPAIR_TIME}s"

ok "Benchmark complete. Cleaning up temporary files."
