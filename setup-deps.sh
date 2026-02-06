#!/bin/bash
# Download and setup tt-metal dependencies for blackhole-py
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/tt-metal-deps"

# Check if already installed
if [ -f "$DEST/sfpi-toolchain/bin/riscv-tt-elf-g++" ]; then
  echo "Dependencies already installed at $DEST"
  exit 0
fi

URL="https://github.com/boopdotpng/blackhole-py/releases/download/v0.2.0/tt-metal-deps.tar.gz"
TARBALL="$SCRIPT_DIR/tt-metal-deps.tar.gz"

echo "Downloading tt-metal dependencies..."
curl -fsSL "$URL" -o "$TARBALL"

echo "Extracting..."
tar -xzf "$TARBALL" -C "$SCRIPT_DIR"
rm "$TARBALL"

echo "Dependencies installed to $DEST"
"$DEST/sfpi-toolchain/bin/riscv-tt-elf-g++" --version | head -1
