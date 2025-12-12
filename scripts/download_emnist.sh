#!/bin/bash
# Script to download and extract EMNIST Letters dataset

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== EMNIST Letters Dataset Downloader ===${NC}\n"

# Default data directory
DATA_DIR="${1:-./data/EMNIST}"

echo -e "${YELLOW}Data will be downloaded to: ${DATA_DIR}${NC}\n"

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# EMNIST download URL (using biometrics.nist.gov mirror)
EMNIST_URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
ZIP_FILE="emnist-gzip.zip"

# Check if files already exist
if [ -f "emnist-letters-train-images-idx3-ubyte" ] && [ -f "emnist-letters-train-labels-idx1-ubyte" ]; then
    echo -e "${GREEN}✓ EMNIST Letters training files already exist!${NC}"
    echo -e "\nFiles found:"
    ls -lh emnist-letters-train-*
    echo -e "\n${GREEN}You're ready to run the experiment!${NC}"
    exit 0
fi

# Download EMNIST dataset
echo -e "${YELLOW}Downloading EMNIST dataset...${NC}"
echo -e "URL: ${EMNIST_URL}"
echo -e "This may take a few minutes (the file is ~560MB)...\n"

if command -v wget &> /dev/null; then
    wget -c "$EMNIST_URL" -O "$ZIP_FILE"
elif command -v curl &> /dev/null; then
    curl -L -C - "$EMNIST_URL" -o "$ZIP_FILE"
else
    echo -e "${RED}Error: Neither wget nor curl found. Please install one of them.${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Download complete!${NC}\n"

# Extract the zip file
echo -e "${YELLOW}Extracting EMNIST dataset...${NC}"
unzip -o "$ZIP_FILE"

# The zip contains gzip files, extract those too
echo -e "\n${YELLOW}Decompressing training files...${NC}"
if [ -f "gzip/emnist-letters-train-images-idx3-ubyte.gz" ]; then
    gunzip -f gzip/emnist-letters-train-images-idx3-ubyte.gz
    gunzip -f gzip/emnist-letters-train-labels-idx1-ubyte.gz
    
    # Move to current directory
    mv gzip/emnist-letters-train-images-idx3-ubyte .
    mv gzip/emnist-letters-train-labels-idx1-ubyte .
fi

# Also extract test files for future use
echo -e "${YELLOW}Decompressing test files...${NC}"
if [ -f "gzip/emnist-letters-test-images-idx3-ubyte.gz" ]; then
    gunzip -f gzip/emnist-letters-test-images-idx3-ubyte.gz
    gunzip -f gzip/emnist-letters-test-labels-idx1-ubyte.gz
    
    mv gzip/emnist-letters-test-images-idx3-ubyte .
    mv gzip/emnist-letters-test-labels-idx1-ubyte .
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up...${NC}"
rm -f "$ZIP_FILE"
rm -rf gzip/

echo -e "\n${GREEN}=== Setup Complete! ===${NC}\n"
echo -e "EMNIST Letters files extracted to: ${DATA_DIR}\n"
echo -e "Files:"
ls -lh emnist-letters-*

echo -e "\n${GREEN}You can now run the visualization experiment:${NC}"
echo -e "${BLUE}cd /home/dean/repos/snnfw/build${NC}"
echo -e "${BLUE}./emnist_letters_visualized $(realpath .)${NC}"

echo -e "\n${YELLOW}Note: The experiment expects these files:${NC}"
echo -e "  - emnist-letters-train-images-idx3-ubyte"
echo -e "  - emnist-letters-train-labels-idx1-ubyte"

