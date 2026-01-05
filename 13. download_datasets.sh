#!/bin/bash

# Download datasets script
# Execution Order: 8

set -e  # Exit on error

echo "Downloading hyperspectral datasets..."
echo "====================================="

# Create data directory
mkdir -p ./data
cd ./data

# Indian Pines dataset
echo "1. Downloading Indian Pines dataset..."
if [ ! -f "Indian_Pines/Indian_pines_corrected.mat" ]; then
    mkdir -p Indian_Pines
    wget -q https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat -O Indian_Pines/Indian_pines_corrected.mat
    wget -q https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat -O Indian_Pines/Indian_pines_gt.mat
    echo "  ✓ Indian Pines downloaded"
else
    echo "  ✓ Indian Pines already exists"
fi

# Pavia University dataset
echo "2. Downloading Pavia University dataset..."
if [ ! -f "Pavia_University/PaviaU.mat" ]; then
    mkdir -p Pavia_University
    wget -q https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat -O Pavia_University/PaviaU.mat
    wget -q https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat -O Pavia_University/PaviaU_gt.mat
    echo "  ✓ Pavia University downloaded"
else
    echo "  ✓ Pavia University already exists"
fi

# Salinas dataset
echo "3. Downloading Salinas dataset..."
if [ ! -f "Salinas/Salinas_corrected.mat" ]; then
    mkdir -p Salinas
    wget -q https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas_corrected.mat -O Salinas/Salinas_corrected.mat
    wget -q https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat -O Salinas/Salinas_gt.mat
    echo "  ✓ Salinas downloaded"
else
    echo "  ✓ Salinas already exists"
fi

# Houston dataset (if available)
echo "4. Downloading Houston dataset..."
if [ ! -f "Houston/Houston.mat" ]; then
    mkdir -p Houston
    # Note: Houston dataset requires registration
    # Uncomment and add URL if you have access
    # wget -q [URL] -O Houston/Houston.mat
    # wget -q [URL] -O Houston/Houston_gt.mat
    echo "  ⚠ Houston dataset requires registration (not downloaded)"
else
    echo "  ✓ Houston already exists"
fi

# Verify downloads
echo ""
echo "Verifying downloads..."
echo "======================"

for dataset in Indian_Pines Pavia_University Salinas; do
    if [ -f "${dataset}/${dataset}_corrected.mat" ] || [ -f "${dataset}/PaviaU.mat" ]; then
        echo "  ✓ ${dataset}: OK"
    else
        echo "  ✗ ${dataset}: MISSING"
    fi
done

echo ""
echo "Dataset download completed!"
echo "Total size: $(du -sh . | cut -f1)"

# Create checksums
echo ""
echo "Creating checksums..."
find . -name "*.mat" -exec md5sum {} \; > ../dataset_checksums.md5
echo "Checksums saved to dataset_checksums.md5"

cd ..
