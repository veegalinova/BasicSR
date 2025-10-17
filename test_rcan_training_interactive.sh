#!/bin/bash
# Interactive test script for RCAN training
# Run this during an interactive session to test the setup

echo "=== RCAN Training Interactive Test ==="
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set environment variables (simulate SLURM environment)
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export SCRATCH=/localscratch

# Create a test scratch directory
TEST_SCRATCH_DIR="$SCRATCH/rcan_training_test_$(date +%Y%m%d_%H%M%S)"
echo "Creating test scratch directory: $TEST_SCRATCH_DIR"
if ! mkdir -p "$TEST_SCRATCH_DIR"; then
    echo "Error: Failed to create test scratch directory"
    exit 1
fi

# Copy project to scratch for testing
echo "Copying project to test scratch directory..."
cp -r /group/jug/Vera/dev/BasicSR/* "$TEST_SCRATCH_DIR/"

# Navigate to the test scratch project directory
cd "$TEST_SCRATCH_DIR"

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in current directory"
    exit 1
fi

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# Setup environment using uv
echo "Setting up environment with uv..."

# Configure UV for optimal performance on this filesystem
export UV_CACHE_DIR="$(pwd)/.uv-cache"  # Cache in same directory as environment
export UV_LINK_MODE=copy                 # Avoid hardlink warnings
export UV_PARALLEL=true                  # Enable parallel downloads

echo "UV cache directory: $UV_CACHE_DIR"
uv sync

# Verify environment
echo "Verifying environment..."
echo "Python version: $(uv run python --version)"
echo "PyTorch version: $(uv run python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(uv run python -c 'import torch; print(torch.cuda.device_count())')"

# Test BasicSR import
echo "Testing BasicSR import..."
uv run python -c "import basicsr; print('BasicSR imported successfully')"

# Test configuration file
echo "Testing configuration file..."
if [ -f "options/train/RCAN/train_rcan_biosr_microtubules.yml" ]; then
    echo "Configuration file found: options/train/RCAN/train_rcan_biosr_microtubules.yml"
else
    echo "Warning: Configuration file not found"
fi

# Test a dry run (just check if the command would work)
echo "Testing training command (dry run)..."
echo "Command that would be run:"
echo "uv run python basicsr/train.py -opt options/train/RCAN/train_rcan_biosr_microtubules.yml"

# Run full training
echo "Running full training..."
uv run python basicsr/train.py -opt options/train/RCAN/train_rcan_biosr_microtubules.yml

# Copy any results back
echo "Copying test results back..."
cp -r experiments/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || true
cp -r tb_logger/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || true
cp -r wandb/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || true

# Clean up test directory
echo "Cleaning up test directory..."
rm -rf "$TEST_SCRATCH_DIR"

echo "End Time: $(date)"
echo "Interactive test completed!"
