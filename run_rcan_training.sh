#!/bin/bash
#SBATCH --job-name=rcan_training
#SBATCH --output=rcan_training_%j.out
#SBATCH --error=rcan_training_%j.err
#SBATCH --time=30:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Check if config file is provided as argument
if [ -z "$1" ]; then
    echo "Error: No config file provided"
    echo "Usage: $0 <path_to_config.yml>"
    exit 1
fi

CONFIG_FILE="$1"
echo "Using config file: $CONFIG_FILE"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Load necessary modules (adjust based on your HPC system)
# Uncomment and modify these lines based on your HPC environment
# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU instead of SLURM_LOCALID which may be undefined
export PYTHONPATH="/group/jug/Vera/dev/BasicSR:${PYTHONPATH}"

# Create scratch directory for this job
SCRATCH_DIR="$TMPDIR/rcan_training_${SLURM_JOB_ID}"
echo "Creating scratch directory: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

# Copy project to scratch for better I/O performance
echo "Copying project to scratch directory..."
cp -r /group/jug/Vera/dev/BasicSR/* "$SCRATCH_DIR/"

# Navigate to the scratch project directory
cd "$SCRATCH_DIR" || exit 1

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in current directory"
    exit 1
fi

# Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
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

# Test configuration file
echo "Testing configuration file..."
if [ -f "$CONFIG_FILE" ]; then
    echo "Configuration file found: $CONFIG_FILE"
else
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Run the training
echo "Starting RCAN training..."
echo "Command: uv run python basicsr/train.py -opt $CONFIG_FILE"

uv run python basicsr/train.py -opt "$CONFIG_FILE"
TRAINING_EXIT_CODE=$?

# Check exit status
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"

    # Copy results back to original location
    echo "Copying results back to original location..."
    cp -r experiments/ /group/jug/Vera/dev/BasicSR/ || echo "Warning: Failed to copy experiments/"
    cp -r tb_logger/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || echo "Warning: Failed to copy tb_logger/"
    cp -r wandb/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || echo "Warning: Failed to copy wandb/"

    echo "Results copied back successfully!"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE"

    # Still copy any partial results
    echo "Copying partial results back to original location..."
    cp -r experiments/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || echo "Warning: Failed to copy experiments/"
    cp -r tb_logger/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || echo "Warning: Failed to copy tb_logger/"
    cp -r wandb/ /group/jug/Vera/dev/BasicSR/ 2>/dev/null || echo "Warning: Failed to copy wandb/"
fi

# Clean up scratch directory
# echo "Cleaning up scratch directory..."
# rm -rf "$SCRATCH_DIR"

echo "End Time: $(date)"
echo "Job completed."
