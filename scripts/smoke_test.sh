#!/bin/bash
# Smoke test for RL training pipeline
# Runs minimal training to verify setup works end-to-end
#
# Usage:
#   bash scripts/smoke_test.sh

set -e  # Exit on error

echo "====================================="
echo "Running Training Pipeline Smoke Test"
echo "====================================="

# Create temp directory with auto cleanup
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo ""
echo "Test directory: $TEMP_DIR"
echo ""

# Create minimal test config
cat > $TEMP_DIR/test_config.yaml << EOF
experiment_name: smoke_test
patterns:
  - cold_stuck
mechanism: bond_only
threshold: 0.9
max_sessions: 20
total_timesteps: 1000
n_envs: 2
batch_size: 32
n_epochs: 2
hidden_size: 64
log_dir: $TEMP_DIR/logs
save_freq: 500
eval_freq: 500
eval_episodes: 5
seed: 42
EOF

# Run integration tests
echo "Running integration tests..."
pytest tests/test_training_integration.py -v

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Integration tests failed"
    exit 1
fi

echo ""
echo "Running minimal training..."
python run_RL_experiment.py \
    --config $TEMP_DIR/test_config.yaml \
    --output-dir $TEMP_DIR/models \
    --no-eval

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "====================================="
echo "✓ Smoke test PASSED"
echo "====================================="
echo ""
echo "Your training pipeline is working correctly!"
echo "You can now run full experiments with confidence."
