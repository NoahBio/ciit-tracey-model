# Omniscient RL Testing Framework

This directory contains comprehensive tests for the omniscient RL implementation.

## Test Structure

### Unit Tests

#### `test_omniscient_wrapper.py`
Tests for the `OmniscientObservationWrapper` Gymnasium wrapper.

**Coverage:**
- Wrapper initialization and observation space definition
- Extended observation space (12 omniscient components)
- Observation extraction from environment
- Normalization functions (u_matrix, RS, entropy)
- Mechanism type mapping (5 mechanism types → integers)
- Perception tracking (actual vs perceived actions)
- Edge cases (perception disabled, identical bounds)
- Gymnasium compatibility
- Multi-step consistency
- Equivalence with unwrapped environment

**Run:**
```bash
pytest tests/test_omniscient_wrapper.py -v
```

#### `test_omniscient_networks.py`
Tests for the `OmniscientTherapyNet` neural network architecture.

**Coverage:**
- Network initialization
- Component verification (embeddings, u_matrix processor, MLP)
- Forward pass (single and batch observations)
- U-matrix compression (64 → 32 dims)
- Embedding dimensions
- Actor and Critic networks
- Network factory function
- Device handling (CPU/CUDA)
- Different hidden sizes
- Batch processing
- Input dimension verification (471 dims)
- Gradient flow

**Run:**
```bash
pytest tests/test_omniscient_networks.py -v
```

### Integration Tests

#### `test_omniscient_training.py`
Integration tests for the complete training pipeline.

**Coverage:**
- Environment creation with wrapper
- Config loading from YAML
- Network integration with wrapped observations
- Tianshou Batch compatibility
- Training loop mechanics
- Checkpoint save/load
- Vectorized environments
- End-to-end smoke test
- Observation consistency during training
- Memory leak detection
- Different configurations (all mechanisms, perception on/off)

**Run:**
```bash
pytest tests/test_omniscient_training.py -v
```

### Smoke Tests

#### `test_omniscient_smoke.py`
Comprehensive end-to-end smoke test that can be run standalone.

**Coverage:**
- Wrapper functionality
- Network initialization and forward pass
- Vectorized environments
- Training loop (200 steps)
- Checkpoint save/load
- Evaluation rollout
- All mechanism types

**Run as pytest:**
```bash
pytest tests/test_omniscient_smoke.py -v
```

**Run as standalone script:**
```bash
python tests/test_omniscient_smoke.py
```

## Running All Tests

### Run all omniscient tests:
```bash
pytest tests/test_omniscient_*.py -v
```

### Run with coverage:
```bash
pytest tests/test_omniscient_*.py --cov=src.environment.omniscient_wrapper --cov=src.training.omniscient_networks -v
```

### Run quick smoke test only:
```bash
python tests/test_omniscient_smoke.py
```

## Test Organization

```
tests/
├── test_omniscient_wrapper.py      # Unit tests for wrapper
├── test_omniscient_networks.py     # Unit tests for networks
├── test_omniscient_training.py     # Integration tests
├── test_omniscient_smoke.py        # Comprehensive smoke test
└── README_OMNISCIENT_TESTS.md      # This file
```

## What Gets Tested

### Wrapper (`OmniscientObservationWrapper`)
✅ Observation space extension (9 new components)
✅ U-matrix normalization to [-1, 1]
✅ RS normalization to [-1, 1]
✅ Bond in [0, 1]
✅ Entropy normalization to [-1, 1]
✅ Mechanism type mapping (5 types)
✅ Perception tracking (actual vs perceived actions)
✅ Misperception rate in [0, 1]
✅ Perception enabled flag (0 or 1)
✅ Edge cases (identical values, identical bounds)
✅ Gymnasium compatibility
✅ Multi-step consistency
✅ Reward/termination equivalence with unwrapped env

### Networks (`OmniscientTherapyNet`, `Actor`, `Critic`)
✅ Network initialization
✅ All components present (embeddings, u_matrix processor, MLP, LayerNorm)
✅ Correct embedding dimensions
✅ U-matrix processor architecture (64→48→32)
✅ Forward pass (single and batch)
✅ Gradient computation
✅ Actor outputs correct logits shape (batch_size, 8)
✅ Critic outputs correct value shape (batch_size, 1)
✅ Probabilities sum to 1.0
✅ Factory function creates both networks
✅ Device placement (CPU/CUDA)
✅ Different hidden sizes
✅ Batch sizes from 1 to 128
✅ Total input dimension is 471
✅ Gradients flow to all components

### Training Integration
✅ Environment creation with wrapper
✅ Config loading from YAML
✅ Networks accept wrapped observations
✅ Tianshou Batch compatibility
✅ Network output → action → environment step
✅ Multi-step training loop
✅ Checkpoint save/load
✅ Checkpoint restores exact weights
✅ Vectorized environments (DummyVectorEnv)
✅ Vectorized observations to networks
✅ Observations remain valid over episode
✅ No memory leaks over 1000 steps
✅ All 5 mechanism types
✅ Perception enabled/disabled

### Smoke Test
✅ Wrapper works end-to-end
✅ Networks work end-to-end
✅ Vectorized environments work
✅ Training loop runs (200 steps)
✅ Checkpoint save/load works
✅ Evaluation rollout works
✅ All mechanism types work

## Expected Test Duration

- **Unit tests (wrapper)**: ~10-15 seconds
- **Unit tests (networks)**: ~15-20 seconds
- **Integration tests**: ~20-30 seconds
- **Smoke test**: ~30-45 seconds
- **Total**: ~2-3 minutes

## Dependencies

All tests use the same dependencies as the main project:
- pytest
- numpy
- torch
- gymnasium
- tianshou

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run omniscient RL tests
  run: |
    pytest tests/test_omniscient_*.py -v --tb=short
```

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure you're running from the project root:
```bash
cd /home/second_partition/Git_Repos/ciit-tracey-model
pytest tests/test_omniscient_*.py
```

### Issue: CUDA tests fail
**Solution:** CUDA tests are skipped if GPU not available. This is expected.

### Issue: Config loading tests fail
**Solution:** Ensure `configs/omniscient_experiment.yaml` exists:
```bash
ls configs/omniscient_experiment.yaml
```

### Issue: Slow tests
**Solution:** Run smoke test only for quick validation:
```bash
python tests/test_omniscient_smoke.py
```

## Test Coverage Goals

- **Wrapper**: >95% coverage
- **Networks**: >90% coverage
- **Integration**: >80% coverage

## Future Extensions

Potential additional tests:
- [ ] Perception mixin interaction tests
- [ ] Comparative performance tests (omniscient vs model-free)
- [ ] Gradient magnitude tests
- [ ] Network capacity tests
- [ ] Long episode tests (100+ sessions)
- [ ] Multi-GPU tests
- [ ] Distributed training tests

## Contributing

When adding new features to omniscient RL:
1. Add unit tests to appropriate test file
2. Add integration test if needed
3. Update smoke test if it's a major feature
4. Run all tests before committing:
   ```bash
   pytest tests/test_omniscient_*.py -v
   ```
