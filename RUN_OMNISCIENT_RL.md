# Running Omniscient RL Training

## Quick Start

```bash
# Run training (1M timesteps = ~10-15 minutes)
python -m src.training.train_ppo --config configs/omniscient_RL_vs_Complementary.yaml

# Monitor training
tensorboard --logdir logs/omniscient_RL_vs_Complementary
```

## What is Omniscient RL?

The omniscient RL agent has **perfect information** about the client's internal state:
- ✅ Client's utility matrix (u_matrix) - what outcomes they prefer
- ✅ Relationship satisfaction (RS) - how satisfied they are
- ✅ Bond level - therapeutic alliance strength
- ✅ Entropy - how predictable/random their behavior is
- ✅ Mechanism type - which expectation mechanism they use
- ✅ Actual vs perceived actions - when perception distortion occurs

**Why train this?**
- Upper bound: Shows best possible performance with perfect information
- Faster training: Should converge in 5-10M steps vs 50M for model-free
- Comparison: Benchmark against complementary strategy and model-free RL

## Configuration

**Environment** (same as RL_vs_Complementary):
- Pattern: `cold_stuck` (client starts cold and gets stuck)
- Mechanism: `frequency_amplifier` (client adapts based on action frequency)
- Threshold: 0.8 (80th percentile - hard task)
- Bond alpha: 5.0 (high bond influence)
- Perception: 50% accuracy baseline

**Key Differences from Model-Free**:
| Parameter | Model-Free | Omniscient | Why |
|-----------|------------|------------|-----|
| Observation dims | 417 | 471 (+54) | +internal state info |
| Timesteps | 50M | 1-10M | Should learn faster |
| Learning rate | 0.0003 | 0.0001 | Lower for complex input |
| Batch size | 64 | 128 | Larger for rich features |

## Training Duration Options

Edit `configs/omniscient_RL_vs_Complementary.yaml` to choose:

```yaml
# OPTION 1: Quick test (default)
total_timesteps: 1000000      # ~10-15 min - verify it works

# OPTION 2: Short training
# total_timesteps: 5000000    # ~90 min - should show convergence

# OPTION 3: Full training
# total_timesteps: 10000000   # ~3 hours - if 5M not enough
```

## Monitoring Training

### Tensorboard
```bash
tensorboard --logdir logs/omniscient_RL_vs_Complementary
```

**What to watch:**
- **Reward**: Should increase over time
- **Success rate**: Percentage of episodes where RS > threshold
- **Loss/entropy**: Should decrease as policy improves

### Checkpoints
Saved to `logs/omniscient_RL_vs_Complementary/`:
- `best_policy.pth` - Best performing policy
- Checkpoints every 10K steps

## Expected Results

**Baselines:**
- Random: ~1-5% success rate
- Complementary: 18% success rate (on this hard task)
- Model-Free RL: TBD (currently training with 50M timesteps)

**Omniscient RL:**
- Should beat complementary (18%)
- Should learn faster than model-free
- Upper bound on what's achievable with this environment

## Why Might This Be Faster?

**Hypothesis: Omniscient learns faster despite more dimensions**

❌ **More compute per step** (20-30% slower):
- +54 observation dimensions (471 vs 417)
- Larger batch size (128 vs 64)
- More network parameters

✅ **But fewer steps to convergence** (5-10x fewer):
- No need to infer u_matrix from behavior
- No need to learn bond dynamics
- Explicit mechanism type tells agent how client responds
- Can learn optimal strategy directly

**Net result: ~4x faster overall** (fewer steps × slower steps = faster total time)

## Troubleshooting

### Config won't load
```bash
# Test config loading
python -c "from src.training.config import load_config; config = load_config('configs/omniscient_RL_vs_Complementary.yaml'); print('✓ Config OK')"
```

### Training fails
```bash
# Run setup test
python tests/test_omniscient_smoke.py
```

### Check omniscient wrapper
```bash
# Test wrapper
python -m src.environment.omniscient_wrapper
```

## Comparison with Other Approaches

After training, compare results:

```bash
# Run baseline comparison (future work)
python -m src.evaluation.baseline_comparison \
    --policy-path logs/omniscient_RL_vs_Complementary/best_policy.pth \
    --config configs/omniscient_RL_vs_Complementary.yaml \
    --n-episodes 1000
```

## Files

**Config:** `configs/omniscient_RL_vs_Complementary.yaml`
**Wrapper:** `src/environment/omniscient_wrapper.py`
**Networks:** `src/training/omniscient_networks.py`
**Training:** `src/training/train_ppo.py` (supports omniscient automatically)
**Tests:** `tests/test_omniscient_*.py`

## Technical Details

**Observation Space (471 dims):**
- Base (417): client_action (16), session_number (1), history (400)
- +U-matrix (32): Compressed from 64 to 32 via MLP
- +RS (1): Relationship satisfaction normalized to [-1, 1]
- +Bond (1): Bond level in [0, 1]
- +Entropy (1): Temperature parameter normalized to [-1, 1]
- +Mechanism (8): Embedded mechanism type (5 types)
- +Perception (11): Actual/perceived actions, misperception rate

**Network Architecture:**
- Feature extraction: OmniscientTherapyNet
- Hidden layers: 2 × 256 (same as model-free)
- Actor: Policy network (8 action logits)
- Critic: Value network (state value)
- Total params: ~195K actor, ~193K critic

## Next Steps

1. **Run quick test** (1M steps, ~15 min)
2. **Check if learning** (reward increasing?)
3. **If yes:** Extend to 5M or 10M steps
4. **If no:** Debug or adjust hyperparameters
5. **Compare:** vs complementary baseline (18%)
