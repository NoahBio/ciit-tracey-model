# Complementarity Visualization Guide

## Summary of Test Results

We ran 5 different tests to explore complementarity dynamics. Here's what we found:

### Test Results Overview

| Test | Pattern | Entropy | Success Rate | Complementarity Pattern |
|------|---------|---------|--------------|------------------------|
| 1 | cold_stuck | 0.1 | 70% | **Flat 100%** - Perfect complementarity throughout |
| 2 | cold_stuck | 0.1 (trial 426 params) | 100% | **Flat 100%** - Perfect complementarity |
| 3 | cold_stuck | 3.0 (high) | 93.3% | **85-90%** mid-sessions - Shows seeding phase |
| 4 | conflictual | 2.0 | 93.3% | **40-80%** mid-sessions - Dramatic seeding |
| 5 | dominant_stuck | 2.5 | 93.3% | **80-85%** mid-sessions - Moderate seeding |

### Key Findings

**1. Entropy is the Key Variable**
- **Low entropy (0.1)**: Deterministic client behavior → Therapist maintains 100% complementarity
- **Medium entropy (2.0-2.5)**: More random client → Therapist must strategically seed, dropping to 80-90%
- **High entropy (3.0)**: Very random client → More seeding needed, but still achieves high success

**2. Complementarity Phases (Visible with Higher Entropy)**
- **Phase 1 (Sessions 1-20):** Relationship building - High complementarity (~100%)
- **Phase 2 (Sessions 20-80):** Ladder climbing/Seeding - **Complementarity drops** as therapist seeds strategic non-complementary actions
- **Phase 3 (Sessions 80+):** Consolidation - Complementarity recovers toward 100%

**3. Pattern-Specific Differences**
- **cold_stuck**: Moderate seeding needed
- **conflictual**: **Most dramatic** complementarity drops (40-80%) - therapist working hard to break conflict cycle
- **dominant_stuck**: Moderate seeding (80-85%)

**4. The Strategic Trade-off**
The omniscient therapist v2 is making a calculated trade-off:
- Sacrifices short-term complementarity (during seeding phase)
- To enable better long-term outcomes (higher success rates)
- This is only visible when client behavior has sufficient randomness (entropy ≥ 2.0)

## How to Run the Visualization

### Basic Usage

```bash
source venv/bin/activate

python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier \
  --patterns cold_stuck \
  --therapist-versions v2 \
  --n-seeds 20 \
  --complementarity-type enacted
```

### To See Complementarity Variation

Use **higher entropy** (2.0-3.0) to see the therapist's strategic seeding behavior:

```bash
python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier \
  --patterns conflictual \
  --therapist-versions v2 \
  --n-seeds 20 \
  --entropy 2.5 \
  --complementarity-type enacted
```

### Compare Multiple Configurations

```bash
python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier conditional_amplifier \
  --patterns cold_stuck conflictual \
  --therapist-versions v2 \
  --n-seeds 20 \
  --entropy 2.0 \
  --complementarity-type enacted
```

## Interactive Visualization

The script includes **interactive filtering** via radio buttons, but it requires an interactive matplotlib backend.

### How to Enable Interactive Mode

**Option 1: Run with X11 Display (if available)**
```bash
export DISPLAY=:0  # Or your display number
python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier \
  --patterns cold_stuck \
  --n-seeds 20 \
  --entropy 2.0
```

**Option 2: Use Jupyter Notebook**
```python
# In a Jupyter notebook with %matplotlib widget or %matplotlib notebook

import sys
sys.path.insert(0, '/home/noah-desktop/GIT_Repos/ciit-tracey-model')

from scripts.visualize_complementarity import *

# Run visualization
# ... (code to run simulation and create viz)
viz.show()  # Will be interactive in notebook
```

**Option 3: Modify Script to Use TkAgg Backend**
Add this to the top of `visualize_complementarity.py` (after imports):
```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you have PyQt5
import matplotlib.pyplot as plt
```

Then run:
```bash
python scripts/visualize_complementarity.py --mechanisms frequency_amplifier --patterns cold_stuck --n-seeds 20 --entropy 2.0
```

### Interactive Features

When running interactively, you can:

1. **Filter by Warm/Cold**: Click radio buttons on the left to switch between:
   - **Overall**: All complementarity (default)
   - **Warm Only**: Only complementarity when client shows warm behaviors (octants 1-3)
   - **Cold Only**: Only complementarity when client shows cold behaviors (octants 5-7)

2. **Zoom and Pan**: Use matplotlib's built-in zoom and pan tools

3. **Save Individual Views**: Click the save button in matplotlib toolbar

## Output Structure

Each run creates a **timestamped directory** with all results:

```
results/complementarity_YYYYMMDD_HHMMSS/
├── complementarity_plot.png    # The visualization
└── command.json                 # Exact command and parameters used
```

### Example command.json

```json
{
  "timestamp": "20260109_143346",
  "command": "scripts/visualize_complementarity.py --mechanisms frequency_amplifier ...",
  "parameters": {
    "mechanisms": ["frequency_amplifier"],
    "patterns": ["cold_stuck"],
    "therapist_versions": ["v2"],
    "n_seeds": 15,
    "window_size": 10,
    "entropy": 3.0,
    "max_sessions": 100,
    ...
  }
}
```

This allows you to:
- **Reproduce** any visualization exactly
- **Track** which parameters generated which results
- **Compare** different runs systematically

## Recommended Test Configurations

### To See Strategic Seeding Behavior

```bash
# Conflictual pattern shows the most dramatic complementarity variation
python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier \
  --patterns conflictual \
  --therapist-versions v2 \
  --n-seeds 30 \
  --entropy 2.0 \
  --window-size 10 \
  --max-sessions 100
```

### To Compare v1 vs v2 Therapist

```bash
python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier \
  --patterns cold_stuck \
  --therapist-versions v1 v2 \
  --n-seeds 30 \
  --entropy 2.0
```

### To Compare Client Mechanisms

```bash
python scripts/visualize_complementarity.py \
  --mechanisms frequency_amplifier conditional_amplifier bond_weighted_conditional_amplifier \
  --patterns cold_stuck \
  --therapist-versions v2 \
  --n-seeds 30 \
  --entropy 2.0
```

## Understanding the Plots

### Top Panel: Complementarity Over Time

- **Y-axis**: Complementarity Rate (0-100%)
- **X-axis**: Session Number
- **Line**: Mean complementarity across all seeds
- **Shaded area**: ± 1 standard deviation (confidence band)
- **Green dotted line**: 100% perfect complementarity reference

**What to look for:**
- **Flat at 100%**: Perfect complementarity (happens with low entropy)
- **Drops during middle sessions**: Strategic seeding phase (visible with higher entropy)
- **Wide confidence bands**: High variance across seeds (more challenging scenarios)

### Bottom Panel: Success Rate

- Bar chart showing % of runs that reached the RS threshold
- Color-coded to match the line in the top panel
- Percentage label on each bar

## Technical Notes

### Why We See 100% Complementarity by Default

With **low entropy** (default 0.1), clients are very deterministic:
- They strongly prefer actions based on their utility matrix
- The omniscient therapist can predict their behavior precisely
- This allows 100% complementarity

### Why Higher Entropy Shows Variation

With **higher entropy** (2.0+), clients are more random:
- They explore more actions, even suboptimal ones
- The therapist must balance complementarity with strategic seeding
- During "ladder climbing" phase, therapist seeds specific non-complementary actions
- This temporarily reduces complementarity but enables better long-term outcomes

### Window Size Impact

- **Small window (5)**: More sensitive to recent changes, noisier
- **Medium window (10)**: Good balance (default)
- **Large window (20)**: Smoother but less responsive to phase transitions

## Troubleshooting

### "This figure includes Axes that are not compatible with tight_layout"

This warning is harmless - it's due to the radio button widgets. The plots still render correctly.

### "FigureCanvasAgg is non-interactive, and thus cannot be shown"

This means matplotlib is using the non-interactive Agg backend (for saving files). This is expected when running from command line without a display. The plot is still saved correctly to the output directory.

### "Mean of empty slice" warnings

These warnings occur when calculating warm/cold complementarity for sessions where no warm/cold actions occurred. This is expected and handled correctly (returns NaN for those data points).

## Citation

If you use this visualization in publications, please cite both the complementarity visualization tool and the underlying therapy simulation model.
