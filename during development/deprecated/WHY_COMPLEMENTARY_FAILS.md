# Why Does the Simple Always-Complementary Strategy Fail?

## The Surprising Problem

A simple always-complementary therapist achieves **95.2% perfect success** across all conditions. But under specific parameter combinations, success drops to **63-77%**. Why?

---

## The Key Finding: History Creates Rigidity

### Parameters That Create Outcome Variation:

**1. History Weight (Most Critical)**
```
history_weight = 0.1:  0.0% outcome variation  → 100% perfect success
history_weight = 1.0:  4.6% outcome variation  →  99.6% mean success
history_weight = 5.0:  9.7% outcome variation  →  97.8% mean success
```

**Insight**: Strong history influence (5.0) creates **9.7% failure rate** vs 0% with weak history (0.1)

**2. Mechanism Vulnerability**
```
bond_only:                           0.0% variation → 100% perfect success
conditional_amplifier:               4.6% variation →  99.1% mean success
bond_weighted_conditional_amplifier: 4.6% variation →  99.2% mean success
frequency_amplifier:                 9.9% variation →  98.2% mean success
```

**Insight**: History-tracking mechanisms fail more; frequency_amplifier is worst

**3. Initial Memory Pattern**
```
cw_50_50:              0.3% variation →  99.99% mean success
mixed_random:          0.9% variation →  99.97% mean success
conflictual:           2.8% variation →  99.43% mean success
complementary_perfect: 15.1% variation →  97.08% mean success
```

**Insight**: "Perfect" complementary history creates **15.1% failure rate** (50x worse than cw_50_50!)

**4. Success Threshold**
```
threshold = 0.4:  2.2% variation →  99.83% mean success
threshold = 0.6:  2.2% variation →  99.83% mean success
threshold = 0.8: 10.0% variation →  97.70% mean success
```

**Insight**: Strict threshold (80th percentile) creates 4.5x more variation

**5. Entropy (Surprisingly Small Effect)**
```
entropy = 0.3:  5.9% variation
entropy = 1.0:  4.3% variation
entropy = 3.0:  4.1% variation
```

**Insight**: Low entropy slightly increases variation, but effect is modest

**6. Bond Alpha**
```
bond_alpha = 2.0:  8.3% variation
bond_alpha = 5.0:  3.0% variation
bond_alpha = 10.0: 3.0% variation
```

**Insight**: Gentle RS→bond mapping (2.0) creates 2.8x more variation

---

## The Worst Case: 63% Success Rate

**Combination:**
```
mechanism: frequency_amplifier
entropy: 0.3
history_weight: 5.0
initial_memory: complementary_perfect
threshold: 0.8
```

**Why this fails:**

### The Mechanism of Failure:

```
Step 1: Perfect complementary memory (all D→S)
  ↓
Step 2: frequency_amplifier learns P(therapist=S) ≈ 100%
  ↓
Step 3: High history_weight (5.0) makes this expectation dominate
  ↓
Step 4: Low entropy (0.3) makes action selection deterministic
  ↓
Step 5: Client gets LOCKED into action 0 (D)
  ↓
Step 6: Even though therapist complements (D→S), if action 0
        is suboptimal for THIS client's utility matrix...
  ↓
Step 7: RS stagnates or improves slowly
  ↓
Step 8: Strict threshold (80th percentile) not reached
  ↓
Result: FAILURE (37% of trials fail!)
```

---

## Why Each Parameter Matters

### 1. History Weight: The Primary Culprit

**Low history_weight (0.1-1.0):**
- Client relies mostly on current utility estimates
- Can adapt to individual preferences
- Bond modulates expectations appropriately

**High history_weight (5.0):**
- Past experience dominates current utility
- Client becomes rigid, locked into historical patterns
- Can't adapt even when therapist is perfectly complementary

**Mechanism:**
```
Expected utility = base_utility + (history_amplification * history_weight)

With history_weight = 5.0:
  history_amplification can be 5x larger than base utility
  → Past overwhelms present
  → Client stuck in historical patterns
```

### 2. Mechanism: How History Locks In

**bond_only (0% failures):**
- No history tracking at all
- Pure utility + bond optimization
- Always finds individually optimal actions
- **Why it works:** No historical bias can lock client in

**conditional_amplifier (4.6% failures):**
- Tracks P(therapist_j | client_i)
- With perfect complementary memory: P(S|D) = 1.0
- Amplifies expectation that "when I do D, therapist does S"
- **Why it fails:** If D is not client's optimal action, they're stuck

**frequency_amplifier (9.9% failures - WORST):**
- Tracks only P(therapist_j), ignoring client action
- With perfect D→S memory: P(S) = 1.0
- Expects therapist to always do S
- **Why it fails most:**
  - Expectation doesn't match client's actual action choices
  - If client should shift to different actions, expectation is wrong
  - Creates persistent mismatch between expected and actual

### 3. Memory Pattern: The Paradox of Perfection

**Why complementary_perfect (D→S repeated) fails 15.1% of the time:**

1. **Strong prior**: All 50 memory slots identical → very confident expectation
2. **No exploration incentive**: High initial bond (0.95+) → no pressure to change
3. **Action lock-in**: Conditional mechanisms learn "always do D"
4. **Individual differences ignored**: D→S is complementary in theory, but:
   - Client's utility matrix is individual
   - Action 0 (D) may not be their highest-utility action
   - But history locks them into it anyway

**Why cw_50_50 (C→W anticomplementary) succeeds 99.7% of the time:**

1. **Weaker prior**: C→W is anticomplementary, not learned as "optimal"
2. **Low initial bond**: Forces exploration and adaptation
3. **Exploration encouraged**: Client tries different actions
4. **Finds individual optimum**: Discovers what actually works for their utility matrix

**The Paradox:**
```
Perfect complementary history → Learning lock → Can't adapt → 15% failure
Imperfect anticomplementary history → Exploration → Adaptation → 0.3% failure
```

### 4. Success Threshold: Strictness Matters

**Why threshold 0.8 creates 10% variation:**

The threshold represents percentile of client's achievable RS range.

```
threshold = 0.5 (median): Client must reach middle of their RS range
  → Easy to achieve with complementary therapy
  → 2.2% variation

threshold = 0.8 (80th percentile): Client must reach top 20% of range
  → Requires finding near-optimal actions
  → If locked into suboptimal action, threshold unreachable
  → 10% variation (4.5x more!)
```

**Example:**
- Client's RS range: [-70, 65]
- threshold 0.5 → must reach RS = -2.5 (very achievable)
- threshold 0.8 → must reach RS = 38.0 (requires optimal actions)

If history locks client into action with RS = 32:
- threshold 0.5: SUCCESS
- threshold 0.8: FAILURE

### 5. Entropy: Determinism vs Exploration

**Why low entropy (0.3) creates more variation:**

Low entropy → deterministic action selection
- Client almost always picks highest-expected-utility action
- If history creates wrong expectations, client is stuck
- No random exploration to escape local optimum

High entropy (3.0) → stochastic selection
- Client sometimes picks suboptimal actions by chance
- Random exploration can escape historical lock-in
- Averages out individual mistakes

**But effect is modest (5.9% vs 4.1%)** because:
- Complementary therapy is robust across action choices
- Most actions eventually lead to improvement
- History weight is more important than exploration

### 6. Bond Alpha: Optimism Calibration

**Why low bond_alpha (2.0) creates more variation:**

Bond alpha controls RS → bond sigmoid steepness.

```
Low alpha (2.0): Gentle slope
  → Small RS changes → small bond changes
  → Client stays pessimistic longer
  → Explores less optimistically
  → May not find optimal actions quickly

High alpha (10.0): Steep slope
  → Small RS changes → large bond changes
  → Client becomes optimistic quickly
  → Explores more broadly
  → Finds optimal actions faster
```

---

## Summary: Why Simple Complementarity Isn't Always Enough

### The Core Insight:

**Complementary responses work when clients can adapt their actions to their individual preferences.**

But when:
1. **History dominates** (high history_weight) AND
2. **History is "perfect"** (complementary_perfect memory) AND
3. **Client can't explore** (low entropy) AND
4. **Success is strict** (high threshold)

Then:
- Client gets locked into historically-learned actions
- Even though therapist complements perfectly
- Those actions may not be optimal for this individual client
- Client reaches local optimum, not global optimum
- Strict threshold exposes this gap

### The Frequency Amplifier Problem:

frequency_amplifier is worst because:
- Ignores conditioning on client action
- Forms global expectations about therapist
- With perfect D→S history: expects therapist mostly does S
- But when client (should) shift to different actions
- Expectation is violated
- Creates persistent confusion
- Prevents optimal learning

### The Complementary Perfect Paradox:

Starting with "perfect" complementary relationship:
- Should be ideal (high bond, good history)
- Actually creates rigidity (no exploration, strong lock-in)
- Prevents individualization
- **Imperfect starts allow adaptation → better outcomes**

---

## Implications for Therapy

1. **History can trap**: Strong reliance on past patterns prevents adaptation
2. **Perfection prevents growth**: Too-good initial relationship removes exploration incentive
3. **Individual differences matter**: One-size-fits-all complementarity misses 5-37% of clients
4. **Strict outcomes expose gaps**: Lenient success criteria hide inadequate adaptation

The always-complementary strategy is **good but not perfect** precisely because it doesn't account for individual client differences in action preferences.
