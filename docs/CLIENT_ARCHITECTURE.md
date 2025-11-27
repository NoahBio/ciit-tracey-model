# CIIT-Tracey Client Architecture: Complete Guide

## Overview: The Big Picture

Your client architecture is designed to model therapy clients who:
1. **Remember** past interactions with therapists (memory system)
2. **Feel** relationship satisfaction based on those interactions (RS calculation)
3. **Trust** the therapist based on satisfaction (bond calculation)
4. **Expect** certain therapist responses (expectation mechanisms - this is where the 5 client types differ)
5. **Choose** actions that maximize expected rewards (action selection)
6. **Optionally misperceive** therapist actions based on their history (perception system)

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   PERCEPTUAL LAYER                          │
│              (perceptual_distortion.py)                     │
│  - Optional: Distorts therapist actions before memory       │
│  - Can be added to ANY client type via mixing              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    BASE CLIENT LAYER                        │
│                  (base_client.py)                           │
│  - Memory management                                        │
│  - Relationship satisfaction calculation                    │
│  - Bond calculation                                         │
│  - Action selection (softmax)                              │
│  - Dropout checking                                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             EXPECTATION MECHANISM LAYER                     │
│              (5 different client types)                     │
│  - Each implements _calculate_expected_payoffs()            │
│  - Different ways of predicting therapist responses         │
└─────────────────────────────────────────────────────────────┘
```

---

## Calculation Flow Diagrams

### Complete Session Flow: Frequency-Amplifier Client with Perception

This diagram shows EVERYTHING that happens in one therapy session:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SESSION START (Session N)                       │
│                                                                         │
│  Current Client State:                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ • memory: deque(50 interactions)                                 │ │
│  │   [(c₀,t₀), (c₁,t₁), ..., (c₄₉,t₄₉)]                           │ │
│  │ • relationship_satisfaction: 35.2                                │ │
│  │ • bond: 0.67                                                     │ │
│  │ • session_count: 23                                              │ │
│  │ • entropy: 3.0                                                   │ │
│  │ • u_matrix: 8×8 utility values                                   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   STEP 1: CLIENT SELECTS ACTION                         │
│                  (Therapist hasn't acted yet!)                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
            ┌───────────────────────────────────────┐
            │  client.select_action()               │
            └───────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 1A: CALCULATE EXPECTED PAYOFFS                             │
│         client._calculate_expected_payoffs()                            │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 1A.1: Calculate Therapist Frequencies                 │   │
│  │ _calculate_marginal_frequencies()                              │   │
│  │                                                                 │   │
│  │  Input: memory (50 interactions)                               │   │
│  │  ┌──────────────────────────────────────────────────────────┐ │   │
│  │  │ for each interaction (c, t) in memory:                   │ │   │
│  │  │   weighted_counts[t] += memory_weight[idx]               │ │   │
│  │  │                                                           │ │   │
│  │  │ memory_weights = get_memory_weights(50)                  │ │   │
│  │  │   [w₀=0.015, w₁=0.016, ..., w₄₉=0.030]                  │ │   │
│  │  │   ↑ Recent interactions have higher weights              │ │   │
│  │  │                                                           │ │   │
│  │  │ frequencies = weighted_counts / sum(memory_weights)      │ │   │
│  │  └──────────────────────────────────────────────────────────┘ │   │
│  │                                                                 │   │
│  │  Output: therapist_frequencies                                 │   │
│  │  [0.05, 0.08, 0.45, 0.28, 0.04, 0.02, 0.03, 0.05]             │   │
│  │   D     WD    W     WS    S     CS    C     CD                 │   │
│  │         ↑           ↑                                           │   │
│  │   Therapist has been mostly Warm (45%) and Warm-Sub (28%)      │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 1A.2: Calculate Expected Payoffs for Each Action     │   │
│  │                                                                 │   │
│  │  for client_action in [0, 1, 2, 3, 4, 5, 6, 7]:               │   │
│  │    ┌─────────────────────────────────────────────────────┐    │   │
│  │    │ Get raw utilities from u_matrix                     │    │   │
│  │    │ raw_utilities = u_matrix[client_action, :]          │    │   │
│  │    │                                                      │    │   │
│  │    │ Example (client_action = 2, Warm):                  │    │   │
│  │    │ raw = [+10, +50, +60, +50, +10, -50, -40, -50]     │    │   │
│  │    │        D    WD   W    WS   S    CS   C    CD        │    │   │
│  │    └─────────────────────────────────────────────────────┘    │   │
│  │                          ↓                                     │   │
│  │    ┌─────────────────────────────────────────────────────┐    │   │
│  │    │ Amplify utilities based on frequency                │    │   │
│  │    │ adjusted = raw + (raw × frequencies × history_weight)│   │   │
│  │    │                                                      │    │   │
│  │    │ history_weight = 1.0 (from config)                  │    │   │
│  │    │                                                      │    │   │
│  │    │ Example calculations:                               │    │   │
│  │    │ adjusted[2] = +60 + (+60 × 0.45 × 1.0) = +87       │    │   │
│  │    │ adjusted[3] = +50 + (+50 × 0.28 × 1.0) = +64       │    │   │
│  │    │ adjusted[1] = +50 + (+50 × 0.08 × 1.0) = +54       │    │   │
│  │    │ adjusted[0] = +10 + (+10 × 0.05 × 1.0) = +10.5     │    │   │
│  │    │ adjusted[5] = -50 + (-50 × 0.02 × 1.0) = -51       │    │   │
│  │    │                                                      │    │   │
│  │    │ Frequent positive outcomes → BOOSTED ↑              │    │   │
│  │    │ Rare outcomes → stays near baseline                 │    │   │
│  │    └─────────────────────────────────────────────────────┘    │   │
│  │                          ↓                                     │   │
│  │    ┌─────────────────────────────────────────────────────┐    │   │
│  │    │ Sort adjusted utilities                             │    │   │
│  │    │ sorted_adjusted = sort(adjusted)                    │    │   │
│  │    │                                                      │    │   │
│  │    │ [-51, -50, -40, +10.5, +54, +64, +87, +87]         │    │   │
│  │    │   ↑0   ↑1   ↑2   ↑3    ↑4   ↑5   ↑6   ↑7           │    │   │
│  │    └─────────────────────────────────────────────────────┘    │   │
│  │                          ↓                                     │   │
│  │    ┌─────────────────────────────────────────────────────┐    │   │
│  │    │ Apply bond-based percentile selection               │    │   │
│  │    │                                                      │    │   │
│  │    │ bond = 0.67                                         │    │   │
│  │    │ position = 0.67 × 7 = 4.69                          │    │   │
│  │    │                                                      │    │   │
│  │    │ lower_idx = floor(4.69) = 4                         │    │   │
│  │    │ upper_idx = 5                                       │    │   │
│  │    │ interpolation_weight = 4.69 - 4 = 0.69              │    │   │
│  │    │                                                      │    │   │
│  │    │ expected_payoff = (1 - 0.69) × sorted[4] +          │    │   │
│  │    │                   0.69 × sorted[5]                  │    │   │
│  │    │                 = 0.31 × 54 + 0.69 × 64             │    │   │
│  │    │                 = 16.74 + 44.16 = 60.9              │    │   │
│  │    │                                                      │    │   │
│  │    │ High bond → expects upper percentile outcomes       │    │   │
│  │    └─────────────────────────────────────────────────────┘    │   │
│  │                                                                 │   │
│  │  End loop                                                       │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: expected_payoffs (8 values, one per client action)           │
│  [12.3, 45.6, 60.9, 38.9, 15.2, -22.4, -18.7, 8.3]                    │
│   D     WD    W     WS    S     CS     C     CD                        │
│         ↑     ↑                                                         │
│   Action 2 (Warm) has highest expected payoff!                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 1B: CONVERT TO PROBABILITIES (SOFTMAX)                     │
│         client._softmax(expected_payoffs)                               │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Input: expected_payoffs, entropy = 3.0                         │   │
│  │                                                                 │   │
│  │ Step 1: Scale by temperature                                   │   │
│  │   scaled = payoffs / entropy                                   │   │
│  │   [4.1, 15.2, 20.3, 13.0, 5.1, -7.5, -6.2, 2.8]               │   │
│  │                                                                 │   │
│  │ Step 2: Subtract max for numerical stability                   │   │
│  │   scaled = scaled - max(scaled)                                │   │
│  │   [-16.2, -5.1, 0.0, -7.3, -15.2, -27.8, -26.5, -17.5]        │   │
│  │                                                                 │   │
│  │ Step 3: Exponentiate                                           │   │
│  │   exp_values = exp(scaled)                                     │   │
│  │   [0.000001, 0.006, 1.0, 0.0007, 0.000002, ~0, ~0, 0.00002]   │   │
│  │                                                                 │   │
│  │ Step 4: Normalize to probabilities                             │   │
│  │   probabilities = exp_values / sum(exp_values)                 │   │
│  │   [0.0001, 0.006, 0.993, 0.0007, 0.0002, ~0, ~0, 0.00002]     │   │
│  │    D      WD     W      WS      S     CS   C    CD             │   │
│  │                  ↑                                              │   │
│  │   Warm action dominates! (99.3% probability)                   │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: probabilities summing to 1.0                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 1C: SAMPLE ACTION                                          │
│                                                                         │
│  client_action = rng.choice([0,1,2,3,4,5,6,7], p=probabilities)       │
│                                                                         │
│  Given probabilities, almost certainly selects: 2 (Warm)               │
│                                                                         │
│  Selected client_action: 2                                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                   STEP 2: THERAPIST ACTS                                │
│                                                                         │
│  (Therapist agent makes its own decision based on its observations)    │
│                                                                         │
│  Therapist selects action: 2 (Warm)                                    │
│                                                                         │
│  actual_therapist_action = 2                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 3: CLIENT PERCEIVES THERAPIST ACTION                       │
│         client._perceive_therapist_action(actual_action=2)              │
│         (Only happens if using PerceptualClientMixin)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 3A: Get Recent Memory Window                          │   │
│  │                                                                 │   │
│  │  recent_memory = last 15 interactions from memory              │   │
│  │  [(c₃₅,t₃₅), (c₃₆,t₃₆), ..., (c₄₉,t₄₉)]                      │   │
│  │                                                                 │   │
│  │  Extract therapist actions: [t₃₅, t₃₆, ..., t₄₉]             │   │
│  │  Count frequencies:                                            │   │
│  │    Action 0 (D):  1 time  → 0.067                             │   │
│  │    Action 1 (WD): 2 times → 0.133                             │   │
│  │    Action 2 (W):  7 times → 0.467                             │   │
│  │    Action 3 (WS): 4 times → 0.267                             │   │
│  │    Action 4 (S):  1 time  → 0.067                             │   │
│  │    Others: 0                                                   │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 3B: STAGE 1 - History-Based Perception                │   │
│  │                                                                 │   │
│  │  actual_action = 2 (W)                                         │   │
│  │  baseline_accuracy = 0.2 (20%)                                 │   │
│  │                                                                 │   │
│  │  Roll 1: Baseline path check                                   │   │
│  │    if random() < 0.2:                                          │   │
│  │      → perceive correctly (stage1_result = 2)                  │   │
│  │                                                                 │   │
│  │  Let's say random() = 0.73 → baseline path FAILS               │   │
│  │                                                                 │   │
│  │  Roll 2: Frequency-based accuracy check                        │   │
│  │    computed_accuracy = frequency[2] = 0.467                    │   │
│  │    if random() < 0.467:                                        │   │
│  │      → perceive correctly (stage1_result = 2)                  │   │
│  │    else:                                                        │   │
│  │      → misperceive: sample from frequency distribution         │   │
│  │                                                                 │   │
│  │  Let's say random() = 0.35 < 0.467 → SUCCESS                   │   │
│  │  stage1_result = 2 (perceived correctly!)                      │   │
│  │                                                                 │   │
│  │  Intuition: Since therapist has been Warm 47% of recent time,  │   │
│  │  it's relatively easy to perceive Warm actions correctly.      │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 3C: STAGE 2 - Adjacency Noise                         │   │
│  │                                                                 │   │
│  │  PERCEPTION_ADJACENCY_NOISE = 0.1 (10% chance)                 │   │
│  │                                                                 │   │
│  │  Roll 3: Adjacency shift check                                 │   │
│  │    if random() < 0.1:                                          │   │
│  │      shift = random_choice([-1, +1])                           │   │
│  │      perceived_action = (stage1_result + shift) % 8            │   │
│  │                                                                 │   │
│  │  Let's say random() = 0.87 > 0.1 → NO SHIFT                    │   │
│  │  perceived_action = stage1_result = 2                          │   │
│  │                                                                 │   │
│  │  (If shift had occurred with stage1_result=2:                  │   │
│  │   shift=-1 → perceived=1 (WD), or shift=+1 → perceived=3 (WS)) │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 3D: Create Perception Record                          │   │
│  │                                                                 │   │
│  │  record = PerceptionRecord(                                    │   │
│  │    client_action = 2,                                          │   │
│  │    actual_therapist_action = 2,                                │   │
│  │    perceived_therapist_action = 2,                             │   │
│  │    stage1_result = 2,                                          │   │
│  │    baseline_path_succeeded = False,                            │   │
│  │    stage1_changed_from_actual = False,                         │   │
│  │    stage2_shifted = False,                                     │   │
│  │    computed_accuracy = 0.467                                   │   │
│  │  )                                                              │   │
│  │                                                                 │   │
│  │  perception_history.append(record)                             │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: perceived_action = 2 (matched actual in this case)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 4: UPDATE MEMORY                                           │
│         client.update_memory(client_action=2, therapist_action=2)       │
│         (Perceptual version stores perceived, not actual)               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 4A: Add to Memory                                     │   │
│  │                                                                 │   │
│  │  Before:                                                        │   │
│  │  memory = deque([(c₀,t₀), ..., (c₄₉,t₄₉)], maxlen=50)        │   │
│  │                                                                 │   │
│  │  memory.append((client_action=2, perceived_action=2))          │   │
│  │                                                                 │   │
│  │  After:                                                         │   │
│  │  memory = deque([(c₁,t₁), ..., (c₄₉,t₄₉), (2,2)], maxlen=50) │   │
│  │                 ↑ (c₀,t₀) was automatically dropped            │   │
│  │                                                                 │   │
│  │  Oldest interaction falls off, newest added                    │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 4B: Increment Session Count                           │   │
│  │                                                                 │   │
│  │  session_count: 23 → 24                                        │   │
│  └────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 5: RECALCULATE RELATIONSHIP SATISFACTION                   │
│         client._calculate_relationship_satisfaction()                   │
│         (Called automatically by update_memory)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 5A: Lookup Utilities                                  │   │
│  │                                                                 │   │
│  │  for each (client_oct, therapist_oct) in memory:               │   │
│  │    utility = u_matrix[client_oct, therapist_oct]               │   │
│  │                                                                 │   │
│  │  Example for newest interaction (2, 2):                        │   │
│  │    utility = u_matrix[2, 2]  # Client W, Therapist W          │   │
│  │            = +60  (both warm → high utility!)                  │   │
│  │                                                                 │   │
│  │  utilities = [u₀, u₁, ..., u₄₈, +60]                          │   │
│  │              [38, 42, ..., 55, 60]                             │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 5B: Get Memory Weights                                │   │
│  │                                                                 │   │
│  │  weights = get_memory_weights(50)                              │   │
│  │                                                                 │   │
│  │  Formula: w[i] = 1 + sqrt(i / 49)                              │   │
│  │           normalized to sum to 1.0                             │   │
│  │                                                                 │   │
│  │  Oldest (i=0):  w₀ = 1.0      → normalized: 0.0157            │   │
│  │  Middle (i=25): w₂₅ = 1.707   → normalized: 0.0268            │   │
│  │  Newest (i=49): w₄₉ = 2.0     → normalized: 0.0314            │   │
│  │                                                                 │   │
│  │  Newest memory has 2× weight of oldest (2.0 / 1.0)            │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 5C: Weighted Average                                  │   │
│  │                                                                 │   │
│  │  rs = Σ(utilities[i] × weights[i]) / Σ(weights)               │   │
│  │                                                                 │   │
│  │  rs = (38×0.0157 + 42×0.0160 + ... + 60×0.0314)               │   │
│  │     = 0.597 + 0.672 + ... + 1.884                              │   │
│  │     = 36.8                                                      │   │
│  │                                                                 │   │
│  │  New RS: 36.8 (was 35.2, improved by 1.6!)                     │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: relationship_satisfaction = 36.8                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 6: RECALCULATE BOND                                        │
│         client._calculate_bond()                                        │
│         (Called automatically by update_memory)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 6A: Normalize RS to [0, 1]                            │   │
│  │                                                                 │   │
│  │  rs = 36.8                                                      │   │
│  │  rs_min = -70 (minimum value in this client's u_matrix)       │   │
│  │  rs_max = +70 (maximum value in this client's u_matrix)       │   │
│  │                                                                 │   │
│  │  rs_normalized = (rs - rs_min) / (rs_max - rs_min)            │   │
│  │                = (36.8 - (-70)) / (70 - (-70))                 │   │
│  │                = 106.8 / 140                                    │   │
│  │                = 0.763                                          │   │
│  │                                                                 │   │
│  │  Client is at 76th percentile of possible satisfaction         │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 6B: Apply Offset Shift                                │   │
│  │                                                                 │   │
│  │  BOND_OFFSET = 0.8 (inflection point at 80th percentile)      │   │
│  │                                                                 │   │
│  │  rs_shifted = 2 × (rs_normalized - offset)                     │   │
│  │             = 2 × (0.763 - 0.8)                                 │   │
│  │             = 2 × (-0.037)                                      │   │
│  │             = -0.074                                            │   │
│  │                                                                 │   │
│  │  Slightly below inflection point (negative shift)              │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Sub-step 6C: Apply Sigmoid Transformation                      │   │
│  │                                                                 │   │
│  │  BOND_ALPHA = 5 (steepness parameter)                          │   │
│  │                                                                 │   │
│  │  bond = 1 / (1 + exp(-alpha × rs_shifted))                     │   │
│  │       = 1 / (1 + exp(-5 × -0.074))                             │   │
│  │       = 1 / (1 + exp(0.37))                                     │   │
│  │       = 1 / (1 + 1.448)                                         │   │
│  │       = 1 / 2.448                                               │   │
│  │       = 0.409                                                   │   │
│  │                                                                 │   │
│  │  New bond: 0.409 (was 0.67, DECREASED!)                        │   │
│  │                                                                 │   │
│  │  Why decrease if RS improved? The old RS (35.2) was also high, │   │
│  │  but the sigmoid calculation is sensitive to the position      │   │
│  │  relative to the offset. Small changes near the inflection     │   │
│  │  can cause noticeable bond changes.                            │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: bond = 0.409                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│         STEP 7: CHECK DROPOUT (Only at session 10)                      │
│         client.check_dropout()                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │  session_count = 24, dropout_checked = False                   │   │
│  │                                                                 │   │
│  │  if session_count != 10 or dropout_checked:                    │   │
│  │    return False  # No dropout                                  │   │
│  │                                                                 │   │
│  │  In this case: 24 ≠ 10, so return False immediately            │   │
│  │  (Dropout already checked at session 10)                       │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Output: False (client continues therapy)                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         SESSION END (Session N)                         │
│                                                                         │
│  Updated Client State:                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │ • memory: deque(50 interactions) - oldest dropped, (2,2) added  │ │
│  │ • relationship_satisfaction: 36.8 (was 35.2, +1.6)              │ │
│  │ • bond: 0.409 (was 0.67, -0.261)                                │ │
│  │ • session_count: 24 (was 23, +1)                                │ │
│  │ • perception_history: 24 records (new record added)             │ │
│  │ • entropy: 3.0 (unchanged)                                      │ │
│  │ • u_matrix: 8×8 utility values (unchanged)                      │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Ready for Session N+1 with updated state!                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Simplified Flow: Major Components

```
┌────────────────────┐
│   Memory (50)      │  Stores: [(client_action, therapist_action), ...]
│   [oldest ... new] │  Structure: deque with maxlen=50
└──────┬─────────────┘
       │
       ├──→ [Calculate RS] ──→ Weighted average of utilities
       │                       Recent memories weigh more
       │                       Output: scalar RS value
       │
       ├──→ [Calculate Bond] ─→ Normalize RS → Shift → Sigmoid
       │                        Output: bond ∈ [0, 1]
       │
       └──→ [Calculate Frequencies] → Count therapist actions
                                      Apply recency weights
                                      Output: 8 probabilities

                    ↓↓↓ All feed into ↓↓↓

┌────────────────────────────────────────────────────────────┐
│            EXPECTATION MECHANISM                           │
│  (Different for each client type)                          │
│                                                             │
│  Inputs: memory, bond, u_matrix, frequencies               │
│  Output: expected_payoffs[8]                               │
└─────────────────────┬──────────────────────────────────────┘
                      │
                      ↓
┌────────────────────────────────────────────────────────────┐
│                   SOFTMAX                                   │
│  payoffs → scale by entropy → exponentiate → normalize     │
│  Output: probabilities[8]                                  │
└─────────────────────┬──────────────────────────────────────┘
                      │
                      ↓
┌────────────────────────────────────────────────────────────┐
│                   SAMPLE ACTION                             │
│  Choose action according to probabilities                  │
│  Output: client_action ∈ {0,1,2,3,4,5,6,7}                │
└────────────────────────────────────────────────────────────┘
```

---

### Perception Flow Detail

```
Actual Therapist Action (Ground Truth)
              │
              ↓
┌─────────────────────────────────────────────────────────┐
│          STAGE 1: History-Based Perception              │
│                                                          │
│  Get recent memory (last 15 interactions)               │
│  Calculate frequency[0..7] for therapist actions        │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Path A: Baseline (20%)                        │    │
│  │  ─────────────────────────────────────────     │    │
│  │  if random() < 0.2:                            │    │
│  │    stage1_result = actual_action ✓             │    │
│  │                                                 │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Path B: Frequency-Based (80%)                 │    │
│  │  ─────────────────────────────────────────     │    │
│  │  accuracy = frequency[actual_action]           │    │
│  │                                                 │    │
│  │  if random() < accuracy:                       │    │
│  │    stage1_result = actual_action ✓             │    │
│  │  else:                                          │    │
│  │    stage1_result = sample(frequency) ✗         │    │
│  │    (Perceive what you EXPECT to see)           │    │
│  │                                                 │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└──────────────────┬───────────────────────────────────────┘
                   │ stage1_result
                   ↓
┌─────────────────────────────────────────────────────────┐
│          STAGE 2: Adjacency Noise                       │
│                                                          │
│  if random() < 0.1:  (10% chance)                       │
│    shift = random_choice([-1, +1])                      │
│    perceived = (stage1_result + shift) % 8              │
│  else:                                                   │
│    perceived = stage1_result                            │
│                                                          │
│  Models: Confusion between adjacent octants             │
│          W (2) ↔ WD (1) or WS (3)                       │
│                                                          │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ↓
         Perceived Action
    (Stored in memory instead of actual)
```

---

## Part 1: Base Client (`base_client.py`)

### What It Does

The `BaseClientAgent` class contains **all shared functionality** across different client types. Think of it as a template that defines the structure but leaves one key piece (expectation mechanism) for subclasses to fill in.

### Key Components

#### 1. **Initialization** (`__init__`)

**When a client is created, it needs:**

```python
def __init__(
    self,
    u_matrix: NDArray[np.float64],      # 8x8 matrix: utility of each interaction
    entropy: float,                      # How random/exploratory the client is
    initial_memory: List[Tuple[int, int]], # 50 past interactions
    random_state: Optional[int] = None   # For reproducibility
)
```

**What happens during initialization:**

1. **Validates inputs**: Memory must be exactly 50 interactions long
2. **Stores utility matrix**: This defines what this specific client finds rewarding/punishing
3. **Initializes memory**: Past interactions are stored as (client_octant, therapist_octant) pairs
4. **Calculates RS bounds**: Finds min/max possible satisfaction for normalization
5. **Calculates initial state**: Computes starting RS and bond
6. **Sets success threshold**: Determines what counts as "successful therapy" for this client

#### 2. **Memory System** (`update_memory`)

**How memory works:**

```python
def update_memory(self, client_action: int, therapist_action: int) -> None:
    # Add to memory (automatically removes oldest if full)
    self.memory.append((client_action, therapist_action))

    # Increment session count
    self.session_count += 1

    # Recalculate RS and bond based on new memory
    self.relationship_satisfaction = self._calculate_relationship_satisfaction()
    self.bond = self._calculate_bond()
```

**Key insight:** Memory is a sliding window (deque with maxlen=50). When you add the 51st interaction, the oldest one automatically falls off.

#### 3. **Relationship Satisfaction Calculation** (`_calculate_relationship_satisfaction`)

**What it calculates:** How satisfied is the client with the therapeutic relationship?

**The math:**

```python
def _calculate_relationship_satisfaction(self) -> float:
    # Step 1: Look up utility for each remembered interaction
    utilities = []
    for client_oct, therapist_oct in self.memory:
        utility = self.u_matrix[client_oct, therapist_oct]
        utilities.append(utility)

    # Step 2: Get weights (recent memories count more)
    weights = get_memory_weights(len(utilities))
    # These weights use square root recency: oldest has 50% weight of newest

    # Step 3: Weighted average
    rs = weighted_average(utilities, weights)
    return rs
```

**Example:**
- Client did action 2 (W = Warm), therapist responded 3 (WS = Warm-Submissive)
- Look up `u_matrix[2, 3]` → might be +40 (positive!)
- Do this for all 50 memories
- Recent memories weigh more heavily
- Average them → that's your RS

#### 4. **Bond Calculation** (`_calculate_bond`)

**What it calculates:** How much does the client trust/like the therapist?

**The math:**

```python
def _calculate_bond(self) -> float:
    # Step 1: Normalize RS to 0-1 range
    rs_normalized = (self.relationship_satisfaction - self.rs_min) / (self.rs_max - self.rs_min)

    # Step 2: Apply offset (shift inflection point to 80th percentile)
    rs_shifted = 2 * (rs_normalized - 0.8)

    # Step 3: Apply sigmoid function
    bond = 1.0 / (1.0 + exp(-5 * rs_shifted))

    return bond
```

**What this means:**
- RS gets normalized to 0-1 based on what's possible for this client
- The sigmoid makes bond grow slowly at first, then rapidly, then plateau
- `offset=0.8` means you need to reach 80% of your RS range to have medium bond (0.5)
- `alpha=5` controls steepness (how quickly bond changes)

**Example progression:**
- RS at 10th percentile → bond ≈ 0.05 (very low trust)
- RS at 50th percentile → bond ≈ 0.20 (still low trust)
- RS at 80th percentile → bond ≈ 0.50 (medium trust)
- RS at 95th percentile → bond ≈ 0.90 (high trust)

#### 5. **Action Selection** (`select_action`)

**What it does:** Client chooses which interpersonal behavior (octant) to perform.

**The process:**

```python
def select_action(self) -> int:
    # Step 1: Calculate expected payoffs (subclass-specific!)
    payoffs = self._calculate_expected_payoffs()  # [8 values, one per octant]

    # Step 2: Convert to probabilities using softmax
    probabilities = self._softmax(payoffs)

    # Step 3: Sample action based on probabilities
    action = self.rng.choice(8, p=probabilities)
    return action
```

**Softmax calculation:**

```python
def _softmax(self, payoffs: NDArray[np.float64]) -> NDArray[np.float64]:
    # Divide by entropy (temperature parameter)
    scaled = payoffs / self.entropy

    # Numerical stability: subtract max
    scaled = scaled - max(scaled)

    # Exponentiate
    exp_values = exp(scaled)

    # Normalize to probabilities
    probabilities = exp_values / sum(exp_values)
    return probabilities
```

**What entropy does:**
- **Low entropy (e.g., 1.5)**: Client strongly prefers best option (more deterministic)
- **High entropy (e.g., 5.0)**: Client explores more, tries different actions (more random)

**Example:**
```
Expected payoffs = [10, 20, 30, 15, 5, 8, 12, 18]
Entropy = 3.0

After softmax:
Probabilities = [0.02, 0.08, 0.54, 0.04, 0.01, 0.01, 0.03, 0.27]
                    ↑              ↑                            ↑
                    Low         Highest                   Second best

Client will probably choose action 2 (54% chance), sometimes action 7 (27%), rarely others.
```

#### 6. **Dropout Mechanism** (`check_dropout`)

**What it does:** Determines if client quits therapy.

**The logic:**

```python
def check_dropout(self) -> bool:
    # Only check ONCE, at session 10
    if self.session_count != 10 or self.dropout_checked:
        return False

    # Mark that we've checked
    self.dropout_checked = True

    # Dropout if RS has DECREASED from initial
    return self.relationship_satisfaction < self.initial_rs
```

**Psychology:** If therapy isn't helping by session 10, client leaves.

#### 7. **Static Method: Generate Problematic Memory**

This is a **helper function** (not instance-specific) to create initial memories representing clients with interpersonal problems:

```python
@staticmethod
def generate_problematic_memory(
    pattern_type: str = "cold_stuck",
    n_interactions: int = 50,
    random_state: Optional[int] = None
) -> List[Tuple[int, int]]:
```

**What it does:**
- Creates 50 past interactions showing problematic patterns
- Client stuck in certain octants (80% of time)
- Others respond anticomplementarily (causing interpersonal tension)

**Example patterns:**
- `"cold_stuck"`: Client acts cold (C, CS, CD), others respond warm (creates conflict)
- `"dominant_stuck"`: Client dominates, others also dominate (power struggles)

---

## Part 2: Perceptual Distortion (`perceptual_distortion.py`)

### What Problem Does This Solve?

In real therapy, clients don't perceive therapist actions perfectly. A therapist might be warm, but a client with a history of cold relationships might perceive it as neutral or even suspicious.

### How It Works: The Mixin Pattern

**Mixin** = a class that adds functionality to other classes through inheritance.

```python
class PerceptualClientMixin:
    """Adds imperfect perception to ANY client type."""

    def __init__(
        self,
        baseline_accuracy: float = 0.2,   # 20% always perceive correctly
        enable_perception: bool = True,    # Can turn off for control experiments
        **kwargs
    ):
        super().__init__(**kwargs)  # Pass everything to base client
        self.baseline_accuracy = baseline_accuracy
        self.enable_perception = enable_perception
        self.perception_history: List[PerceptionRecord] = []
```

### The Two-Stage Perception Process

When the therapist does action X, the client might perceive action Y:

#### **Stage 1: History-Based Perception**

```python
def _perceive_therapist_action(self, actual_action: int) -> Tuple[int, PerceptionRecord]:
    # Get last 15 interactions
    recent_memory = list(self.memory)[-15:]

    # Calculate frequency: How often has therapist done each action?
    frequency = count_therapist_actions(recent_memory)  # 8 probabilities summing to 1

    # Roll the dice!
    if random() < 0.2:  # 20% baseline path
        stage1_result = actual_action  # Perceive correctly!
    else:  # 80% history-based path
        accuracy = frequency[actual_action]  # More frequent = easier to perceive

        if random() < accuracy:
            stage1_result = actual_action  # Got it right!
        else:
            # Misperception: sample from what you expect to see
            stage1_result = sample_from_distribution(frequency)
```

**Key insight:** If therapist has been consistently warm (action 2) for 15 sessions, and suddenly acts cold (action 6), the client might *misperceive* it as warm because that's what they expect.

#### **Stage 2: Adjacency Noise**

```python
    # Stage 2: Apply random ±1 shift with 10% probability
    if random() < 0.1:
        shift = choice([-1, +1])
        perceived_action = (stage1_result + shift) % 8  # Wraps around
    else:
        perceived_action = stage1_result

    return perceived_action
```

**What this models:** Even if you generally perceive correctly, you might confuse adjacent behaviors:
- Warm-Dominant (1) might be seen as Warm (2)
- Cold (6) might be seen as Cold-Submissive (5) or Cold-Dominant (7)

### How It Integrates: Overriding `update_memory`

```python
def update_memory(self, client_action: int, therapist_action: int) -> None:
    if not self.enable_perception:
        # Perfect perception: store actual
        super().update_memory(client_action, therapist_action)
    else:
        # Imperfect perception: apply distortion
        perceived_action, record = self._perceive_therapist_action(therapist_action)

        # Store record for analysis
        self.perception_history.append(record)

        # Update memory with PERCEIVED action (client's subjective reality)
        super().update_memory(client_action, perceived_action)
```

**Critical insight:** The client's entire psychology (RS, bond, expectations) is based on *perceived* actions, not actual ones!

### The Factory Function: `with_perception`

This is a clever way to add perception to any client type:

```python
def with_perception(client_class):
    """Add perception to any client type."""
    class PerceptualClient(PerceptualClientMixin, client_class):
        pass

    PerceptualClient.__name__ = f"Perceptual{client_class.__name__}"
    return PerceptualClient
```

**Usage:**
```python
from src.agents.client_agents import BondOnlyClient, with_perception

# Create perceptual version
PerceptualBondOnly = with_perception(BondOnlyClient)

# Use it
client = PerceptualBondOnly(
    u_matrix=my_matrix,
    entropy=3.0,
    initial_memory=my_memory,
    baseline_accuracy=0.2,  # Perception parameter
    enable_perception=True   # Perception parameter
)
```

---

## Part 3: The Five Client Types (Expectation Mechanisms)

All five clients share everything from `BaseClientAgent` but differ in **one method**: `_calculate_expected_payoffs()`.

This method answers: *"If I do action i, what payoff do I expect to get?"*

### Type 1: Bond-Only Client (`bond_only_client.py`)

**Philosophy:** Expectations based purely on trust level, not history.

**The calculation:**

```python
def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
    expected_payoffs = np.zeros(8)

    for client_action in range(8):
        # Get all possible utilities for this action
        utilities = self.u_matrix[client_action, :]  # 8 values
        sorted_utilities = np.sort(utilities)

        # Bond determines percentile
        # bond=0 → expect worst outcome (index 0)
        # bond=1 → expect best outcome (index 7)
        position = self.bond * 7

        # Interpolate between adjacent indices
        lower_idx = int(position)  # Floor
        upper_idx = min(lower_idx + 1, 7)
        interpolation_weight = position - lower_idx

        expected_payoffs[client_action] = (
            (1 - interpolation_weight) * sorted_utilities[lower_idx] +
            interpolation_weight * sorted_utilities[upper_idx]
        )

    return expected_payoffs
```

**Example:**

```
Client action 2 (W = Warm) utility row:
[+10, +50, +60, +50, +10, -50, -40, -50]

Sorted:
[-50, -50, -40, +10, +10, +50, +50, +60]
  ↑0   ↑1   ↑2   ↑3   ↑4   ↑5   ↑6   ↑7

If bond = 0.25:
  position = 0.25 * 7 = 1.75
  lower_idx = 1, upper_idx = 2
  interpolation_weight = 0.75
  expected = 0.25 * (-50) + 0.75 * (-40) = -42.5
  → Low bond = expect poor outcomes

If bond = 0.85:
  position = 0.85 * 7 = 5.95
  lower_idx = 5, upper_idx = 6
  interpolation_weight = 0.95
  expected = 0.05 * (+50) + 0.95 * (+50) = +50
  → High bond = expect good outcomes
```

### Type 2: Frequency-Amplifier Client (`frequency_amplifier_client.py`)

**Philosophy:** History amplifies expectations. Frequently-seen therapist actions become more salient.

**The calculation:**

```python
def _calculate_expected_payoffs(self) -> NDArray[np.float64]:
    # Step 1: Calculate therapist behavior frequencies
    therapist_frequencies = self._calculate_marginal_frequencies()
    # Returns: [P(therapist does action 0), ..., P(therapist does action 7)]

    expected_payoffs = np.zeros(8)

    for client_action in range(8):
        # Step 2: Get raw utilities
        raw_utilities = self.u_matrix[client_action, :]

        # Step 3: Amplify based on frequency
        adjusted_utilities = raw_utilities + (
            raw_utilities * therapist_frequencies * self.history_weight
        )

        # Step 4: Sort and apply bond percentile
        sorted_adjusted = np.sort(adjusted_utilities)

        position = self.bond * 7
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, 7)
        interpolation_weight = position - lower_idx

        expected_payoffs[client_action] = (
            (1 - interpolation_weight) * sorted_adjusted[lower_idx] +
            interpolation_weight * sorted_adjusted[upper_idx]
        )

    return expected_payoffs
```

**Example:**

```
Client action 2 (W), raw utilities:
[+10, +50, +60, +50, +10, -50, -40, -50]

Therapist frequencies (from memory):
[0.05, 0.10, 0.40, 0.30, 0.05, 0.02, 0.03, 0.05]
  ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
  D     WD    W     WS    S     CS    C     CD

Therapist has been mostly Warm (40%) and Warm-Submissive (30%)

Amplification (history_weight = 1.0):
adjusted[2] = +60 + (+60 * 0.40 * 1.0) = +60 + 24 = +84
adjusted[3] = +50 + (+50 * 0.30 * 1.0) = +50 + 15 = +65
adjusted[0] = +10 + (+10 * 0.05 * 1.0) = +10 + 0.5 = +10.5

Frequently-seen positive outcomes get BOOSTED!

If therapist had been cold (high frequency for actions 5,6,7):
adjusted[5] = -50 + (-50 * 0.40 * 1.0) = -50 - 20 = -70

Frequently-seen negative outcomes get WORSE!
```

**Key insight:** This mechanism makes the client's expectations align with recent history. If therapy has been going well (warm therapist responses), the client expects even better. If it's been going poorly, they expect even worse.

---

## How These Pieces Work Together: A Session Walkthrough

Let's trace a complete therapy session with a **Perceptual Frequency-Amplifier Client**:

### Session N: Before the Interaction

```
Client state:
- memory: 50 past (client_action, therapist_action) pairs
- relationship_satisfaction: 35.2 (has been calculated from memory)
- bond: 0.67 (calculated from RS)
- session_count: 23
```

### Step 1: Therapist Acts

```
Therapist selects action: 2 (W = Warm)
```

### Step 2: Client Selects Action

```python
# Client doesn't see therapist action yet!
# Must choose based on expectations

# 2a: Calculate expected payoffs for each possible client action
payoffs = client._calculate_expected_payoffs()
# For FrequencyAmplifier, this:
#   - Calculates therapist frequencies from memory
#   - Amplifies utilities based on those frequencies
#   - Applies bond-based percentile selection
# Result: [12.3, 45.6, 52.1, 38.9, 15.2, -22.4, -18.7, 8.3]

# 2b: Convert to probabilities
probabilities = client._softmax(payoffs)
# Result: [0.02, 0.15, 0.58, 0.11, 0.03, 0.001, 0.002, 0.09]

# 2c: Sample action
client_action = client.rng.choice(8, p=probabilities)
# Most likely: action 2 (58% chance)
# Result: client_action = 2
```

### Step 3: Client Perceives Therapist Action

```python
# Actual therapist action: 2 (W)
perceived_action, record = client._perceive_therapist_action(actual_action=2)

# Stage 1: History-based perception
recent_memory = last_15_interactions_from_memory()
frequencies = [0.05, 0.08, 0.50, 0.25, 0.03, 0.02, 0.03, 0.04]
#                                ↑ Therapist has been Warm 50% of time

# Roll dice for baseline path
if random() < 0.2:  # 20% chance
    stage1_result = 2  # Perceive correctly
else:  # 80% chance, frequency-based
    accuracy = frequencies[2] = 0.50  # 50% chance of correct perception
    if random() < 0.50:
        stage1_result = 2  # Correct!
    else:
        stage1_result = sample_from([0.05, 0.08, 0.50, 0.25, ...])
        # Might get 3 (WS) since that's also frequent

# Let's say: stage1_result = 2 (perceived correctly)

# Stage 2: Adjacency noise
if random() < 0.1:  # 10% chance
    stage1_result += choice([-1, +1])  # Might become 1 or 3

# Let's say: perceived_action = 2 (no shift)
```

### Step 4: Update Memory

```python
client.update_memory(client_action=2, therapist_action=2)
# Actually calls the perceptual version, which stores:
# (client_action=2, perceived_action=2)

# Memory now:
# - Oldest interaction drops off
# - New (2, 2) added
# - session_count increments to 24
```

### Step 5: Recalculate Internal State

```python
# Happens automatically in update_memory()

# 5a: Recalculate RS
utilities = [u_matrix[c, t] for c, t in memory]  # 50 values
weights = get_memory_weights(50)  # Recency-weighted
new_rs = weighted_average(utilities, weights)
# Let's say: new_rs = 36.8 (improved!)

# 5b: Recalculate bond
rs_normalized = (36.8 - rs_min) / (rs_max - rs_min)  # → 0.72
rs_shifted = 2 * (0.72 - 0.8)  # → -0.16
new_bond = 1 / (1 + exp(-5 * -0.16))  # → 0.55
# Bond changed from 0.67 to 0.55 (slight decrease)
```

### Step 6: Check Dropout (if session 10)

```python
if client.session_count == 10 and not client.dropout_checked:
    client.dropout_checked = True
    if client.relationship_satisfaction < client.initial_rs:
        return DROPOUT_SIGNAL
```

### Session N+1: The Cycle Repeats

Now with updated memory, RS, and bond, the client will have different expectations next session!

---

## Key Concepts for Understanding the Architecture

### 1. **Inheritance and Method Overriding**

```python
# BaseClientAgent defines the structure
class BaseClientAgent:
    def _calculate_expected_payoffs(self):
        raise NotImplementedError  # MUST be implemented by subclass

    def select_action(self):
        payoffs = self._calculate_expected_payoffs()  # Calls subclass version!
        # ... rest of logic

# FrequencyAmplifierClient provides the specific implementation
class FrequencyAmplifierClient(BaseClientAgent):
    def _calculate_expected_payoffs(self):
        # Frequency-specific logic here
        return payoffs
```

**Why this matters:** `BaseClientAgent` can use `_calculate_expected_payoffs()` even though it doesn't know *how* it's implemented. Each subclass provides its own version.

### 2. **The Mixin Pattern**

```python
# Multiple inheritance: left-to-right priority
class PerceptualFrequencyClient(PerceptualClientMixin, FrequencyAmplifierClient):
    pass

# Method Resolution Order (MRO):
# 1. PerceptualFrequencyClient (empty, just combines)
# 2. PerceptualClientMixin (overrides update_memory)
# 3. FrequencyAmplifierClient (overrides _calculate_expected_payoffs)
# 4. BaseClientAgent (provides base functionality)

# When you call client.update_memory():
# - Finds it in PerceptualClientMixin (uses that version)
# - That version calls super().update_memory()
# - Which finds BaseClientAgent.update_memory()
```

### 3. **Static Methods vs Instance Methods**

```python
# Instance method: works with specific client's data
def _calculate_relationship_satisfaction(self):
    utilities = [self.u_matrix[c, t] for c, t in self.memory]
    #           ↑ Uses THIS client's matrix and memory

# Static method: doesn't need client instance
@staticmethod
def generate_problematic_memory(pattern_type):
    # Creates memory without needing an actual client
    memory = [(6, 2)] * 50  # All cold-warm interactions
    return memory

# Usage:
memory = BaseClientAgent.generate_problematic_memory("cold_stuck")
# No client instance needed!
```

### 4. **Configuration-Driven Design**

Many values come from `config.py`:

```python
# In base_client.py
from src.config import (
    U_MATRIX,           # Default utility matrix
    MEMORY_SIZE,        # Always 50
    get_memory_weights, # Function for recency weighting
    rs_to_bond,         # Bond calculation function
)

# This means you can change behavior globally by editing config.py
```

---

## Summary: What Calculations Happen?

### Every Session:

1. **Expectation Calculation** (`_calculate_expected_payoffs`)
   - Input: Current memory, bond, u_matrix
   - Process: Mechanism-specific (bond-only, frequency-amplifier, etc.)
   - Output: 8 expected payoffs (one per possible client action)

2. **Action Selection** (`select_action`)
   - Input: Expected payoffs, entropy
   - Process: Softmax → probabilities → sample
   - Output: Chosen client action (0-7)

3. **Perception** (if perceptual client) (`_perceive_therapist_action`)
   - Input: Actual therapist action, recent memory
   - Process: Stage 1 (history-based) → Stage 2 (adjacency noise)
   - Output: Perceived therapist action

4. **Memory Update** (`update_memory`)
   - Input: Client action, (perceived) therapist action
   - Process: Add to deque, recalculate RS and bond
   - Output: Updated internal state

5. **RS Calculation** (`_calculate_relationship_satisfaction`)
   - Input: 50 memory pairs, u_matrix
   - Process: Lookup utilities, apply recency weights, average
   - Output: Relationship satisfaction (float)

6. **Bond Calculation** (`_calculate_bond`)
   - Input: RS, rs_min, rs_max
   - Process: Normalize → shift → sigmoid
   - Output: Bond (0 to 1)

### Every 10th Session:

7. **Dropout Check** (`check_dropout`)
   - Input: Current RS, initial RS
   - Process: Compare (only once, at session 10)
   - Output: Boolean (drop out or continue)

---

## Where to Look for Specific Information

- **Memory management**: `src/agents/client_agents/base_client.py:218-237`
- **RS calculation**: `src/agents/client_agents/base_client.py:100-121`
- **Bond calculation**: `src/agents/client_agents/base_client.py:123-131`, `src/config.py:166-212`
- **Action selection**: `src/agents/client_agents/base_client.py:172-193`
- **Perception logic**: `src/agents/client_agents/perceptual_distortion.py:112-183`
- **Frequency amplification**: `src/agents/client_agents/frequency_amplifier_client.py:87-125`
- **Utility matrix bounds**: `src/config.py:47-100`

---

## The Five Client Types Summary

| Client Type | File | Key Difference | How Expectations Form |
|-------------|------|----------------|----------------------|
| **BondOnlyClient** | `bond_only_client.py` | No history tracking | Bond alone determines percentile of raw utilities |
| **FrequencyAmplifierClient** | `frequency_amplifier_client.py` | Frequency amplifies utilities | Frequent outcomes become more salient (positive → more positive, negative → more negative) |
| **ConditionalAmplifierClient** | `conditional_amplifier_client.py` | Uses conditional probabilities | P(therapist|client_action) - tracks specific response patterns |
| **BondWeightedFrequencyClient** | `bond_weighted_frequency_amplifier_client.py` | History weight varies by bond | High bond → more history influence, Low bond → less history influence |
| **BondWeightedConditionalClient** | `bond_weighted_conditional_amplifier_client.py` | Conditional + bond-weighted | Combines conditional probabilities with bond-modulated history weight |

All five can be wrapped with `with_perception()` to add perceptual distortion!
