"""
Test if "always complementary" strategy is too effective.

If success rate is >80%, the model is broken - simple complementarity 
shouldn't be enough to model the Three-Step process.
"""

import numpy as np
from src.agents.client_agent import create_client
from src.config import SUCCESS_THRESHOLD, MAX_SESSIONS

def always_complement(client_action: int) -> int:
    """
    Simple complementary strategy:
    - Dominant ↔ Submissive (0↔4, 1↔3, 7↔5)
    - Warm ↔ Warm (2↔2)
    - Cold ↔ Cold (6↔6)
    """
    complementary_map = {
        0: 4,  # D → S
        1: 3,  # WD → WS
        2: 2,  # W → W
        3: 1,  # WS → WD
        4: 0,  # S → D
        5: 7,  # CS → CD
        6: 6,  # C → C
        7: 5,  # CD → CS
    }
    return complementary_map[client_action]


def simulate_therapy_episode(client, strategy_fn, max_sessions=MAX_SESSIONS):
    """
    Simulate one complete therapy episode.
    
    Returns
    -------
    dict with:
        - outcome: 'success', 'dropout', or 'max_length'
        - num_sessions: number of sessions completed
        - final_rs: final relationship satisfaction
        - final_bond: final bond level
        - rs_trajectory: list of RS values over time
    """
    rs_trajectory = [client.relationship_satisfaction]
    
    for session in range(max_sessions):
        # Check dropout first (happens at session 10)
        if client.check_dropout():
            return {
                'outcome': 'dropout',
                'num_sessions': session,
                'final_rs': client.relationship_satisfaction,
                'final_bond': client.bond,
                'rs_trajectory': rs_trajectory,
            }
        
        # Client acts
        client_action = client.select_action()
        
        # Therapist responds complementarily
        therapist_action = strategy_fn(client_action)
        
        # Update client's memory and state
        client.update_memory(client_action, therapist_action)
        rs_trajectory.append(client.relationship_satisfaction)
        
        # Check success
        if client.relationship_satisfaction >= SUCCESS_THRESHOLD:
            return {
                'outcome': 'success',
                'num_sessions': session + 1,
                'final_rs': client.relationship_satisfaction,
                'final_bond': client.bond,
                'rs_trajectory': rs_trajectory,
            }
    
    # Reached max sessions without success or dropout
    return {
        'outcome': 'max_length',
        'num_sessions': max_sessions,
        'final_rs': client.relationship_satisfaction,
        'final_bond': client.bond,
        'rs_trajectory': rs_trajectory,
    }

def simulate_therapy_episode_detailed(client, strategy_fn, max_sessions=MAX_SESSIONS):
    """
    Simulate one complete therapy episode WITH detailed interaction tracking.
    
    Returns dict with outcome info PLUS:
        - interaction_history: list of (client_action, therapist_action) tuples
        - bond_trajectory: list of bond values over time
    """
    rs_trajectory = [client.relationship_satisfaction]
    bond_trajectory = [client.bond]
    interaction_history = []
    
    for session in range(max_sessions):
        # Check dropout first
        if client.check_dropout():
            return {
                'outcome': 'dropout',
                'num_sessions': session,
                'initial_rs': rs_trajectory[0],
                'final_rs': client.relationship_satisfaction,
                'final_bond': client.bond,
                'rs_trajectory': rs_trajectory,
                'bond_trajectory': bond_trajectory,
                'interaction_history': interaction_history,
            }
        
        # Client acts
        client_action = client.select_action()
        
        # Therapist responds
        therapist_action = strategy_fn(client_action)
        
        # Record interaction
        interaction_history.append((client_action, therapist_action))
        
        # Update client
        client.update_memory(client_action, therapist_action)
        rs_trajectory.append(client.relationship_satisfaction)
        bond_trajectory.append(client.bond)
        
        # Check success
        if client.relationship_satisfaction >= SUCCESS_THRESHOLD:
            return {
                'outcome': 'success',
                'num_sessions': session + 1,
                'initial_rs': rs_trajectory[0],
                'final_rs': client.relationship_satisfaction,
                'final_bond': client.bond,
                'rs_trajectory': rs_trajectory,
                'bond_trajectory': bond_trajectory,
                'interaction_history': interaction_history,
            }
    
    # Max length reached
    return {
        'outcome': 'max_length',
        'num_sessions': max_sessions,
        'initial_rs': rs_trajectory[0],
        'final_rs': client.relationship_satisfaction,
        'final_bond': client.bond,
        'rs_trajectory': rs_trajectory,
        'bond_trajectory': bond_trajectory,
        'interaction_history': interaction_history,
    }


def run_experiment(n_clients=100, pattern_type="cold_stuck"):
    """
    Run therapy simulation on multiple clients.
    """
    print(f"Testing 'Always Complementary' Strategy")
    print("=" * 70)
    print(f"Number of clients: {n_clients}")
    print(f"Pattern type: {pattern_type}")
    print(f"Success threshold: {SUCCESS_THRESHOLD:.2f}")
    print(f"Max sessions: {MAX_SESSIONS}")
    print("=" * 70)
    
    results = []
    
    for i in range(n_clients):
        if (i + 1) % 20 == 0:
            print(f"  Simulating client {i + 1}/{n_clients}...")
        
        # Create client
        client = create_client(pattern_type=pattern_type, random_state=i)
        
        # Simulate therapy
        result = simulate_therapy_episode_detailed(client, always_complement)
        results.append(result)
    
    # Analyze results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    outcomes = [r['outcome'] for r in results]
    num_success = outcomes.count('success')
    num_dropout = outcomes.count('dropout')
    num_max_length = outcomes.count('max_length')
    
    success_rate = num_success / n_clients * 100
    dropout_rate = num_dropout / n_clients * 100
    max_length_rate = num_max_length / n_clients * 100
    
    print(f"\nOUTCOMES:")
    print(f"  Success:     {num_success:3d} / {n_clients} ({success_rate:5.1f}%)")
    print(f"  Dropout:     {num_dropout:3d} / {n_clients} ({dropout_rate:5.1f}%)")
    print(f"  Max length:  {num_max_length:3d} / {n_clients} ({max_length_rate:5.1f}%)")
    
    # Success statistics
    successes = [r for r in results if r['outcome'] == 'success']
    if successes:
        sessions_to_success = [r['num_sessions'] for r in successes]
        print(f"\nSUCCESS STATISTICS:")
        print(f"  Mean sessions: {np.mean(sessions_to_success):.1f}")
        print(f"  Median sessions: {np.median(sessions_to_success):.1f}")
        print(f"  Min sessions: {np.min(sessions_to_success)}")
        print(f"  Max sessions: {np.max(sessions_to_success)}")
    
    # Final RS distribution
    final_rs_values = [r['final_rs'] for r in results]
    print(f"\nFINAL RS DISTRIBUTION:")
    print(f"  Mean: {np.mean(final_rs_values):.2f}")
    print(f"  Median: {np.median(final_rs_values):.2f}")
    print(f"  Min: {np.min(final_rs_values):.2f}")
    print(f"  Max: {np.max(final_rs_values):.2f}")
    
    # RS improvement
    initial_rs_values = [r['rs_trajectory'][0] for r in results]
    rs_improvements = [final - initial for initial, final in zip(initial_rs_values, final_rs_values)]
    print(f"\nRS IMPROVEMENT:")
    print(f"  Mean: {np.mean(rs_improvements):.2f}")
    print(f"  Median: {np.median(rs_improvements):.2f}")
        # Initial RS statistics
    initial_rs_values = [r['initial_rs'] for r in results]
    print(f"\nINITIAL RS DISTRIBUTION:")
    print(f"  Mean: {np.mean(initial_rs_values):.2f}")
    print(f"  Median: {np.median(initial_rs_values):.2f}")
    print(f"  Std Dev: {np.std(initial_rs_values):.2f}")
    print(f"  Min: {np.min(initial_rs_values):.2f}")
    print(f"  Max: {np.max(initial_rs_values):.2f}")
    
    # Analyze interaction patterns
    from src.config import OCTANTS
    
    print(f"\nINTERACTION PATTERN ANALYSIS:")
    
    # Count octant usage across all successful episodes
    all_client_actions = []
    all_therapist_actions = []
    for r in results:
        for client_act, therapist_act in r['interaction_history']:
            all_client_actions.append(client_act)
            all_therapist_actions.append(therapist_act)
    
    print(f"\n  Client Octant Distribution:")
    for octant_idx in range(8):
        count = all_client_actions.count(octant_idx)
        pct = count / len(all_client_actions) * 100 if all_client_actions else 0
        print(f"    {OCTANTS[octant_idx]:3s} ({octant_idx}): {count:4d} ({pct:5.1f}%)")
    
    print(f"\n  Therapist Octant Distribution:")
    for octant_idx in range(8):
        count = all_therapist_actions.count(octant_idx)
        pct = count / len(all_therapist_actions) * 100 if all_therapist_actions else 0
        print(f"    {OCTANTS[octant_idx]:3s} ({octant_idx}): {count:4d} ({pct:5.1f}%)")
    
    # Show detailed trajectory for first 3 successful clients
    successes = [r for r in results if r['outcome'] == 'success'][:3]
    
    if successes:
        print(f"\nDETAILED TRAJECTORIES (First 3 Successful Cases):")
        print("=" * 70)
        
        for idx, result in enumerate(successes, 1):
            print(f"\nClient {idx}:")
            print(f"  Initial RS: {result['initial_rs']:.2f}, Final RS: {result['final_rs']:.2f}")
            print(f"  Sessions to success: {result['num_sessions']}")
            print(f"\n  Session-by-session interactions (first 10 sessions):")
            print(f"  {'Sess':>4s} | {'Client':>6s} | {'Therapist':>9s} | {'RS':>6s} | {'Bond':>5s}")
            print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*5}")
            
            # Show first 10 sessions
            for sess_num in range(min(10, len(result['interaction_history']))):
                client_oct, therapist_oct = result['interaction_history'][sess_num]
                rs = result['rs_trajectory'][sess_num + 1]  # +1 because trajectory includes initial
                bond = result['bond_trajectory'][sess_num + 1]
                
                print(f"  {sess_num+1:4d} | {OCTANTS[client_oct]:>6s} | {OCTANTS[therapist_oct]:>9s} | {rs:6.2f} | {bond:5.3f}")
            
            if len(result['interaction_history']) > 10:
                print(f"  ... ({len(result['interaction_history']) - 10} more sessions)")


    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if success_rate > 80:
        print("🔴 MODEL IS BROKEN!")
        print("   Simple complementarity is TOO effective.")
        print("   The Three-Step Model is unnecessary in current design.")
        print("   → NEEDS FUNDAMENTAL REDESIGN")
    elif success_rate > 60:
        print("🟡 MODEL IS TOO EASY")
        print("   Simple complementarity works well.")
        print("   Strategic timing may not be necessary.")
        print("   → Consider adding constraints or costs")
    elif success_rate > 40:
        print("🟢 MODEL DIFFICULTY SEEMS APPROPRIATE")
        print("   Simple complementarity helps but isn't sufficient.")
        print("   Strategic approaches may be needed.")
        print("   → Proceed with RL training")
    else:
        print("🔵 MODEL IS CHALLENGING")
        print("   Even complementarity struggles.")
        print("   Complex strategies will be needed.")
        print("   → May need to verify model isn't too hard")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Run the experiment
    results = run_experiment(n_clients=100, pattern_type="cold_stuck")
    
    # Optional: Save detailed results
    import json
    
    # Convert numpy types to Python types for JSON serialization
    results_serializable = []
    for r in results:
        r_copy = r.copy()
        r_copy['rs_trajectory'] = [float(x) for x in r['rs_trajectory']]
        r_copy['final_rs'] = float(r['final_rs'])
        r_copy['final_bond'] = float(r['final_bond'])
        results_serializable.append(r_copy)
    
    with open('always_complement_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print("\n✓ Detailed results saved to 'always_complement_results.json'")
