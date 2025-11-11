"""
Visualize results from always_complement_results.json

Creates plots showing:
- Success rates and session distributions
- RS and bond trajectories
- Octant usage patterns
- Payoff and probability evolution
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Go up two levels to project root
sys.path.insert(0, str(project_root))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def normalize_trajectories(results):
    """Ensure all trajectory fields are proper lists of floats."""
    for r in results:
        # Normalize rs_trajectory
        if not isinstance(r.get('rs_trajectory'), list):
            r['rs_trajectory'] = [r['rs_trajectory']]
        r['rs_trajectory'] = [float(x) for x in r['rs_trajectory']]
        
        # Normalize bond_trajectory
        if not isinstance(r.get('bond_trajectory'), list):
            r['bond_trajectory'] = [r['bond_trajectory']]
        r['bond_trajectory'] = [float(x) for x in r['bond_trajectory']]
        
        # Ensure interaction_history is a list of tuples
        if not isinstance(r.get('interaction_history'), list):
            r['interaction_history'] = []
        # Convert any non-tuple items to tuples if needed
        r['interaction_history'] = [tuple(x) if isinstance(x, list) else x for x in r['interaction_history']]
        
        # Ensure expected_payoffs_history is a list
        if not isinstance(r.get('expected_payoffs_history'), list):
            r['expected_payoffs_history'] = []
        
        # Ensure softmax_probabilities_history is a list
        if not isinstance(r.get('softmax_probabilities_history'), list):
            r['softmax_probabilities_history'] = []

def load_results(json_path):
    """Load and preprocess results from JSON."""
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Normalize data structures
    normalize_trajectories(results)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Add computed columns
    df['rs_improvement'] = df['final_rs'] - df['initial_rs']
    df['success'] = df['outcome'] == 'success'
    
    return df, results

def plot_outcome_distribution(df):
    """Plot distribution of outcomes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Outcome counts
    outcome_counts = df['outcome'].value_counts()
    outcome_counts.plot(kind='bar', ax=ax1, color=['green', 'red', 'orange'])
    ax1.set_title('Outcome Distribution')
    ax1.set_ylabel('Number of Clients')
    ax1.tick_params(axis='x', rotation=45)
    
    # Success rate
    success_rate = (df['outcome'] == 'success').mean() * 100
    ax2.pie([success_rate, 100-success_rate], 
            labels=[f'Success\n{success_rate:.1f}%', f'Other\n{100-success_rate:.1f}%'],
            colors=['green', 'lightcoral'], autopct='%1.1f%%')
    ax2.set_title('Success Rate')
    
    plt.tight_layout()
    return fig

def plot_session_distributions(df):
    """Plot session distributions for successful cases."""
    success_df = df[df['outcome'] == 'success']
    
    if success_df.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sessions to success
    ax1.hist(success_df['num_sessions'], bins=20, alpha=0.7, edgecolor='black')
    ax1.axvline(success_df['num_sessions'].mean(), color='red', linestyle='--', 
                label=f'Mean: {success_df["num_sessions"].mean():.1f}')
    ax1.axvline(success_df['num_sessions'].median(), color='blue', linestyle='--',
                label=f'Median: {success_df["num_sessions"].median():.1f}')
    ax1.set_title('Sessions to Success Distribution')
    ax1.set_xlabel('Sessions')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Box plot
    success_df.boxplot(column='num_sessions', ax=ax2)
    ax2.set_title('Sessions to Success (Box Plot)')
    ax2.set_ylabel('Sessions')
    
    plt.tight_layout()
    return fig

def plot_rs_distributions(df):
    """Plot RS distributions and improvements."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Initial RS distribution
    ax1.hist(df['initial_rs'], bins=30, alpha=0.7, edgecolor='black')
    ax1.set_title('Initial RS Distribution')
    ax1.set_xlabel('Relationship Satisfaction')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['initial_rs'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["initial_rs"].mean():.2f}')
    ax1.legend()
    
    # Final RS distribution
    ax2.hist(df['final_rs'], bins=30, alpha=0.7, edgecolor='black')
    ax2.set_title('Final RS Distribution')
    ax2.set_xlabel('Relationship Satisfaction')
    ax2.set_ylabel('Frequency')
    ax2.axvline(df['final_rs'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["final_rs"].mean():.2f}')
    ax2.legend()
    
    # RS improvement distribution
    ax3.hist(df['rs_improvement'], bins=30, alpha=0.7, edgecolor='black')
    ax3.set_title('RS Improvement Distribution')
    ax3.set_xlabel('RS Change')
    ax3.set_ylabel('Frequency')
    ax3.axvline(df['rs_improvement'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["rs_improvement"].mean():.2f}')
    ax3.legend()
    
    # RS improvement by outcome
    outcome_improvements = []
    labels = []
    for outcome in ['success', 'dropout', 'max_length']:
        outcome_data = df[df['outcome'] == outcome]['rs_improvement']
        if not outcome_data.empty:
            outcome_improvements.append(outcome_data)
            labels.append(outcome.capitalize())
    
    if outcome_improvements:
        ax4.boxplot(outcome_improvements, tick_labels=labels)
        ax4.set_title('RS Improvement by Outcome')
        ax4.set_ylabel('RS Change')
    
    plt.tight_layout()
    return fig

def plot_trajectories(results, max_clients=10):
    """Plot RS and bond trajectories for successful clients."""
    success_results = [r for r in results if r['outcome'] == 'success' and len(r['rs_trajectory']) > 1][:max_clients]
    
    if not success_results:
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # RS trajectories
    for i, result in enumerate(success_results):
        sessions = range(len(result['rs_trajectory']))
        ax1.plot(sessions, result['rs_trajectory'], 
                label=f'Client {i+1} ({result["num_sessions"]} sess)',
                alpha=0.7)
    
    if success_results:
        ax1.axhline(y=success_results[0]['success_threshold'], color='red', linestyle='--',
                    label='Success Threshold')
    ax1.set_title(f'RS Trajectories (First {len(success_results)} Successful Clients)')
    ax1.set_xlabel('Session')
    ax1.set_ylabel('Relationship Satisfaction')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bond trajectories
    for i, result in enumerate(success_results):
        sessions = range(len(result['bond_trajectory']))
        ax2.plot(sessions, result['bond_trajectory'], 
                label=f'Client {i+1}',
                alpha=0.7)
    
    ax2.set_title(f'Bond Trajectories (First {len(success_results)} Successful Clients)')
    ax2.set_xlabel('Session')
    ax2.set_ylabel('Bond Level')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_octant_usage(results):
    """Plot octant usage distributions."""
    from src.config import OCTANTS
    
    # Collect all interactions
    client_counts = defaultdict(int)
    therapist_counts = defaultdict(int)
    total_interactions = 0
    
    for result in results:
        for client_act, therapist_act in result['interaction_history']:
            client_counts[client_act] += 1
            therapist_counts[therapist_act] += 1
            total_interactions += 1
    
    if total_interactions == 0:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Client octant usage
    client_labels = [f'{OCTANTS[i]} ({i})' for i in range(8)]
    client_values = [client_counts[i] / total_interactions * 100 for i in range(8)]
    bars1 = ax1.bar(range(8), client_values, alpha=0.7, edgecolor='black')
    ax1.set_title('Client Octant Usage Distribution')
    ax1.set_xlabel('Octant')
    ax1.set_ylabel('Percentage of Interactions')
    ax1.set_xticks(range(8))
    ax1.set_xticklabels(client_labels, rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars1, client_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Therapist octant usage
    therapist_values = [therapist_counts[i] / total_interactions * 100 for i in range(8)]
    bars2 = ax2.bar(range(8), therapist_values, alpha=0.7, edgecolor='black', color='orange')
    ax2.set_title('Therapist Octant Usage Distribution')
    ax2.set_xlabel('Octant')
    ax2.set_ylabel('Percentage of Interactions')
    ax2.set_xticks(range(8))
    ax2.set_xticklabels(client_labels, rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars2, therapist_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def plot_payoff_evolution(results, max_clients=5):
    """Plot payoff evolution for first few clients."""
    from src.config import OCTANTS
    
    clients_with_payoffs = [r for r in results if r['expected_payoffs_history'] and len(r['expected_payoffs_history']) > 0]
    clients_to_plot = clients_with_payoffs[:max_clients]
    
    if not clients_to_plot:
        return None
    
    fig, axes = plt.subplots(len(clients_to_plot), 1, figsize=(12, 4*len(clients_to_plot)))
    if len(clients_to_plot) == 1:
        axes = [axes]
    
    for idx, result in enumerate(clients_to_plot):
        ax = axes[idx]
        
        # Plot payoff evolution for each octant
        for octant_idx in range(8):
            payoffs_over_time = []
            for session_data in result['expected_payoffs_history']:
                payoff = next((item['payoff'] for item in session_data if item['octant'] == OCTANTS[octant_idx]), 0)
                payoffs_over_time.append(payoff)
            
            if payoffs_over_time:
                ax.plot(range(1, len(payoffs_over_time)+1), payoffs_over_time, 
                       label=f'{OCTANTS[octant_idx]}', marker='o', markersize=3)
        
        ax.set_title(f'Client {idx+1} - Expected Payoffs Evolution (First {len(result["expected_payoffs_history"])} Sessions)')
        ax.set_xlabel('Session')
        ax.set_ylabel('Expected Payoff')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_long_session_clients(results, output_dir):
    """Detailed analysis of clients who took >55 sessions to succeed."""
    from src.config import OCTANTS
    
    # Debug: Check session distribution
    success_sessions = [r['num_sessions'] for r in results if r['outcome'] == 'success']
    if success_sessions:
        print(f"Session count statistics:")
        print(f"  Min sessions to success: {min(success_sessions)}")
        print(f"  Max sessions to success: {max(success_sessions)}")
        print(f"  Mean sessions to success: {np.mean(success_sessions):.1f}")
        print(f"  Median sessions to success: {np.median(success_sessions):.1f}")
        print(f"  Clients with >55 sessions: {sum(1 for s in success_sessions if s > 55)}")
    
    long_session_clients = [r for r in results if r['outcome'] == 'success' and r['num_sessions'] > 55]
    
    if not long_session_clients:
        print("No clients took more than 55 sessions to succeed.")
        print("Try lowering the threshold (e.g., >30) to analyze slower-progressing clients.")
        return
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS OF {len(long_session_clients)} LONG-SESSION CLIENTS (>55 sessions)")
    print(f"{'='*80}")
    
    # Extract and plot RS trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, client in enumerate(long_session_clients):
        ax.plot(client['rs_trajectory'], label=f'Client {i+1} ({client["num_sessions"]} sess)', alpha=0.7)
    
    ax.set_title('RS Trajectories of Clients with >55 Sessions to Success')
    ax.set_xlabel('Session')
    ax.set_ylabel('Relationship Satisfaction')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'long_session_rs_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Analyze and plot bond trajectory patterns
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, client in enumerate(long_session_clients):
        ax.plot(client['bond_trajectory'], label=f'Client {i+1}', alpha=0.7)
    
    ax.set_title('Bond Trajectories of Clients with >55 Sessions to Success')
    ax.set_xlabel('Session')
    ax.set_ylabel('Bond Level')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'long_session_bond_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Detailed session-by-session analysis for the first client as an example
    example_client = long_session_clients[0]
    example_sessions = range(len(example_client['rs_trajectory']))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(example_sessions, example_client['rs_trajectory'], label='RS', marker='o')
    ax.plot(example_sessions, example_client['bond_trajectory'], label='Bond', marker='s')
    
    ax.set_title('Session-by-Session RS and Bond Trajectory (Client 1)')
    ax.set_xlabel('Session')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'example_client_session_analysis.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("Analysis plots saved:")
    print(f"- RS trajectories of long-session clients: {output_dir}/long_session_rs_trajectories.png")
    print(f"- Bond trajectories of long-session clients: {output_dir}/long_session_bond_trajectories.png")
    print(f"- Example client session analysis: {output_dir}/example_client_session_analysis.png")

def main():
    """Main visualization function."""
    # Find JSON file
    script_dir = Path(__file__).parent
    json_path = script_dir / 'always_complement_results.json'
    
    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        return
    
    print(f"Loading results from {json_path}...")
    df, results = load_results(json_path)
    
    print(f"Loaded {len(results)} client results")
    print(f"Success rate: {(df['outcome'] == 'success').mean() * 100:.1f}%")
    
    # Create output directory
    output_dir = script_dir / 'visualization_output'
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    plots = [
        ('outcome_distribution', plot_outcome_distribution(df)),
        ('session_distributions', plot_session_distributions(df)),
        ('rs_distributions', plot_rs_distributions(df)),
        ('trajectories', plot_trajectories(results)),
        ('octant_usage', plot_octant_usage(results)),
        ('payoff_evolution', plot_payoff_evolution(results)),
    ]
    
    # Save plots
    for plot_name, fig in plots:
        if fig is not None:
            output_path = output_dir / f'{plot_name}.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved {plot_name}.png")
            plt.close(fig)
        else:
            print(f"Skipped {plot_name} (no data)")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    success_df = df[df['outcome'] == 'success']
    print(f"Total clients: {len(df)}")
    print(f"Success rate: {(df['outcome'] == 'success').mean() * 100:.1f}%")
    print(f"Dropout rate: {(df['outcome'] == 'dropout').mean() * 100:.1f}%")
    print(f"Max length rate: {(df['outcome'] == 'max_length').mean() * 100:.1f}%")
    
    if not success_df.empty:
        print(f"\nSuccessful clients:")
        print(f"  Mean sessions to success: {success_df['num_sessions'].mean():.1f}")
        print(f"  Median sessions to success: {success_df['num_sessions'].median():.1f}")
        print(f"  Mean RS improvement: {success_df['rs_improvement'].mean():.2f}")
    
    print(f"\nOverall RS change: {df['rs_improvement'].mean():.2f} ± {df['rs_improvement'].std():.2f}")
    
    print(f"\nPlots saved to: {output_dir}/")
    print("Open the PNG files to view visualizations.")
    
    # Analyze long session clients
    analyze_long_session_clients(results, output_dir)

if __name__ == "__main__":
    main()