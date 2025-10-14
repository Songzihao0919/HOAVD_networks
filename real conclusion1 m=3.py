import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from itertools import combinations
import math
import csv
from tqdm import tqdm
import time


class HigherOrderNetworkAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.nodes = set()
        self.time_windows = None
        self.hyper_edges = defaultdict(list)  # Store hyperedges for each time window
        self.a_i = {}  # Node activity for m=3
        self.b_i = {}  # Node vulnerability for m=3
        self.edge_ratios = []  # Store ratios for each hyperedge
        self.edge_ratios_squared = []  # Store squared ratios for each hyperedge
        self.original_steps = None  # Store original number of time steps

    def load_and_prepare_data(self):
        """Load dataset and prepare analysis environment"""
        # Load CSV file
        self.df = pd.read_csv(self.file_path)
        print(f"Dataset contains {len(self.df)} rows")

        # Extract all unique nodes
        self.nodes = set(self.df['node1']).union(set(self.df['node2']))
        print(f"Dataset contains {len(self.nodes)} unique nodes")

        # Create continuous time windows (1626 to 7341)
        min_win = self.df['time_window'].min()
        max_win = self.df['time_window'].max()
        self.time_windows = list(range(min_win, max_win + 1))
        self.original_steps = len(self.time_windows)
        print(f"Total time windows: {self.original_steps} (from {min_win} to {max_win})")

        # Identify 4-node interactions (m=3 hyperedges)
        self.identify_4node_interactions()

    def identify_4node_interactions(self):
        """Identify 4-node interactions (m=3 hyperedges) in each time window"""
        print("Identifying 4-node interactions...")

        # Group by time window
        grouped = self.df.groupby('time_window')

        # For each time window, identify 4-node interactions
        for time_win, group in tqdm(grouped, total=len(grouped)):
            # Create graph for this time window
            G = nx.Graph()
            for _, row in group.iterrows():
                G.add_edge(row['node1'], row['node2'])

            # Find all cliques of size 4
            cliques = list(nx.find_cliques(G))
            four_cliques = [tuple(sorted(clique)) for clique in cliques if len(clique) == 4]

            # Store unique 4-node interactions
            self.hyper_edges[time_win] = list(set(four_cliques))

        # Add empty time windows
        for time_win in self.time_windows:
            if time_win not in self.hyper_edges:
                self.hyper_edges[time_win] = []

        total_hyperedges = sum(len(v) for v in self.hyper_edges.values())
        print(f"Identified {total_hyperedges} 4-node interactions")

    def calculate_activity(self):
        """Calculate node activity for m=3 (a_i)"""
        # Track which time windows each node participates in a 4-node interaction
        node_active_windows = defaultdict(set)

        for time_win, hyperedges in self.hyper_edges.items():
            for hyperedge in hyperedges:
                for node in hyperedge:
                    node_active_windows[node].add(time_win)

        # Calculate activity = active windows / total windows
        total_windows = len(self.time_windows)
        for node in self.nodes:
            active_count = len(node_active_windows.get(node, set()))
            self.a_i[node] = active_count / total_windows

        print("Node activity (m=3) calculation completed")

    def calculate_vulnerability(self, inactive_threshold=1000):
        """Calculate node vulnerability for m=3 (b_i)"""
        # Track interruption count for each node
        interruption_count = defaultdict(int)

        # Find all time windows where each node is active in a 4-node interaction
        node_active_windows = defaultdict(list)
        for time_win, hyperedges in self.hyper_edges.items():
            for hyperedge in hyperedges:
                for node in hyperedge:
                    node_active_windows[node].append(time_win)

        # For each node, check for interruptions
        for node in tqdm(self.nodes, desc="Calculating vulnerability"):
            active_windows = sorted(node_active_windows[node])
            for i in range(len(active_windows)):
                current_window = active_windows[i]

                # Only consider events that have at least 1000 future windows
                if current_window <= max(self.time_windows) - inactive_threshold:
                    # Check if there's another interaction within the next 1000 windows
                    found_next = False
                    for j in range(i + 1, len(active_windows)):
                        if active_windows[j] <= current_window + inactive_threshold:
                            found_next = True
                            break

                    # If no interaction found within next 1000 windows, count as interruption
                    if not found_next:
                        interruption_count[node] += 1

        # Calculate vulnerability
        valid_windows = len(self.time_windows) - inactive_threshold
        for node in self.nodes:
            if valid_windows > 0:
                self.b_i[node] = interruption_count[node] / valid_windows
            else:
                self.b_i[node] = 0

        print("Node vulnerability (m=3) calculation completed")

    def calculate_theoretical_percolation(self):
        """Calculate theoretical percolation threshold based on the formula for m=3"""
        # For each hyperedge in the dataset, calculate the ratio
        for time_win, hyperedges in self.hyper_edges.items():
            for hyperedge in hyperedges:
                # Calculate numerator: sum of a_i for nodes in hyperedge
                numerator = sum(self.a_i[node] for node in hyperedge)

                # Calculate denominator: sum of (a_i + b_i) for nodes in hyperedge
                denominator = sum(self.a_i[node] + self.b_i[node] for node in hyperedge)

                # Avoid division by zero
                if denominator > 0:
                    ratio = numerator / denominator
                    self.edge_ratios.append(ratio)
                    self.edge_ratios_squared.append(ratio ** 2)

        # Calculate expectations
        if len(self.edge_ratios) > 0:
            E_ratio = np.mean(self.edge_ratios)
            E_ratio_squared = np.mean(self.edge_ratios_squared)

            # Calculate left side of the equation
            left_side = E_ratio_squared / E_ratio

            # Calculate right side of the equation (for m=3)
            n = len(self.nodes)
            right_side = math.factorial(2) / (n ** 3)  # (m-1)! / n^m, where m=3

            return left_side, right_side, E_ratio, E_ratio_squared
        else:
            return 0, 0, 0, 0

    def simulate_network(self, T_max, model_type='HOAVD'):
        """Simulate network evolution without extending time dimension"""
        active_hyperedges = set()

        # Simulate T_max time steps
        for t in tqdm(range(T_max), desc=f"Simulating {model_type}"):
            # Calculate current time window in original data
            current_time_win = self.time_windows[t % self.original_steps]
            current_hyperedges = self.hyper_edges[current_time_win]

            # HOAVD model: Remove hyperedges based on vulnerability
            if model_type == 'HOAVD':
                for hyperedge in list(active_hyperedges):
                    # Calculate average vulnerability for this hyperedge
                    avg_vulnerability = np.mean([self.b_i.get(node, 0) for node in hyperedge])
                    if np.random.rand() < avg_vulnerability:
                        active_hyperedges.remove(hyperedge)

            # Add new hyperedges based on activity
            for hyperedge in current_hyperedges:
                # Calculate average activity for this hyperedge
                avg_activity = np.mean([self.a_i.get(node, 0) for node in hyperedge])
                if np.random.rand() < avg_activity:
                    active_hyperedges.add(tuple(sorted(hyperedge)))

        return active_hyperedges

    def calculate_network_metrics(self, active_hyperedges):
        """Calculate network metrics for hypergraph"""
        # Project hypergraph to simple graph
        G = nx.Graph()

        # Add all pairwise connections from hyperedges
        for hyperedge in active_hyperedges:
            for pair in combinations(hyperedge, 2):
                G.add_edge(pair[0], pair[1])

        # Calculate average degree
        if G.number_of_nodes() == 0:
            return 0, 0

        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees)

        # Calculate largest connected component size
        if nx.is_empty(G):
            return avg_degree, 0

        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc) / len(self.nodes)

        return avg_degree, largest_cc_size

    def validate_conclusions(self):
        """Validate theoretical conclusions"""
        # Calculate key parameters
        n = len(self.nodes)
        m = 3
        n_m = n ** m
        print(f"n = {n}, m = {m}, n^m = {n_m}")

        # Calculate theoretical percolation threshold
        left_side, right_side, E_ratio, E_ratio_squared = self.calculate_theoretical_percolation()

        print("\n=== Theoretical Percolation Threshold Calculation ===")
        print(f"E[ratio] = {E_ratio:.6f}")
        print(f"E[ratio^2] = {E_ratio_squared:.6f}")
        print(f"Left side (E[ratio^2] / E[ratio]) = {left_side:.6f}")
        print(f"Right side ((m-1)! / n^m) = {right_side:.6f}")

        # Check if theoretical percolation condition is satisfied
        if left_side > right_side:
            print("✅ Theoretical percolation condition satisfied: Left side > Right side")
            theoretical_percolation = True
        else:
            print("❌ Theoretical percolation condition not satisfied: Left side ≤ Right side")
            theoretical_percolation = False

        # Set time thresholds
        T_short = 30000  # T << n^m (405^3 ≈ 66,430,125)
        T_long = 2 * 10 ** 9  # T >> n^m (1,000,000 >> 66 million)

        # Simulate networks for short timescale
        print("\nSimulating networks for short timescale...")
        hyperedges_hoavd_short = self.simulate_network(T_short, 'HOAVD')
        hyperedges_hoad_short = self.simulate_network(T_short, 'HOAD')

        # Calculate metrics for short timescale
        print("\nCalculating metrics for short timescale...")
        avg_deg_hoavd_short, cc_hoavd_short = self.calculate_network_metrics(hyperedges_hoavd_short)
        avg_deg_hoad_short, cc_hoad_short = self.calculate_network_metrics(hyperedges_hoad_short)

        # Output short timescale results immediately
        print("\n=== Short Timescale (T << n^m) Validation Results ===")
        print(f"   HOAVD (a+b): Avg degree = {avg_deg_hoavd_short:.4f}, Largest CC = {cc_hoavd_short:.4f}")
        print(f"   HOAD (a only): Avg degree = {avg_deg_hoad_short:.4f}, Largest CC = {cc_hoad_short:.4f}")

        # Validate conclusion 1 for short timescale
        print("\n=== Conclusion Validation for Short Timescale ===")
        cc_diff_short = abs(cc_hoavd_short - cc_hoad_short)
        if cc_diff_short < 0.05:
            print(
                f"✅ Conclusion 1 verified: At T={T_short}<<n^m={n_m}, HOAVD and HOAD CC size difference ({cc_diff_short:.4f}) < 0.05")
        else:
            print(f"⚠️ Conclusion 1 not verified: Significant difference ({cc_diff_short:.4f}) at short timescale")

        # Ask user if they want to continue to long timescale simulation
        print("\nShort timescale simulation completed. Would you like to continue to long timescale simulation?")
        print("This may take a long time. Enter 'y' to continue or any other key to exit.")

        user_input = input("Continue? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Exiting as per user request.")
            return

        # Simulate networks for long timescale
        print("\nSimulating networks for long timescale...")
        hyperedges_hoavd_long = self.simulate_network(T_long, 'HOAVD')
        hyperedges_hoad_long = self.simulate_network(T_long, 'HOAD')

        # Calculate metrics for long timescale
        print("\nCalculating metrics for long timescale...")
        avg_deg_hoavd_long, cc_hoavd_long = self.calculate_network_metrics(hyperedges_hoavd_long)
        avg_deg_hoad_long, cc_hoad_long = self.calculate_network_metrics(hyperedges_hoad_long)

        # Output results
        print("\n=== Validation Results ===")
        print("1. Short timescale (T << n^m):")
        print(f"   HOAVD (a+b): Avg degree = {avg_deg_hoavd_short:.4f}, Largest CC = {cc_hoavd_short:.4f}")
        print(f"   HOAD (a only): Avg degree = {avg_deg_hoad_short:.4f}, Largest CC = {cc_hoad_short:.4f}")

        print("\n2. Long timescale (T >> n^m):")
        print(f"   HOAVD (a+b): Avg degree = {avg_deg_hoavd_long:.4f}, Largest CC = {cc_hoavd_long:.4f}")
        print(f"   HOAD (a only): Avg degree = {avg_deg_hoad_long:.4f}, Largest CC = {cc_hoad_long:.4f}")

        # Validate conclusions
        print("\n=== Conclusion Validation ===")

        # Conclusion 1: Vulnerability negligible at T << n^m
        cc_diff_short = abs(cc_hoavd_short - cc_hoad_short)
        if cc_diff_short < 0.05:
            print(
                f"✅ Conclusion 1 verified: At T={T_short}<<n^m={n_m}, HOAVD and HOAD CC size difference ({cc_diff_short:.4f}) < 0.05")
        else:
            print(f"⚠️ Conclusion 1 not verified: Significant difference ({cc_diff_short:.4f}) at short timescale")

        # Conclusion 2: Percolation at T >> n^m
        print(f"\nPercolation threshold (Largest CC > 0.05):")
        print(f"HOAVD model: {cc_hoavd_long:.4f} {'✅' if cc_hoavd_long > 0.05 else '❌'}")
        print(f"HOAD model: {cc_hoad_long:.4f} {'✅' if cc_hoad_long > 0.05 else '❌'}")

        # Compare theoretical prediction with simulation results
        print(f"\n=== Theoretical vs. Simulation Comparison ===")
        if theoretical_percolation and cc_hoavd_long > 0.05:
            print("✅ Theory and simulation agree: Percolation predicted and observed")
        elif not theoretical_percolation and cc_hoavd_long <= 0.05:
            print("✅ Theory and simulation agree: No percolation predicted and observed")
        else:
            print("⚠️ Theory and simulation disagree: Check model assumptions")

        # Visualization
        self.visualize_results(
            cc_hoavd_short, cc_hoad_short, cc_hoavd_long, cc_hoad_long,
            avg_deg_hoavd_short, avg_deg_hoad_short, avg_deg_hoavd_long, avg_deg_hoad_long,
            T_short, T_long, n_m, left_side, right_side
        )

    def visualize_results(self, cc_hoavd_short, cc_hoad_short, cc_hoavd_long, cc_hoad_long,
                          avg_deg_hoavd_short, avg_deg_hoad_short, avg_deg_hoavd_long, avg_deg_hoad_long,
                          T_short, T_long, n_m, left_side, right_side):
        """Visualize results"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Short timescale CC comparison
        ax[0, 0].bar(['HOAVD', 'HOAD'], [cc_hoavd_short, cc_hoad_short], color=['blue', 'orange'])
        ax[0, 0].set_title(f'Short Timescale (T={T_short} << n^3={n_m})')
        ax[0, 0].set_ylabel('Largest Connected Component')
        ax[0, 0].set_ylim(0, 1)

        # Short timescale degree comparison
        ax[0, 1].bar(['HOAVD', 'HOAD'], [avg_deg_hoavd_short, avg_deg_hoad_short], color=['blue', 'orange'])
        ax[0, 1].set_title(f'Short Timescale (T={T_short} << n^3={n_m})')
        ax[0, 1].set_ylabel('Average Degree')

        # Long timescale CC comparison
        ax[1, 0].bar(['HOAVD', 'HOAD'], [cc_hoavd_long, cc_hoad_long], color=['blue', 'orange'])
        ax[1, 0].set_title(f'Long Timescale (T={T_long} >> n^3={n_m})')
        ax[1, 0].set_ylabel('Largest Connected Component')
        ax[1, 0].set_ylim(0, 1)

        # Long timescale degree comparison
        ax[1, 1].bar(['HOAVD', 'HOAD'], [avg_deg_hoavd_long, avg_deg_hoad_long], color=['blue', 'orange'])
        ax[1, 1].set_title(f'Long Timescale (T={T_long} >> n^3={n_m})')
        ax[1, 1].set_ylabel('Average Degree')

        plt.tight_layout()
        plt.savefig('higher_order_conclusion_validation_m3.png')

        # Create a separate figure for theoretical results
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.bar(['Left Side', 'Right Side'], [left_side, right_side], color=['green', 'red'])
        ax2.set_title('Theoretical Percolation Threshold (m=3)')
        ax2.set_ylabel('Value')
        ax2.text(0, left_side, f'{left_side:.6f}', ha='center', va='bottom')
        ax2.text(1, right_side, f'{right_side:.6f}', ha='center', va='bottom')
        plt.savefig('theoretical_percolation_m3.png')

        print(
            "\nVisualizations saved to 'higher_order_conclusion_validation_m3.png' and 'theoretical_percolation_m3.png'")


# Main program
if __name__ == "__main__":
    # Set file path
    file_path = "tij_SFHH.dat_.csv"

    print("=== Higher-Order Network Analyzer (m=3) ===")
    print(f"Loading dataset: {file_path}")

    # Create analyzer instance
    analyzer = HigherOrderNetworkAnalyzer(file_path)

    # Step 1: Load and prepare data
    analyzer.load_and_prepare_data()

    # Step 2: Calculate node properties
    analyzer.calculate_activity()
    analyzer.calculate_vulnerability()

    # Step 3: Validate conclusions
    analyzer.validate_conclusions()

    print("\n=== Analysis Complete ===")