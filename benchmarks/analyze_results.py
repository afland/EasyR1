#!/usr/bin/env python3
"""
Analyze benchmark results by turn counter.

This script reads benchmark_results.json and computes statistics
for chamfer distances grouped by turn counter.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path


def analyze_benchmark_results():
    """Analyze benchmark results by turn counter."""
    # Load the JSON data
    json_file = Path("benchmark_results.json")
    
    if not json_file.exists():
        print(f"Error: {json_file} not found!")
        return
    
    print("Loading benchmark results...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Group data by turn counter
    turn_data = defaultdict(list)
    
    for entry in data:
        turn = entry['turn']
        chamfer_distance = entry['chamfer_distance']
        
        # Only include valid chamfer distances (not -1.0)
        if chamfer_distance != -1.0:
            turn_data[turn].append(chamfer_distance)
    
    # Sort turn counters for consistent output
    sorted_turns = sorted(turn_data.keys())
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS ANALYSIS BY TURN COUNTER")
    print("="*60)
    
    for turn in sorted_turns:
        distances = turn_data[turn]
        count = len(distances)
        
        if count == 0:
            continue
            
        # Compute statistics
        mean_dist = statistics.mean(distances)
        median_dist = statistics.median(distances)
        min_dist = min(distances)
        max_dist = max(distances)
        
        print(f"\nTurn {turn}:")
        print(f"  Number of rows: {count}")
        print(f"  Mean chamfer distance: {mean_dist:.6f}")
        print(f"  Median chamfer distance: {median_dist:.6f}")
        print(f"  Min chamfer distance: {min_dist:.6f}")
        print(f"  Max chamfer distance: {max_dist:.6f}")
    
    # Also show count of entries with invalid chamfer distances (-1.0)
    invalid_counts = defaultdict(int)
    total_counts = defaultdict(int)
    
    for entry in data:
        turn = entry['turn']
        chamfer_distance = entry['chamfer_distance']
        total_counts[turn] += 1
        if chamfer_distance == -1.0:
            invalid_counts[turn] += 1
    
    print("\n" + "="*60)
    print("SUMMARY OF ALL ENTRIES (including invalid chamfer distances)")
    print("="*60)
    
    for turn in sorted(total_counts.keys()):
        total = total_counts[turn]
        invalid = invalid_counts[turn]
        valid = total - invalid
        
        print(f"\nTurn {turn}:")
        print(f"  Total rows: {total}")
        print(f"  Valid chamfer distances: {valid}")
        print(f"  Invalid chamfer distances (-1.0): {invalid}")


if __name__ == "__main__":
    analyze_benchmark_results() 