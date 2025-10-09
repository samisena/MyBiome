"""
Analyze and compare interventions extracted by each batch size.
"""
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path(__file__).parent / "data" / "results"

# Load all experiments
experiments = {}
for exp_id in ['EXP-001', 'EXP-002', 'EXP-003', 'EXP-004']:
    result_file = results_dir / f'{exp_id}_results.json'
    if result_file.exists():
        with open(result_file, 'r') as f:
            experiments[exp_id] = json.load(f)

if not experiments:
    print("No experiment results found yet. Experiments may still be running.")
    exit(0)

print("\n" + "="*80)
print("DETAILED INTERVENTION COMPARISON BY BATCH SIZE")
print("="*80)

# Analyze each experiment
for exp_id in ['EXP-001', 'EXP-002', 'EXP-003', 'EXP-004']:
    if exp_id not in experiments:
        continue

    data = experiments[exp_id]
    batch_size = data['batch_size']
    interventions = data['phase2_results'].get('interventions', [])

    print(f"\n{exp_id} - Batch Size {batch_size}")
    print(f"Total Interventions: {len(interventions)}")
    print("-"*80)

    if not interventions:
        print("  No intervention details available")
        continue

    # Group by intervention name
    by_name = defaultdict(list)
    for interv in interventions:
        name = interv.get('intervention_name', 'Unknown')
        by_name[name].append(interv)

    # Print each unique intervention
    for i, (name, instances) in enumerate(sorted(by_name.items()), 1):
        print(f"\n{i}. {name}")
        for inst in instances:
            condition = inst.get('health_condition', 'Unknown')
            mechanism = inst.get('mechanism', 'Not specified')
            correlation = inst.get('correlation_type', 'Unknown')
            pmid = inst.get('paper_id', 'Unknown')

            print(f"   Condition: {condition}")
            print(f"   Mechanism: {mechanism}")
            print(f"   Correlation: {correlation}")
            print(f"   PMID: {pmid}")

# Cross-experiment comparison
print("\n\n" + "="*80)
print("CROSS-EXPERIMENT COMPARISON")
print("="*80)

# Collect all unique intervention names from each experiment
intervention_sets = {}
for exp_id, data in experiments.items():
    batch_size = data['batch_size']
    interventions = data['phase2_results'].get('interventions', [])
    names = set(i.get('intervention_name', 'Unknown') for i in interventions)
    intervention_sets[batch_size] = names

# Find interventions unique to each batch size
print("\nUnique Interventions by Batch Size:")
print("-"*80)

for batch_size in sorted(intervention_sets.keys()):
    other_batches = [b for b in intervention_sets.keys() if b != batch_size]
    unique = intervention_sets[batch_size]
    for other in other_batches:
        unique = unique - intervention_sets[other]

    if unique:
        print(f"\nBatch={batch_size} ONLY:")
        for name in sorted(unique):
            print(f"  - {name}")
    else:
        print(f"\nBatch={batch_size}: No unique interventions")

# Find common interventions across all batch sizes
print("\n" + "="*80)
print("INTERVENTIONS EXTRACTED BY ALL BATCH SIZES:")
print("-"*80)

if intervention_sets:
    common = set.intersection(*intervention_sets.values())
    if common:
        for name in sorted(common):
            print(f"  - {name}")
    else:
        print("  No interventions were extracted by all batch sizes")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\n{'Batch Size':<12} {'Total':<8} {'Unique Names':<15} {'Papers/Hour':<15}")
print("-"*80)

for exp_id in ['EXP-001', 'EXP-002', 'EXP-003', 'EXP-004']:
    if exp_id not in experiments:
        continue

    data = experiments[exp_id]
    batch_size = data['batch_size']
    interventions = data['phase2_results'].get('interventions', [])
    unique_names = len(set(i.get('intervention_name') for i in interventions))
    papers_hour = data.get('papers_per_hour', 0)

    print(f"{batch_size:<12} {len(interventions):<8} {unique_names:<15} {papers_hour:<15.1f}")

print("\n" + "="*80)
