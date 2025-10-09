"""Show all interventions extracted by each batch size."""
import json
from pathlib import Path

results_dir = Path(__file__).parent / "data" / "results"

for exp_id in ['EXP-001', 'EXP-002', 'EXP-003', 'EXP-004']:
    result_file = results_dir / f'{exp_id}_results.json'

    with open(result_file, 'r') as f:
        data = json.load(f)

    batch_size = data['batch_size']
    total_interventions = data['phase2_results']['total_interventions']

    print('\n' + '='*70)
    print(f'{exp_id} - Batch Size {batch_size}')
    print(f'Total Interventions: {total_interventions}')
    print('='*70 + '\n')

    # Extract all intervention names from all papers
    all_interventions = []
    paper_results = data['phase2_results'].get('paper_results', [])

    for paper_result in paper_results:
        for intervention in paper_result.get('interventions', []):
            name = intervention.get('intervention_name', 'Unknown')
            condition = intervention.get('health_condition', 'Unknown')
            pmid = paper_result.get('pmid', 'Unknown')
            all_interventions.append({
                'name': name,
                'condition': condition,
                'pmid': pmid
            })

    # Print interventions
    for i, interv in enumerate(all_interventions, 1):
        print(f'{i:2}. {interv["name"]:<45} -> {interv["condition"]}')
        print(f'    (PMID: {interv["pmid"]})')

print('\n' + '='*70)
print('COMPARISON SUMMARY')
print('='*70)
