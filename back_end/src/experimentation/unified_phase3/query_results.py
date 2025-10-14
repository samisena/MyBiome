import sqlite3

conn = sqlite3.connect('experiment_results.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT e.experiment_id, e.experiment_name, er.entity_type,
           er.num_clusters, er.num_singleton_clusters,
           er.silhouette_score, er.davies_bouldin_score
    FROM experiments e
    JOIN experiment_results er ON e.experiment_id = er.experiment_id
    ORDER BY e.experiment_id, er.entity_type
''')

rows = cursor.fetchall()

print("=" * 120)
print(f"{'ID':<4} | {'Experiment Name':<35} | {'Entity Type':<12} | {'Clusters':<8} | {'Singletons':<10} | {'Silhouette':<10} | {'DB Score':<8}")
print("=" * 120)

for row in rows:
    exp_id, exp_name, entity_type, num_clusters, num_singletons, silhouette, davies_bouldin = row
    silhouette_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
    davies_bouldin_str = f"{davies_bouldin:.3f}" if davies_bouldin is not None else "N/A"
    print(f"{exp_id:<4} | {exp_name:<35} | {entity_type:<12} | {num_clusters:<8} | {num_singletons:<10} | {silhouette_str:<10} | {davies_bouldin_str:<8}")

conn.close()
