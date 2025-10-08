import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

print("=" * 60)
print("FAILED INTERVENTIONS (intervention_category IS NULL)")
print("=" * 60)

cursor.execute('''
    SELECT id, intervention_name, health_condition, paper_id
    FROM interventions
    WHERE intervention_category IS NULL
    ORDER BY id
''')

failed_interventions = cursor.fetchall()
print(f"\nTotal: {len(failed_interventions)} interventions\n")

for row in failed_interventions:
    print(f"ID: {row[0]}")
    print(f"  Intervention: {row[1]}")
    print(f"  Condition: {row[2]}")
    print(f"  Paper ID: {row[3]}")
    print()

print("=" * 60)
print("FAILED CONDITIONS (condition_category IS NULL)")
print("=" * 60)

cursor.execute('''
    SELECT DISTINCT health_condition
    FROM interventions
    WHERE condition_category IS NULL
    ORDER BY health_condition
''')

failed_conditions = cursor.fetchall()
print(f"\nTotal: {len(failed_conditions)} unique conditions\n")

for row in failed_conditions:
    print(f"  - {row[0]}")

print("\n" + "=" * 60)
print("CHECKING FOR PATTERNS")
print("=" * 60)

# Check if failed interventions have anything in common
cursor.execute('''
    SELECT intervention_name, COUNT(*) as cnt
    FROM interventions
    WHERE intervention_category IS NULL
    GROUP BY intervention_name
    HAVING cnt > 1
''')

print("\nRepeated failed intervention names:")
repeats = cursor.fetchall()
if repeats:
    for row in repeats:
        print(f"  - {row[0]} ({row[1]} times)")
else:
    print("  None (all failures are unique)")

conn.close()
