"""
Fix the failed categorizations by updating the database with correct categories.
"""
import sqlite3

conn = sqlite3.connect('back_end/data/processed/intervention_research.db')
cursor = conn.cursor()

print("=" * 60)
print("FIXING FAILED CATEGORIZATIONS")
print("=" * 60)

# Fix interventions
intervention_fixes = {
    1483: "lifestyle",  # self-care education based on conceptual mapping
    1789: "exercise",   # remote rehabilitation training
    1792: "exercise",   # postoperative home-based pulmonary rehabilitation
    1872: "medication", # Sodium-glucose cotransporter 2 inhibitors
    1879: "medication", # Sodium-Glucose Cotransporter-2 inhibitors
    1881: "emerging",   # BI 1595043
}

print("\nFixing interventions...")
for intervention_id, category in intervention_fixes.items():
    cursor.execute(
        "UPDATE interventions SET intervention_category = ? WHERE id = ?",
        (category, intervention_id)
    )
    print(f"  ID {intervention_id} -> {category}")

# Fix conditions
condition_fixes = {
    "female infertility": "endocrine",
    "metabolic adverse effects": "endocrine",
    "metabolic conditions": "endocrine",
    "metabolic pathways in adipose tissue": "endocrine",
    "moderate or severe hypercholesterolemia": "cardiac",
    "moderate or severe hypercholesterolemia, amyloidosis": "cardiac",
}

print("\nFixing conditions...")
for condition_name, category in condition_fixes.items():
    cursor.execute(
        "UPDATE interventions SET condition_category = ? WHERE health_condition = ?",
        (category, condition_name)
    )
    affected = cursor.rowcount
    print(f"  '{condition_name}' -> {category} ({affected} rows)")

conn.commit()

# Verify
cursor.execute('SELECT COUNT(*) FROM interventions WHERE intervention_category IS NULL')
remaining_uncategorized = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM interventions WHERE condition_category IS NULL')
remaining_condition_null = cursor.fetchone()[0]

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"Remaining uncategorized interventions: {remaining_uncategorized}")
print(f"Remaining null condition categories: {remaining_condition_null}")

if remaining_uncategorized == 0:
    print("\n✓ All interventions successfully categorized!")
else:
    print(f"\n⚠ {remaining_uncategorized} interventions still uncategorized")

conn.close()
