# Compare macro-average AUROC from table data vs CSV file

# AUROC values from the user's table (first column)
table_auroc_values = [
    0.7958,  # Normal ECG
    0.9247,  # Sinus Rhythm
    0.9882,  # Sinus Bradycardia
    0.9751,  # Atrial Fibrillation
    0.9911,  # Sinus Tachycardia
    0.8609,  # Left Axis Deviation
    0.9551,  # PVC
    0.9411,  # RBBB
    0.7290,  # LAE
    0.9722,  # PAC
    0.8785,  # PSVC
    0.7519,  # LBBB
    0.6881,  # LVH
    0.7406,  # Short QT
    0.8025,  # Long QT
    0.9088,  # Atrial Flutter
    0.9418,  # With Sinus Arrhythmia
    0.9065,  # LAFB
    0.8588,  # RAD
    0.7621,  # Ectopic Atrial Rhythm
    0.7600,  # Short PR
    0.6019,  # Repolarization Abn.
    0.6219,  # RAE
    0.6310,  # Voltage Criteria LVH
    0.7822,  # LPFB
    0.8752,  # 1st Degree AV Block
    0.8047,  # RVH
    0.8427,  # STEMI
    0.8374,  # SVT
    0.8815,  # VT
    0.7927,  # Early Repolarization
    0.8831,  # WPW
    0.8469,  # Acute
    0.8437,  # Acute MI
    0.8785,  # SVC
    0.7428   # 2:1 AV conduction
]

# Calculate macro-average AUROC from table
table_macro_auc = sum(table_auroc_values) / len(table_auroc_values)

print("=== COMPARISON: Table Data vs CSV File ===")
print(f"Table data - Number of classes: {len(table_auroc_values)}")
print(f"Table data - Macro-average AUROC: {table_macro_auc:.4f}")

# Get CSV data
import pandas as pd
df = pd.read_csv('classification/res618/real2.csv')
ecg_conditions = df[df['Field_ID'] != 'macro_auc']
csv_auroc_values = ecg_conditions['AUROC'].tolist()
csv_macro_auc = sum(csv_auroc_values) / len(csv_auroc_values)

print(f"\nCSV file - Number of classes: {len(csv_auroc_values)}")
print(f"CSV file - Macro-average AUROC: {csv_macro_auc:.4f}")

print(f"\nDifference: {abs(table_macro_auc - csv_macro_auc):.6f}")

# Check which conditions are missing or different
table_conditions = [
    "Normal ECG", "Sinus Rhythm", "Sinus Bradycardia", "Atrial Fibrillation", 
    "Sinus Tachycardia", "Left Axis Deviation", "PVC", "RBBB", "LAE", "PAC", 
    "PSVC", "LBBB", "LVH", "Short QT", "Long QT", "Atrial Flutter", 
    "With Sinus Arrhythmia", "LAFB", "RAD", "Ectopic Atrial Rhythm", 
    "Short PR", "Repolarization Abn.", "RAE", "Voltage Criteria LVH", 
    "LPFB", "1st Degree AV Block", "RVH", "STEMI", "SVT", "VT", 
    "Early Repolarization", "WPW", "Acute", "Acute MI", "SVC", "2:1 AV conduction"
]

csv_conditions = ecg_conditions['Field_ID'].tolist()

print(f"\n=== MISSING CONDITIONS ===")
print("Conditions in table but not in CSV:")
for cond in table_conditions:
    if cond not in csv_conditions:
        print(f"  - {cond}")

print(f"\nConditions in CSV but not in table:")
for cond in csv_conditions:
    if cond not in table_conditions:
        print(f"  - {cond}")

print(f"\n=== SUMMARY ===")
print(f"Macro-average AUROC from your table data: {table_macro_auc:.4f}")
print(f"Macro-average AUROC from CSV file: {csv_macro_auc:.4f}")
print(f"Official macro-average AUROC (from CSV): 0.8005") 