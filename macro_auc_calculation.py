# Calculate macro-average AUROC from the provided data

# AUROC values from the table (first column)
auroc_values = [
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

# Calculate macro-average AUROC
macro_auc = sum(auroc_values) / len(auroc_values)

print(f"Number of classes: {len(auroc_values)}")
print(f"Macro-average AUROC: {macro_auc:.4f}")
print(f"Macro-average AUROC (rounded to 4 decimal places): {macro_auc:.4f}")

# Also calculate from the CSV file for verification
import pandas as pd

# Read the CSV file
df = pd.read_csv('classification/res618/real2.csv')

# Calculate macro-average AUROC from CSV
csv_macro_auc = df['AUROC'].mean()
print(f"\nMacro-average AUROC from CSV: {csv_macro_auc:.4f}")

# Show the difference
print(f"Difference: {abs(macro_auc - csv_macro_auc):.6f}") 