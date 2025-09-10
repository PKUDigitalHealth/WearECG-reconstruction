# Calculate macro-average AUROC from the CSV file data

import pandas as pd

# Read the CSV file
df = pd.read_csv('classification/res618/real2.csv')

# Filter out the macro_auc row and get only the ECG conditions
ecg_conditions = df[df['Field_ID'] != 'macro_auc']

# Get the AUROC values
auroc_values = ecg_conditions['AUROC'].tolist()

# Calculate macro-average AUROC
macro_auc = sum(auroc_values) / len(auroc_values)

print(f"Number of ECG conditions: {len(auroc_values)}")
print(f"Macro-average AUROC: {macro_auc:.4f}")
print(f"Macro-average AUROC (rounded to 4 decimal places): {macro_auc:.4f}")

# Verify with the value in the CSV
csv_macro_auc = df[df['Field_ID'] == 'macro_auc']['AUROC'].iloc[0]
print(f"\nMacro-average AUROC from CSV: {csv_macro_auc:.4f}")

# Show the difference
print(f"Difference: {abs(macro_auc - csv_macro_auc):.6f}")

# Print all AUROC values for verification
print(f"\nAll AUROC values:")
for i, (condition, auroc) in enumerate(zip(ecg_conditions['Field_ID'], auroc_values)):
    print(f"{i+1:2d}. {condition}: {auroc:.4f}")

print(f"\nSum of all AUROC values: {sum(auroc_values):.4f}")
print(f"Average: {sum(auroc_values)/len(auroc_values):.4f}") 
