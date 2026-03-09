import pandas as pd
import numpy as np

df_raw = pd.DataFrame({
    "target": ["Yes", "No", "False", "True", np.nan]
})

df_clean = df_raw.copy(deep=True)
target_column = "target"

# Step 1
for col in df_clean.select_dtypes(include=['object', 'string']).columns:
    df_clean[col] = df_clean[col].astype(str).str.strip().replace("", np.nan)
    df_clean[col] = df_clean[col].replace("nan", np.nan)

print("Unique values:", set(df_clean[target_column].dropna().unique()))
print("Subset check:", set(df_clean[target_column].dropna().unique()).issubset({"Yes", "No", "True", "False"}))

is_text_type = pd.api.types.is_object_dtype(df_clean[target_column]) or pd.api.types.is_string_dtype(df_clean[target_column])
if is_text_type:
    if set(df_clean[target_column].dropna().unique()).issubset({"Yes", "No", "True", "False"}):
        mapping = {"No": 0, "Yes": 1, "False": 0, "True": 1}
        df_clean[target_column] = df_clean[target_column].map(mapping)
        df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors="coerce")

print("Dtype:", df_clean[target_column].dtype)
print("Series:", df_clean[target_column])
