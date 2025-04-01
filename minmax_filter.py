import pandas as pd
import sys
import os

def normalize_and_split(input_csv):
    # Derive output filenames from input prefix
    prefix = os.path.splitext(os.path.basename(input_csv))[0]
    output_boundary_csv = f"{prefix}_train.csv"
    output_other_csv = f"{prefix}_test.csv"

    # Load CSV
    df = pd.read_csv(input_csv)

    # Identify class column (case insensitive)
    class_col = next((col for col in df.columns if col.lower() == 'class'), None)
    if class_col is None:
        raise ValueError("No column named 'class' found (case insensitive).")

    # Separate features and class
    features = [col for col in df.columns if col != class_col]

    # Min-max normalization of features
    df_normalized = df.copy()
    df_normalized[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())

    # Collect boundary indices
    boundary_indices = set()
    for cls, group in df_normalized.groupby(class_col):
        for feature in features:
            min_val = group[feature].min()
            max_val = group[feature].max()
            min_indices = group[group[feature] == min_val].index
            max_indices = group[group[feature] == max_val].index
            boundary_indices.update(min_indices)
            boundary_indices.update(max_indices)

    # Split data
    boundary_df = df.loc[sorted(boundary_indices)]
    other_df = df.drop(index=boundary_indices)

    # Save output
    boundary_df.to_csv(output_boundary_csv, index=False)
    other_df.to_csv(output_other_csv, index=False)

    print(f"Boundary cases written to: {output_boundary_csv}")
    print(f"Other cases written to: {output_other_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    normalize_and_split(input_csv)
