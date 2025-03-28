# pip install pandas numpy scikit-learn tqdm
import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys


def separation_distance(candidate, others, dim, axis_value):
    # Calculate absolute differences between candidate and others along dimension dim
    diffs = np.abs(others[dim] - candidate[dim])
    
    # Check if there are any points that cross the candidate along axis_value
    # (x_i - x_c)(x_c - x_j) <= 0 means x_c is between x_i and x_j
    crossings = (others[axis_value] - candidate[axis_value]) * (candidate[axis_value] - others[axis_value]) <= 0
    
    # If there are crossings, return the minimum distance among crossing points
    if crossings.any():
        return diffs[crossings].min()
    else:
        # No crossings found, return 0
        return 0


def process_permutation(args):
    pi, df, class_col, feature_columns, classes = args
    d = len(feature_columns)  # Number of dimensions/features
    local_indices = {c: set() for c in classes}  # Initialize empty sets for each class

    for c in classes:
        # Split data into current class and other classes
        class_group = df[df[class_col] == c]
        other_group = df[df[class_col] != c]

        for j in range(d):
            # For each dimension j in the permutation:
            fj = pi[j]      # Current feature
            fL = pi[(j - 1) % d]        # Left neighboring feature
            fR = pi[(j + 1) % d]        # Right neighboring feature

            # Consider both minimum and maximum points along feature fj
            for t in ['min', 'max']:
                # Get the extreme value (min or max) for the current feature
                val = class_group[fj].min() if t == 'min' else class_group[fj].max()
                
                # Find all candidates that have this extreme value
                candidates = class_group[class_group[fj] == val]

                # Initialize variables to track best candidates
                best_L_score = -1  # Best separation score for left neighbor
                best_R_score = -1  # Best separation score for right neighbor
                best_L_idx = None  # Index of best candidate for left neighbor
                best_R_idx = None  # Index of best candidate for right neighbor

                # Evaluate each candidate point
                for idx, candidate in candidates.iterrows():
                    # Calculate separation distances along left and right neighbor dimensions
                    # deltaL = distance to nearest point of other class along dimension fL
                    deltaL = separation_distance(candidate, other_group, fL, fj)
                    # deltaR = distance to nearest point of other class along dimension fR
                    deltaR = separation_distance(candidate, other_group, fR, fj)

                    # Update best candidates if better separation is found
                    if deltaL > best_L_score:
                        best_L_score = deltaL
                        best_L_idx = idx
                    if deltaR > best_R_score:
                        best_R_score = deltaR
                        best_R_idx = idx

                # Add the best candidates to the local indices set
                if best_L_idx is not None:
                    local_indices[c].add(best_L_idx)
                if best_R_idx is not None:
                    local_indices[c].add(best_R_idx)

    # Return the set of indices for each class that form the envelope
    return local_indices


def extract_envelope_cases(input_csv, output_csv):
    # Load the dataset
    df = pd.read_csv(input_csv)

    # Find the class column (case-insensitive)
    class_col = next((col for col in df.columns if col.lower() == 'class'), None)
    if class_col is None:
        raise ValueError("No column named 'class' found (case-insensitive).")

    # Identify feature columns (all columns except class)
    feature_columns = [col for col in df.columns if col != class_col]
    # Get unique class values
    classes = df[class_col].unique()

    # Normalize all features to [0,1] range
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # Generate all possible permutations of feature columns
    # Each permutation represents a different ordering of dimensions
    dimension_permutations = list(permutations(feature_columns))
    args_list = [(pi, df, class_col, feature_columns, classes) for pi in dimension_permutations]

    # Process all permutations in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_permutation, args_list), total=len(args_list), desc="Processing permutations"))

    # Combine results from all permutations
    envelope_indices = {c: set() for c in classes}
    for result in results:
        for c in result:
            envelope_indices[c].update(result[c])

    # Collect all indices from all classes
    selected_indices = set()
    for idx_set in envelope_indices.values():
        selected_indices.update(idx_set)

    # Create the filtered dataframe with only envelope (coreset) points
    coreset_df = df.loc[sorted(selected_indices)]

    # Compute the evaluation set by dropping coreset indices
    eval_df = df.drop(index=sorted(selected_indices))

    # Save both sets
    coreset_df.to_csv("output_coreset.csv", index=False)
    eval_df.to_csv("output_eval.csv", index=False)
    print(f"Saved {len(coreset_df)} coreset cases to output_coreset.csv")
    print(f"Saved {len(eval_df)} evaluation cases to output_eval.csv")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter.py input.csv output.csv")
        sys.exit(1)
    extract_envelope_cases(sys.argv[1], sys.argv[2])
