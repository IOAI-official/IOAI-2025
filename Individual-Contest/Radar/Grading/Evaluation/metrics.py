import json
import pandas as pd
import numpy as np
import os

def load_ground_truth_from_csv(csv_path):
    """Load ground truth labels from CSV file"""
    df = pd.read_csv(csv_path)
    
    ground_truth = {}
    for _, row in df.iterrows():
        filename = row['filename']
        
        # Extract ground truth labels (all columns except filename)
        gt_cols = [col for col in df.columns if col.startswith('pixel_')]
        gt_labels = row[gt_cols].values.astype(int)
        
        ground_truth[filename] = gt_labels
    
    print(f"Loaded {len(ground_truth)} samples from CSV")
    return ground_truth

def calculate_score_from_csv(csv_path, ground_truth, bonus=50):
    df = pd.read_csv(csv_path)

    total_score = 0
    total_theo = 0

    for _, row in df.iterrows():
        filename = row['filename']
        if filename not in ground_truth:
            raise ValueError(f"Missing ground truth for file: {filename}")
        
        pred_cols = [col for col in df.columns if col.startswith('pixel_')]
        if len(pred_cols) == 0:
            raise ValueError("No prediction columns found (e.g., pixel_0, pixel_1, ...)")

        predictions = row[pred_cols].values.astype(int)
        gt_labels = ground_truth[filename]

        if len(predictions) != len(gt_labels):
            raise ValueError(f"Length mismatch for {filename}: prediction({len(predictions)}), ground truth({len(gt_labels)})")

        equal_mask = predictions == gt_labels
        neg_one_mask = gt_labels == -1

        score_neg_one = np.sum(equal_mask & neg_one_mask) * 1
        score_other = np.sum(equal_mask & ~neg_one_mask) * bonus
        score_theo = np.sum(neg_one_mask) * 1 + np.sum(~neg_one_mask) * bonus

        total_score += score_neg_one + score_other
        total_theo += score_theo

    if total_theo == 0:
        print("Warning: Theoretical total score is zero. Returning score 0.0")
        return 0.0

    score = total_score / total_theo

    if not np.isfinite(score) or score < 0 or score > 1:
        print(f"Warning: Score {score} is invalid. Returning 0.0")
        return 0.0

    return score

def main():
    """Main function to calculate scores for both validation and testing sets"""
    bonus = 50  # Score weights for non-background categories
    if os.environ.get('METRIC_PATH'):
        METRIC_PATH = os.environ.get("METRIC_PATH")+"/" 
    else:
        METRIC_PATH =""
    # Paths for predictions
    val_csv_path = 'submission_val.csv'
    test_csv_path = 'submission_test.csv'
    
    # Paths for ground truth CSV files
    val_gt_csv_path = METRIC_PATH + 'ground_truth_val.csv'
    test_gt_csv_path = METRIC_PATH + 'ground_truth_test.csv'
    
    # Load ground truth for validation set from CSV
    val_ground_truth = load_ground_truth_from_csv(val_gt_csv_path)
    
    # Load ground truth for testing set from CSV
    test_ground_truth = load_ground_truth_from_csv(test_gt_csv_path)
    
    # Calculate scores
    print("\nCalculating scores...")
    
    # Validation set score
    val_score = calculate_score_from_csv(val_csv_path, val_ground_truth, bonus)
    print(f"Validation set score: {val_score:.4f}")
    if val_score > 1:
        val_score = 0

    # Testing set score
    test_score = calculate_score_from_csv(test_csv_path, test_ground_truth, bonus)
    print(f"Testing set score: {test_score:.4f}")
    if test_score > 1:
        test_score = 0
        
    # Save results
    score = {
        "public_a": val_score,
        "private_b": test_score,
    }
    ret_json = {
        "status": True,
        "score": score,
        "msg": "Success!",
    }
    
    with open('score.json', 'w') as f:
        json.dump(ret_json, f, indent=2)
    
if __name__ == '__main__':
    main()