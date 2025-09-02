import numpy as np
import json
import os
import sys

def calculate_precision_at_1(predictions, ground_truth):
    """
    Calculate precision@1 for the given predictions and ground truth.
    
    Args:
        predictions (np.array): Array of predicted gallery indices for each query
        ground_truth (np.array): Array of correct gallery indices for each query
    
    Returns:
        float: Precision@1 score (percentage of correct predictions)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(f"Predictions length ({len(predictions)}) doesn't match ground truth length ({len(ground_truth)})")
    
    # Count correct predictions
    correct_predictions = np.sum(predictions == ground_truth)
    total_predictions = len(predictions)
    
    # Calculate precision@1 as percentage
    precision_at_1 = (correct_predictions / total_predictions)
    
    return precision_at_1

def load_submission_file(filepath):
    """
    Load submission file and handle potential errors.
    
    Args:
        filepath (str): Path to the submission file
    
    Returns:
        np.array or None: Loaded array or None if file doesn't exist or is invalid
    """
    try:
        if not os.path.exists(filepath):
            print(f"Warning: Submission file {filepath} not found")
            return None
        
        submission = np.load(filepath)
        print(f"Loaded {filepath}: shape {submission.shape}, dtype {submission.dtype}")
        return submission
    
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def load_ground_truth_file(filepath):
    """
    Load ground truth file.
    
    Args:
        filepath (str): Path to the ground truth file
    
    Returns:
        np.array: Loaded ground truth array
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ground truth file {filepath} not found")
        
        ground_truth = np.load(filepath)
        print(f"Loaded ground truth {filepath}: shape {ground_truth.shape}, dtype {ground_truth.dtype}")
        return ground_truth
    
    except Exception as e:
        print(f"Error loading ground truth {filepath}: {str(e)}")
        raise

def evaluate_test_set(submission_file, ground_truth_file, test_name):
    """
    Evaluate a single test set.
    
    Args:
        submission_file (str): Path to submission file
        ground_truth_file (str): Path to ground truth file
        test_name (str): Name of the test set for logging
    
    Returns:
        float or None: Precision@1 score or None if evaluation failed
    """
    print(f"\n=== Evaluating {test_name} ===")
    
    # Load ground truth
    try:
        ground_truth = load_ground_truth_file(ground_truth_file)
    except Exception as e:
        print(f"Failed to load ground truth for {test_name}: {str(e)}")
        return None
    
    # Load submission
    submission = load_submission_file(submission_file)
    if submission is None:
        print(f"Failed to load submission for {test_name}")
        return None
    
    # Validate submission format
    if submission.shape != ground_truth.shape:
        print(f"Shape mismatch for {test_name}: submission {submission.shape} vs ground truth {ground_truth.shape}")
        return None
    
    # Calculate precision@1
    try:
        score = calculate_precision_at_1(submission, ground_truth)
        print(f"{test_name} - Precision@1: {score:.2f}")
        
        # Log some statistics
        correct_count = np.sum(submission == ground_truth)
        total_count = len(submission)
        print(f"{test_name} - Correct predictions: {correct_count}/{total_count}")
        
        return score
    
    except Exception as e:
        print(f"Error calculating precision@1 for {test_name}: {str(e)}")
        return None

def main():
    """
    Main evaluation function.
    """
    print("Starting evaluation...")
    if os.environ.get('METRIC_PATH'):
        METRIC_PATH = os.environ.get("METRIC_PATH") + "/" 
    else:
        METRIC_PATH = ""  # Fallback for local testing
    # File paths
    submission_a_file = "submission_a.npy"
    submission_b_file = "submission_b.npy"
    ground_truth_a_file = METRIC_PATH + "answer_a.npy"
    ground_truth_b_file = METRIC_PATH + "answer_b.npy"
    output_file = "score.json"
    
    # Evaluate test set A
    score_a = evaluate_test_set(submission_a_file, ground_truth_a_file, "Test Set A")
    

    # Evaluate test set B
    score_b = evaluate_test_set(submission_b_file, ground_truth_b_file, "Test Set B")
    

    # Determine overall status
    status = True
    msg = "Success!"
    
    # Handle missing or failed evaluations
    if score_a is None:
        score_a = 0.0
        status = False
        msg = "Failed to evaluate Test Set A"
    
    if score_b is None:
        score_b = 0.0
        if status:  # Only update if not already failed
            status = False
            msg = "Failed to evaluate Test Set B"
        else:
            msg = "Failed to evaluate both test sets"
    if score_a > 1:
        score_a = 0.0
    if score_b > 1:
        score_b = 0.0
    def sanitize_score(value):
        """处理单个分数值，将NaN和inf替换为0"""
        if not np.isfinite(value):
            return 0.0
        return value
    # Create result dictionary
    result = {
        "status": status,
        "score": {
            "public_a": sanitize_score(score_a),
            "private_b": sanitize_score(score_b),
        },
        "msg": msg,
    }
    
    # Save results to JSON
    try:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {str(e)}")
        sys.exit(1)
    
    # Print final summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Status: {status}")
    print(f"Test Set A (public) Score: {score_a:.2f}")
    print(f"Test Set B (private) Score: {score_b:.2f}")
    print(f"Message: {msg}")
    
    return result

if __name__ == "__main__":
    main()
