import numpy as np
import logging
import math
import json
from datasets import load_from_disk
import os

# Function to configure logging levels
def logging_level(level='info'):
    str_format = '%(asctime)s - %(levelname)s: %(message)s'
    if level == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=str_format, datefmt='%Y-%m-%d %H:%M:%S')
    elif level == 'info':
        logging.basicConfig(level=logging.INFO, format=str_format, datefmt='%Y-%m-%d %H:%M:%S')
    return logging


def validate_predictions_shape(predictions_np, expected_shape_suffix):
    """
    Validate that predictions have the correct shape.
    Accepts either (n, 1, 180, 320) or (n, 180, 320) for this competition.
    Number of samples (n) must be exactly 100.
    """
    if not isinstance(predictions_np, np.ndarray):
        return False, "Prediction data must be a numpy array"
    
    # Accept (n, 1, 180, 320) or (n, 180, 320)
    if len(predictions_np.shape) == 4:
        # (n, 1, 180, 320)
        if predictions_np.shape[1:] != expected_shape_suffix:
            return False, "Prediction data has incorrect dimensions"
        if predictions_np.shape[1] != 1:
            return False, "Prediction data must have exactly 1 channel"
        if predictions_np.shape[0] != 100:
            return False, "Invalid number of samples in prediction data"
    elif len(predictions_np.shape) == 3:
        # (n, 180, 320)
        if predictions_np.shape[1:] != expected_shape_suffix[1:]:
            return False, "Prediction data has incorrect dimensions"
        if predictions_np.shape[0] != 100:
            return False, "Invalid number of samples in prediction data"
    else:
        return False, "Incorrect dimensions in prediction data"
    
    return True, None


def validate_predictions_values(predictions_np):
    """
    Validate that prediction values are reasonable (real, non-negative, finite).
    """
    if np.iscomplexobj(predictions_np):
        return False, "Prediction data contains complex values"
    
    if not np.isfinite(predictions_np).all():
        return False, "Prediction data contains non-finite values"
    
    if (predictions_np < 0).any():
        return False, "Prediction data contains negative values"
    
    return True, None


def safe_evaluate_predictions(predictions_np, targets_np):
    """
    Safely evaluate predictions with error handling for mathematical operations.
    """
    try:
        N = predictions_np.shape[0]
        preds_sum = predictions_np.reshape(N, -1).sum(axis=1)
        true_sum = targets_np.reshape(N, -1).sum(axis=1)
        
        # Check for invalid sums
        if not np.isfinite(preds_sum).all() or not np.isfinite(true_sum).all():
            return None, "Invalid sum values detected"
        
        diffs = np.abs(preds_sum - true_sum)
        
        # Safe division - handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.abs(1 - preds_sum / true_sum)
            # Replace inf and nan values with a high penalty
            rates = np.where(np.isfinite(rates), rates, 1.0)
        
        mae = diffs.mean()
        mse = (diffs**2).mean()
        rate = rates.mean()
        predict_num_avg = preds_sum.mean()
        true_num_avg = true_sum.mean()
        
        # Check for invalid intermediate results
        if not all(np.isfinite([mae, mse, rate, predict_num_avg, true_num_avg])):
            return None, "Invalid intermediate calculation results"
        
        # Safe exponential calculation
        if rate > 100:  # Prevent exp overflow
            score = 0.0
        else:
            score = math.exp(-rate)
        
        logging.info(f'test ---- Score: {score:.3f}, MSE: {mse:.4f}, MAE: {mae:.4f}, Chicken_avg: {predict_num_avg:.4f}')
        return score, None
        
    except Exception as e:
        logging.error(f"Error in evaluation: {str(e)}")
        return None, "Evaluation calculation failed"


# Add function to evaluate predictions and targets arrays
def evaluate_predictions(predictions_np, targets_np):
    score, error = safe_evaluate_predictions(predictions_np, targets_np)
    if error:
        raise ValueError(error)
    return score


def safe_test(preds, test_path, expected_shape):
    """
    Safely run the test with comprehensive error handling.
    """
    try:
        # Validate prediction shape
        valid_shape, shape_error = validate_predictions_shape(preds, expected_shape)
        if not valid_shape:
            return None, shape_error
        
        # Validate prediction values
        valid_values, values_error = validate_predictions_values(preds)
        if not valid_values:
            return None, values_error
        
        # Load target dataset
        test_dataset = load_from_disk(test_path)
        
        # Extract density data directly as numpy arrays without torch
        targets = []
        for item in test_dataset:
            density = np.array(item["density"], dtype=np.float32)
            # Add batch dimension to match expected shape
            targets.append(density[np.newaxis, :])
        
        # Concatenate all targets into a single numpy array
        targets = np.concatenate(targets, axis=0)
        
        # Remove channel dimension from predictions for shape comparison and evaluation
        # Predictions are (n, 1, 180, 320), targets are (n, 180, 320)
        if len(preds.shape) == 4 and preds.shape[1] == 1:
            preds_squeezed = preds.squeeze(axis=1)  # Remove channel dimension
        else:
            return None, "Invalid prediction format for evaluation"
        
        # Validate that prediction and target shapes match after removing channel
        if preds_squeezed.shape != targets.shape:
            return None, "Prediction and target data shape mismatch"
        
        # Safely evaluate predictions (using squeezed predictions without channel dim)
        score, eval_error = safe_evaluate_predictions(preds_squeezed, targets)
        if eval_error:
            return None, eval_error
        
        # Final safety check: clamp score to [0.0, 1.0]
        if score < 0.0 or score > 1.0:
            logging.warning(f"Score {score} out of valid range, setting to 0.0")
            score = 0.0
        
        return score, None
        
    except Exception as e:
        logging.error(f"Error in test function: {str(e)}")
        return None, "Test execution failed"


# Main function to run the validation
def test(preds, test_path):
    score, error = safe_test(preds, test_path, (1, 180, 320))
    if error:
        raise ValueError(error)
    return score


def create_error_response(error_message):
    """Create standardized error response."""
    return {
        "status": False,
        "score": {
            "public_a": 0.0,
            "private_b": 0.0,
        },
        "msg": f"Error: {error_message}",
    }


def create_success_response(score_a, score_b):
    """Create standardized success response."""
    # 处理 NaN 和 inf，替换为 0.0
    if not np.isfinite(score_a):  # np.isfinite 同时检查 NaN 和 inf
        score_a = 0.0
    if not np.isfinite(score_b):
        score_b = 0.0
    return {
        "status": True,
        "score": {
            "public_a": score_a,
            "private_b": score_b,
        },
        "msg": "Success!",
    }


if __name__ == '__main__':
    ################################################################################
    # Dataset paths
    if os.environ.get('METRIC_PATH'):
        METRIC_PATH = os.environ.get("METRIC_PATH") + "/" 
    else:
        METRIC_PATH = ""  # Fallback for local testing
    testA_path = METRIC_PATH + "test_a_targets"
    testB_path = METRIC_PATH + "test_b_targets"
    
    try:
        # Safely load the npz file
        try:
            preds = np.load("submission.npz", allow_pickle=False)
        except FileNotFoundError:
            ret_json = create_error_response("Submission file not found")
        except Exception as e:
            ret_json = create_error_response("Failed to load submission file")
        else:
            # Check for required keys
            required_keys = ['pred_a', 'pred_b']
            missing_keys = [key for key in required_keys if key not in preds.files]
            
            if missing_keys:
                ret_json = create_error_response(f"Missing required keys in submission file")
            else:
                try:
                    # Extract predictions safely
                    pred_a = preds['pred_a']
                    pred_b = preds['pred_b']
                    
                    logging = logging_level('info')
                    
                    # Test both predictions with error handling
                    score_a, error_a = safe_test(pred_a, testA_path, (1, 180, 320))
                    if error_a:
                        ret_json = create_error_response(f"Error in test A evaluation: {error_a}")
                    else:
                        score_b, error_b = safe_test(pred_b, testB_path, (1, 180, 320))
                        if error_b:
                            ret_json = create_error_response(f"Error in test B evaluation: {error_b}")
                        else:
                            # Final safety check on scores
                            score_a = max(0.0, min(1.0, score_a))
                            score_b = max(0.0, min(1.0, score_b))
                            
                            ret_json = create_success_response(score_a, score_b)
                            
                except Exception as e:
                    logging.error(f"Unexpected error during evaluation: {str(e)}")
                    ret_json = create_error_response("Evaluation failed due to invalid submission format")
    
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        ret_json = create_error_response("Critical evaluation error")
    
    # Write result to file
    try:
        with open('score.json', 'w') as f:
            f.write(json.dumps(ret_json))
    except Exception as e:
        logging.error(f"Failed to write score file: {str(e)}")
