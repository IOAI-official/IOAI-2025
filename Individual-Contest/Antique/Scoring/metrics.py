# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score
import zipfile
import os

if __name__ == '__main__':
    with zipfile.ZipFile('submission.zip', 'r') as zip_ref:
        zip_ref.extractall('submission/')
    if os.environ.get('ANSWER_PATH'):
        ANSWER_PATH = os.environ.get("ANSWER_PATH") + "/" 
    else:
        ANSWER_PATH = ""  # Fallback for local testing
    # A榜
    predA_dir = "submission/submissionA.csv"
    test_dir = ANSWER_PATH + "label.csv"
    y_predA = pd.read_csv(predA_dir, header=None)
    y_test = pd.read_csv(test_dir)
    accuracy_A = accuracy_score(y_predA, y_test['validation_label'])
    if accuracy_A > 1:
        accuracy_A = 0
    print(f"Accuracy for test A: {accuracy_A:.2f}")
    # B榜
    predB_dir = "submission/submissionB.csv"
    y_predB = pd.read_csv(predB_dir, header=None)
    accuracy_B = accuracy_score(y_predB, y_test['testing_label'])
    if accuracy_B > 1:
        accuracy_B = 0
    print(f"Accuracy for test B: {accuracy_B:.2f}")
    #----------calculate the score on the leaderboard------------#
    score = {
        "public_a": accuracy_A,
        "public_detail": {
            "Accuracy": accuracy_A,
        },
        "private_b": accuracy_B,
        "private_detail":{
            "Accuracy": accuracy_B,
        },
    }
    #print(score)
    ret_json = {
        "status": True,
        "score": score,
        "msg": "Success!",
    }
    with open('score.json', 'w') as f:
        f.write(json.dumps(ret_json))    
    
