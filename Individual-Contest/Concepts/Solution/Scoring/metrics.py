import os
import pickle
import json
import math
import httpx
from httpx import AsyncClient
from datasets import load_from_disk
from tqdm import tqdm
import json
from asyncio import Semaphore
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

if os.environ.get('DATA_PATH'):
    TEST_PATH = os.environ.get("DATA_PATH") + "/test/" 
else:
    TEST_PATH = ""

TEST_A_LEN = 50
TEST_B_LEN = 100
API_URL = ""
API_KEY = "" # God API key for evaluation
# Please import your LLM API, such as from open router

a_client = AsyncClient()

class APIError(Exception):
    pass

def read_clues(path: str):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def hits_at_10(predictions, correct_answer):
    return 1.0 if correct_answer in predictions[:10] else 0.0

def ndcg_at_10(predictions, correct_answer):
    if correct_answer not in predictions:
        return 0.0
    try:
        rank = predictions[:10].index(correct_answer) + 1
    except ValueError:
        return 0.0
    return 1 / math.log2(rank + 1)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, httpx.RequestError, ValueError))
)
async def get_predictions(clues, options):
    guesser_response = await a_client.post(f"{API_URL}/guess", json={
        "clues": clues,
        "options": options
        }, headers={
            "Authorization": f"Bearer {API_KEY}"
        }, timeout=60)
    guesser_response = guesser_response.json()
    if "guesses" not in guesser_response:
        raise ValueError(f"Unable to generate guesses: {guesser_response}")
    if not isinstance(guesser_response["guesses"], list):
        raise ValueError(f"Guesses is not a list: {guesser_response}")
    predictions = [p.lower() for p in guesser_response["guesses"]]
    return predictions

async def evaluate(clues, testset, test_len, test_name):
    # Validate all inputs first
    for i in range(test_len):
        if not isinstance(clues[i], list):
            raise TypeError(f"`{test_name}[{i}]` must be a list, but got {type(clues[i])}")
        if len(clues[i]) > 4:
            raise ValueError(f"Too many elements in `{test_name}[{i}]`. You can only provide up to 4 sequences of markers.")
        for sequence in clues[i]:
            if not isinstance(sequence, list):
                raise TypeError(f"Each sequence in `{test_name}[{i}]` must be a list, but got {type(sequence)}")
            if len(sequence) > 8:
                raise ValueError(f"Too many markers in `{test_name}[{i}]`. You can only provide up to 8 markers per sequence.")
            for marker in sequence:
                if not isinstance(marker, int):
                    raise TypeError(f"Each marker in `{test_name}[{i}]` must be an integer, but got {type(marker)}")
                if marker < 0 or marker > 117:
                    raise ValueError(f"Invalid marker value in `{test_name}[{i}]`. Expected an integer between 0 and 117.")
    
    # Create semaphore for concurrency control (max 50 parallel requests)
    semaphore = asyncio.Semaphore(50)
    
    async def process_single_item(i):
        async with semaphore:
            try:
                predictions = await get_predictions(clues[i], testset[i]['options'])
                hit10 = hits_at_10(predictions, testset[i]['label'])
                ndcg10 = ndcg_at_10(predictions, testset[i]['label'])
                score = 0.9 * hit10 + 0.1 * ndcg10
                return score

            except Exception as e:
                if isinstance(e, httpx.HTTPStatusError):
                    print(f"HTTP Error  {e.response.status_code}: {e.response.text}")
                    try:
                        error_detail = e.response.json().get("detail", "Unknown error")
                        print(f"Error details: {error_detail}")
                    except:
                        print(f"Could not parse error response {e.response.text}")
                
                elif isinstance(e, ValueError):
                    print(f"Value error: {e}")

                elif isinstance(e, httpx.TimeoutException):
                    print("request timed out")

                elif isinstance(e, httpx.ConnectError):
                    print(f"Could not connect to {API_URL}")

                elif isinstance(e, httpx.RequestError):
                    print(f"Request error: {e}")

                else:
                    print(f"Unknown error: {e}")
                raise APIError(f"Error getting predictions for {test_name}[{i}]", e)
    
    # Create tasks for all items
    tasks = [process_single_item(i) for i in range(test_len)]
    
    # Run all tasks in parallel with progress tracking
    scores = []
    with tqdm(total=test_len, desc=f"Evaluating {test_name}") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                score = await coro
                scores.append(score)
                pbar.update(1)
            except Exception as e:
                pbar.close()
                raise e
    
    return scores

async def main():
    testset_a = load_from_disk(os.path.join(TEST_PATH, "test_a"))["test"]
    testset_b = load_from_disk(os.path.join(TEST_PATH, "test_b"))["test"]
    try:
        clues_a = read_clues("clues_a.jsonl")
        clues_b = read_clues("clues_b.jsonl")
        if not isinstance(clues_a, list):
            raise TypeError(f"`clues_a` must be a list, but got {type(clues_a)}")
        if not isinstance(clues_b, list):
            raise TypeError(f"`clues_b` must be a list, but got {type(clues_b)}")
        if len(clues_a) != TEST_A_LEN:
            raise ValueError(f"{"Too many" if len(clues_a) > TEST_A_LEN else "Too few"} clues in `clues_a`. Expected {TEST_A_LEN} clues.")
        if len(clues_b) != TEST_B_LEN:
            raise ValueError(f"{"Too many" if len(clues_b) > TEST_B_LEN else "Too few"} clues in `clues_b`. Expected {TEST_B_LEN} clues.")
        scores_a = await evaluate(clues_a, testset_a, TEST_A_LEN, "clues_a")
        scores_b = await evaluate(clues_b, testset_b, TEST_B_LEN, "clues_b")
        score_a = sum(scores_a) / len(scores_a)
        score_b = sum(scores_b) / len(scores_b)
        print(f"Average score for clues_a: {score_a}")
        print(f"Average score for clues_b: {score_b}")
        import math

        # 假设 score_a 和 score_b 已经定义
        # 检查并处理 score_a
        if math.isnan(score_a) or math.isinf(score_a):
            score_a = 0

        # 检查并处理 score_b
        if math.isnan(score_b) or math.isinf(score_b):
            score_b = 0

        ret_json = {
            "status": True,
            "score": {
                "public_a": score_a,
                "private_b": score_b,
            },
            "message": "Success!"
        }
        with open("score.json", "w") as f:
            json.dump(ret_json, f)
        
    except Exception as e:
        print(f"Error: {e}")
        ret_json = {
            "status": False,
            "score": {
                "public_a": 0.0,
                "private_b": 0.0,
            },
            "message": str(e)
        }
        with open("score.json", "w") as f:
            json.dump(ret_json, f)
    
if __name__ == "__main__":
    asyncio.run(main())
