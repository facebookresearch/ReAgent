#!/usr/bin/env python3

import json
import random
import sys
import time
import urllib.error
import urllib.request
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple


random.seed(0)


def post(url: str, content: Any) -> Any:
    last_error: Optional[urllib.error.HTTPError] = None
    for _retry in range(10):
        try:
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json; charset=utf-8")
            jsondataasbytes = json.dumps(content).encode("utf-8")  # needs to be bytes
            req.add_header("Content-Length", str(len(jsondataasbytes)))
            response = urllib.request.urlopen(req, jsondataasbytes)
            assert (
                response.getcode() == 200
            ), "Error making request to ReAgent server: {} {}".format(
                response.getcode(), response.read().decode("utf-8")
            )
            return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print("Error: {} {}".format(e.getcode(), e.read().decode("utf-8")))
            last_error = e
            time.sleep(1)
    raise last_error


plan_name = sys.argv[1]
EPOCHS = 1000


def serve_customer(epoch) -> Tuple[str, float]:
    if epoch % 100 == 0:
        print(epoch)

    # 10% chance to be rib lover
    rib_lover = random.random() <= 0.1

    result = post(
        "http://localhost:3000/api/request",
        {
            "plan_name": plan_name,
            "context_features": {0: float(rib_lover), 1: 1.0},
            "actions": {"names": ["Bacon", "Ribs"]},
        },
    )
    action_taken = result["actions"][0]["name"]
    if action_taken == "Bacon":
        # 50% chance of click
        reward = 1.0 if random.random() <= 0.5 else 0.0
    elif action_taken == "Ribs":
        if rib_lover:
            # Rib lover: 90% chance of click
            reward = 1.0 if random.random() <= 0.9 else 0.0
        else:
            # Not a rib lover, 10% chance of click
            reward = 1.0 if random.random() <= 0.1 else 0.0

    action_feedback = {"name": action_taken, "metrics": {"reward": reward}}
    post(
        "http://localhost:3000/api/feedback",
        {
            "request_id": result["request_id"],
            "plan_name": plan_name,
            "actions": [action_feedback],
        },
    )

    return action_taken, reward


p = Pool(32)
results: List[Tuple[str, float]] = p.map(serve_customer, list(range(EPOCHS)))
total_reward = 0.0
action_histogram: Dict[str, int] = {}
for result in results:
    action_taken, reward = result
    total_reward += reward
    action_histogram[action_taken] = action_histogram.get(action_taken, 0) + 1

print("Average reward:", (total_reward / EPOCHS))
print("Action Distribution:", str(action_histogram))
