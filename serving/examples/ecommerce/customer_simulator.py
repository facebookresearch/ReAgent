from typing import Any
import json
import urllib.request, urllib.error
import random


random.seed(0)


def post(url: str, content: Any) -> str:
    req = urllib.request.Request(url)
    req.add_header("Content-Type", "application/json; charset=utf-8")
    jsondataasbytes = json.dumps(content).encode("utf-8")  # needs to be bytes
    req.add_header("Content-Length", str(len(jsondataasbytes)))
    print(jsondataasbytes)
    try:
        response = urllib.request.urlopen(req, jsondataasbytes)
        assert (
            response.getcode() == 200
        ), "Error making request to ReAgent server: {} {}".format(
            response.getcode(), response.read().decode("utf-8")
        )
    except urllib.error.HTTPError as e:
        print("Error: {} {}".format(e.getcode(), e.read().decode("utf-8")))
        raise e
    return json.loads(response.read().decode("utf-8"))


plan_name = "ConstantSoftmax"
total_reward = 0
action_histogram = {}
EPOCHS = 10000
for x in range(EPOCHS):
    result = post(
        "http://localhost:3000/api/request",
        {"plan_name": plan_name, "actions": {"names": ["Bacon", "Ribs"]}},
    )
    action_taken = result["actions"][0]["name"]
    if action_taken == "Bacon":
        # 50% chance of click
        reward = 1.0 if random.random() <= 0.5 else 0.0
    elif action_taken == "Ribs":
        if random.random() <= 0.1:
            # Rib lover: 90% chance of click
            reward = 1.0 if random.random() <= 0.9 else 0.0
        else:
            # Not a rib lover, 10% chance of click
            reward = 1.0 if random.random() <= 0.1 else 0.0
    total_reward += reward
    action_histogram[action_taken] = action_histogram.get(action_taken, 0) + 1

    action_feedback = {"name": action_taken, "metrics": {"reward": reward}}
    port(
        "http://localhost:3000/api/feedback",
        {
            "request_id": result["request_id"],
            "plan_name": plan_name,
            "actions": [action_feedback],
        },
    )

print("Average reward:", (total_reward / EPOCHS))
print("Action Distribution:", str(action_histogram))

