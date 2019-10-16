#!/usr/bin/env python3

import json
import sys


input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as input_fp:
    with open(output_file, "w") as output_fp:
        while True:
            input_line = input_fp.readline()
            if not input_line:
                break
            input_line_json = json.loads(input_line)
            output_line = {
                "ds": "2019-01-01",
                "mdp_id": input_line_json["decision_request"]["request_id"],
                "sequence_number": 0,  # TODO: Support sequences
                "state_features": input_line_json["decision_request"][
                    "context_features"
                ],
                "action": input_line_json["decision_response"]["actions"][0]["name"],
                "reward": input_line_json["feedback"]["actions"][0]["computed_reward"],
                "action_probability": input_line_json["decision_response"]["actions"][
                    0
                ]["propensity"],
                "possible_actions": input_line_json["decision_request"]["actions"][
                    "names"
                ],
                "metrics": input_line_json["feedback"]["actions"][0]["metrics"],
            }
            output_fp.write(json.dumps(output_line) + "\n")
