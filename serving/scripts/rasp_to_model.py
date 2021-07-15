#!/usr/bin/env python3

import json
import logging
import sys
from typing import Any, Dict

import pandas as pd


logger = logging.getLogger(__name__)


def keys_to_int(d: Dict[str, Any]) -> Dict[int, Any]:
    new_d = {}
    for k in d:
        new_d[int(k)] = d[k]
    return new_d


input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as input_fp:
    rows = []
    while True:
        input_line = input_fp.readline()
        if not input_line:
            break
        input_line_json = json.loads(input_line)
        output_line = {
            "ds": "2019-01-01",
            "mdp_id": hash(input_line_json["decision_request"]["request_id"]),
            "sequence_number": 0,  # TODO: Support sequences
            "state_features": keys_to_int(
                input_line_json["decision_request"]["context_features"]
            ),
            "action": input_line_json["decision_response"]["actions"][0]["name"],
            "reward": input_line_json["feedback"]["actions"][0]["computed_reward"],
            "action_probability": input_line_json["decision_response"]["actions"][0][
                "propensity"
            ],
            "possible_actions": input_line_json["decision_request"]["actions"]["names"],
            "metrics": input_line_json["feedback"]["actions"][0]["metrics"],
        }
        rows.append(output_line)

    df = pd.DataFrame(rows)
    df.to_pickle(output_file)
    logger.info(f"Saved pickled dataframe to {output_file}. Example rows:")
    logger.info(df.head(5))
