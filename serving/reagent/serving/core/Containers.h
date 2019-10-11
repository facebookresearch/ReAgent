#pragma once

#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace reagent {
using nlohmann::json;

using StringStringMap = std::unordered_map<std::string, std::string>;
using StringDoubleMap = std::unordered_map<std::string, double>;
using StringIntMap = std::unordered_map<std::string, int64_t>;
using StringList = std::vector<std::string>;

struct ActionDetails {
  std::string name;
  double propensity;  // Probability of choosing this action
};

inline void from_json(const json& j, ActionDetails& p) {
  p.name = j.at("name");
  p.propensity = j.at("propensity");
}

inline void to_json(json& j, const ActionDetails& p) {
  j = json{{"name", p.name}, {"propensity", p.propensity}};
}

using RankedActionList = std::vector<ActionDetails>;

typedef std::variant<
    std::string, int64_t, double, StringList, std::vector<int64_t>,
    std::vector<double>, StringStringMap, StringIntMap, StringDoubleMap,
    std::unordered_map<std::string, StringDoubleMap>, RankedActionList>
    ConstantValue;

ConstantValue json_to_constant_value(const json& j);
void constant_value_to_json(const ConstantValue& value, json& j);
}  // namespace reagent

namespace std {
inline void from_json(const nlohmann::json& j, reagent::ConstantValue& p) {
  p = reagent::json_to_constant_value(j);
}

inline void to_json(nlohmann::json& j, const reagent::ConstantValue& p) {
  reagent::constant_value_to_json(p, j);
}

template <class T>
inline void from_json(const nlohmann::json& j, std::optional<T>& p) {
  if (!j.is_null()) {
    p = j.get<T>();
  } else {
    p = std::nullopt;
  }
}

template <class T>
inline void to_json(nlohmann::json& j, const std::optional<T>& p) {
  if (bool(p)) {
    j = *p;
  } else {
    j = nullptr;
  }
}
}  // namespace std

namespace reagent {

struct OperatorDefinition {
  std::string name;
  std::string op_name;
  StringStringMap input_dep_map;
};

inline void from_json(const json& j, OperatorDefinition& p) {
  j.at("name").get_to(p.name);
  j.at("op_name").get_to(p.op_name);
  j.at("input_dep_map").get_to(p.input_dep_map);
}

using OperatorData = ConstantValue;
using StringOperatorDataMap = std::unordered_map<std::string, OperatorData>;

struct Constant {
  std::string name;
  ConstantValue value;
};

inline void from_json(const json& j, Constant& p) {
  j.at("name").get_to(p.name);
  j.at("value").get_to(p.value);
}

enum DecisionRewardAggreation {
  DRA_INVALID = 0,
  DRA_SUM = 1,
  DRA_MAX = 2,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DecisionRewardAggreation,
                             {
                                 {DRA_INVALID, nullptr},
                                 {DRA_SUM, "sum"},
                                 {DRA_MAX, "max"},
                             });

/*
  DecisionConfig represents a single policy and is stored in configurator.

  We can use QE to fetch different DecisionConfigs based on user_id.
*/
struct DecisionConfig {
  std::vector<OperatorDefinition>
      operators;  // List of operations to perform.  The last
                  // item in the list produces the final output.
  std::vector<Constant> constants;    // List of constants that can be fed to
                                      // operators.
  int64_t num_actions_to_choose = 1;  // Repeat the policy to sample N actions
                                      // w/o replacement (such as for ranking).
  std::string reward_function;
  DecisionRewardAggreation reward_aggregator;
};

inline void from_json(const json& j, DecisionConfig& p) {
  j.at("operators").get_to(p.operators);
  j.at("constants").get_to(p.constants);
  j.at("num_actions_to_choose").get_to(p.num_actions_to_choose);
  j.at("reward_function").get_to(p.reward_function);
  j.at("reward_aggregator").get_to(p.reward_aggregator);
}

using FeatureSet = StringDoubleMap;

struct DecisionRequestActionSet {
  std::unordered_map<std::string, FeatureSet>
      features;      // Set of name->example pairs
  StringList names;  // Discrete list of actions
};

inline void from_json(const json& j, DecisionRequestActionSet& p) {
  p.features =
      j.value<std::unordered_map<std::string, FeatureSet>>("features", {});
  j.at("names").get_to(p.names);
}

inline void to_json(json& j, const DecisionRequestActionSet& p) {
  j["features"] = p.features;
  j["names"] = p.names;
}

struct DecisionRequestMeta {};

inline void from_json(const json&, DecisionRequestMeta&) {}

inline void to_json(json&, const DecisionRequestMeta&) {}

struct DecisionRequest {
  std::string request_id;  // UUID for this request
  DecisionRequestMeta meta_data;
  std::string plan_name;  // Name of DecisionConfig for this request
  DecisionRequestActionSet actions;
  StringDoubleMap context_features;  // Action-independent features
  ConstantValue input;               // Extra inputs
};

inline void from_json(const json& j, DecisionRequest& p) {
  p.request_id = j.value("request_id", "");
  if (j.count("meta_data")) {
    j.at("meta_data").get_to(p.meta_data);
  }
  j.at("plan_name").get_to(p.plan_name);
  j.at("actions").get_to(p.actions);
  if (j.count("context_features")) {
    j.at("context_features").get_to(p.context_features);
  }
  if (j.count("input")) {
    j.at("input").get_to(p.input);
  }
}

inline void to_json(json& j, const DecisionRequest& p) {
  j["request_id"] = p.request_id;
  j["meta_data"] = p.meta_data;
  j["plan_name"] = p.plan_name;
  j["actions"] = p.actions;
  j["context_features"] = p.context_features;
  j["input"] = p.input;
}

struct DecisionResponse {
  std::string request_id;    // UUID echoed back from the request
  std::string plan_name;     // Name of DecisionConfig for the request
  RankedActionList actions;  // Stats on all actions
  int64_t duration;          // Time taken to process this request
};

inline void from_json(const json& j, DecisionResponse& p) {
  p.request_id = j.at("request_id");
  p.plan_name = j.at("plan_name");
  p.actions = j.at("actions").get<RankedActionList>();
  p.duration = j.at("duration");
}

inline void to_json(json& j, const DecisionResponse& p) {
  j = json{{"request_id", p.request_id},
           {"plan_name", p.plan_name},
           {"actions", p.actions},
           {"duration", p.duration}};
}

struct ActionFeedback {
  std::string name;
  StringDoubleMap metrics;
  std::optional<double> computed_reward;
};

inline void from_json(const json& j, ActionFeedback& p) {
  j.at("name").get_to(p.name);
  j.at("metrics").get_to(p.metrics);
  if (j.count("computed_reward")) {
    j.at("computed_reward").get_to(p.computed_reward);
  }
}

inline void to_json(json& j, const ActionFeedback& p) {
  j["name"] = p.name;
  j["metrics"] = p.metrics;
  j["computed_reward"] = p.computed_reward;
}

struct Feedback {
  std::string plan_name;
  std::string request_id;
  std::vector<ActionFeedback> actions;
  std::optional<double> computed_reward;
};

inline void from_json(const json& j, Feedback& p) {
  j.at("plan_name").get_to(p.plan_name);
  j.at("request_id").get_to(p.request_id);
  j.at("actions").get_to(p.actions);
  if (j.count("computed_reward")) {
    double d;
    j.at("computed_reward").get_to(d);
    p.computed_reward = d;
  }
}

inline void to_json(json& j, const Feedback& p) {
  j["plan_name"] = p.plan_name;
  j["request_id"] = p.request_id;
  j["actions"] = p.actions;
  if (bool(p.computed_reward)) {
    j["computed_reward"] = *(p.computed_reward);
  }
}

struct DecisionWithFeedback {
  std::optional<DecisionRequest> decision_request;
  std::optional<DecisionResponse> decision_response;
  std::optional<StringOperatorDataMap> operator_outputs;
  std::optional<Feedback> feedback;
};

inline void from_json(const json& j, DecisionWithFeedback& p) {
  j.at("decision_request").get_to(p.decision_request);
  j.at("decision_response").get_to(p.decision_response);
  j.at("operator_outputs").get_to(p.operator_outputs);
  j.at("feedback").get_to(p.feedback);
}

inline void to_json(json& j, const DecisionWithFeedback& p) {
  j["decision_request"] = p.decision_request;
  j["decision_response"] = p.decision_response;
  j["operator_outputs"] = p.operator_outputs;
  j["feedback"] = p.feedback;
}
}  // namespace reagent
