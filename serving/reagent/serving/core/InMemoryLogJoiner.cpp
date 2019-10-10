#include "reagent/serving/core/InMemoryLogJoiner.h"

namespace reagent {
void InMemoryLogJoiner::logDecision(
    const DecisionRequest& decisionRequest,
    const DecisionResponse& decisionResponse,
    const StringOperatorDataMap& operator_outputs) {
  std::string key = decisionRequest.request_id;
  DecisionWithFeedback decisionWithFeedback;
  decisionWithFeedback.decision_request = decisionRequest;
  decisionWithFeedback.decision_response = decisionResponse;
  decisionWithFeedback.operator_outputs = operator_outputs;
  unjoinedData_[key] = std::move(decisionWithFeedback);
}

void InMemoryLogJoiner::logFeedback(Feedback feedback) {
  addRewardToFeedback(&feedback);
  std::string request_id = feedback.request_id;
  auto it = unjoinedData_.find(request_id);
  if (it == unjoinedData_.end()) {
    return;
  }
  DecisionWithFeedback dwf1 = it->second;
  DecisionWithFeedback dwf2;
  dwf2.feedback = feedback;
  unjoinedData_.erase(it);
  json j1 = dwf1;
  json j2 = dwf2;
  deserializeAndJoinDecisionAndFeedback({j1.dump(), j2.dump()});
}

DecisionWithFeedback InMemoryLogJoiner::deserializeAndJoinDecisionAndFeedback(
    StringList decisionAndFeedback) {
  DecisionWithFeedback mergedDwf =
      LogJoiner::deserializeAndJoinDecisionAndFeedback(decisionAndFeedback);
  loggedData_[mergedDwf.decision_request->request_id] = mergedDwf;
  nlohmann::json json;
  to_json(json, mergedDwf);
  if (logStream_ != nullptr) {
    (*logStream_) << json.dump() << std::endl;
  }
  return mergedDwf;
}

std::unordered_map<std::string, DecisionWithFeedback>
InMemoryLogJoiner::getLoggedData() {
  return loggedData_;
}

} // namespace reagent
