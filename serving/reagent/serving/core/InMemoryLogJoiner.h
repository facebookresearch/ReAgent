#pragma once

#include "reagent/serving/core/LogJoiner.h"

namespace reagent {

class InMemoryLogJoiner : public LogJoiner {
 public:
  virtual ~InMemoryLogJoiner() override = default;

  void logDecision(
      const DecisionRequest& request,
      const DecisionResponse& decisionResponse,
      const StringOperatorDataMap& operator_outputs) override;

  void logFeedback(Feedback feedback) override;

  virtual DecisionWithFeedback deserializeAndJoinDecisionAndFeedback(
      StringList decisionAndFeedback) override;

  std::unordered_map<std::string, DecisionWithFeedback> getLoggedData();

 protected:
  std::unordered_map<std::string, DecisionWithFeedback> loggedData_;
  std::unordered_map<std::string, DecisionWithFeedback> unjoinedData_;
};

} // namespace reagent
