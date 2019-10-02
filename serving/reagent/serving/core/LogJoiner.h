#pragma once

#include "reagent/serving/core/Headers.h"

namespace reagent {
class DecisionService;

class LogJoiner {
 public:
  LogJoiner() : decisionService_(nullptr) {}

  virtual ~LogJoiner() {}

  virtual void registerDecisionService(DecisionService* decisionService) {
    decisionService_ = decisionService;
  }

  virtual void logDecision(
      const DecisionRequest& request,
      const DecisionResponse& decisionResponse,
      const StringOperatorDataMap& operator_outputs) = 0;

  virtual void logFeedback(Feedback feedback) = 0;

  virtual DecisionWithFeedback deserializeAndJoinDecisionAndFeedback(
      StringList decisionAndFeedback);

 protected:
  DecisionService* decisionService_;

  virtual void addRewardToFeedback(Feedback* feedback);
};

} // namespace reagent
