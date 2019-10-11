#include "reagent/serving/core/LogJoiner.h"

#include "reagent/serving/core/DecisionService.h"

namespace reagent {
void LogJoiner::addRewardToFeedback(Feedback* feedback) {}

DecisionWithFeedback LogJoiner::deserializeAndJoinDecisionAndFeedback(
    StringList decisionAndFeedback) {
  if (decisionAndFeedback.size() != 2) {
    LOG_AND_THROW("Somehow ended up with more than 2 values for the same key: "
                  << decisionAndFeedback.size());
  }

  DecisionWithFeedback first = json::parse(decisionAndFeedback.at(0));
  const DecisionWithFeedback& second = json::parse(decisionAndFeedback.at(1));
  if (bool(first.feedback)) {
    if (bool(second.feedback)) {
      LOG_AND_THROW("Got two feedbacks for the same key");
    }
    first.decision_request = second.decision_request;
    first.decision_response = second.decision_response;
    first.operator_outputs = second.operator_outputs;
  } else {
    if (bool(second.decision_request)) {
      LOG_AND_THROW("Got two requests for the same key");
    }
    first.feedback = second.feedback;
  }
  if (decisionService_) {
    decisionService_->_giveFeedback(first);
  }
  return first;
}

}  // namespace reagent
