#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/ActionValueScorer.h"
#include "reagent/serving/core/ConfigProvider.h"
#include "reagent/serving/core/DecisionPlan.h"
#include "reagent/serving/core/LogJoiner.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/core/RealTimeCounter.h"
#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {

class DecisionService {
 public:
  DecisionService(
      std::shared_ptr<ConfigProvider> configProvider,
      std::shared_ptr<ActionValueScorer> actionValueScorer,
      std::shared_ptr<LogJoiner> logJoiner,
      std::shared_ptr<RealTimeCounter> realTimeCounter,
      std::shared_ptr<SharedParameterHandler> sharedParameterHandler);

  virtual ~DecisionService() {}

  DecisionResponse process(const DecisionRequest& request);

  void computeRewardAndLogFeedback(Feedback feedback);

  virtual void deserializeAndJoinDecisionAndFeedback(
      StringList decisionAndFeedback) {
    logJoiner_->deserializeAndJoinDecisionAndFeedback(decisionAndFeedback);
  }

  std::shared_ptr<DecisionPlan> createPlan(
      const std::string& planName,
      const DecisionConfig& config);

  std::shared_ptr<DecisionPlan> getPlan(const std::string& name);

  inline std::shared_ptr<ConfigProvider> getConfigProvider() const {
    return configProvider_;
  }

  inline std::shared_ptr<ActionValueScorer> getActionValueScorer() const {
    return actionValueScorer_;
  }

  inline std::shared_ptr<LogJoiner> getLogJoiner() const {
    return logJoiner_;
  }

  inline std::shared_ptr<RealTimeCounter> getRealTimeCounter() const {
    return realTimeCounter_;
  }

  inline std::shared_ptr<SharedParameterHandler> getSharedParameterHandler()
      const {
    return sharedParameterHandler_;
  }

  void _giveFeedback(const DecisionWithFeedback& decisionWithFeedback);

 protected:
  std::unordered_map<std::string, std::shared_ptr<DecisionPlan>> planContainer_;
  OperatorRunner runner_;
  std::mutex planMutex_;

  std::shared_ptr<ConfigProvider> configProvider_;
  std::shared_ptr<ActionValueScorer> actionValueScorer_;
  std::shared_ptr<LogJoiner> logJoiner_;
  std::shared_ptr<RealTimeCounter> realTimeCounter_;
  std::shared_ptr<SharedParameterHandler> sharedParameterHandler_;
};

} // namespace reagent
