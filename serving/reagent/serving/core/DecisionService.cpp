#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/LogJoiner.h"

#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
DecisionService::DecisionService(
    std::shared_ptr<ConfigProvider> configProvider,
    std::shared_ptr<ActionValueScorer> actionValueScorer,
    std::shared_ptr<LogJoiner> logJoiner,
    std::shared_ptr<RealTimeCounter> realTimeCounter,
    std::shared_ptr<SharedParameterHandler> sharedParameterHandler)
    : configProvider_(configProvider),
      actionValueScorer_(actionValueScorer),
      logJoiner_(logJoiner),
      realTimeCounter_(realTimeCounter),
      sharedParameterHandler_(sharedParameterHandler) {
  configProvider_->initialize(this);
  logJoiner->registerDecisionService(this);
}

std::shared_ptr<DecisionPlan> DecisionService::getPlan(
    const std::string& name) {
  std::lock_guard<std::mutex> lock(planMutex_);
  auto it = planContainer_.find(name);
  if (it == planContainer_.end()) {
    LOG_AND_THROW("Tried to get a config that doesn't exist: " << name);
  }

  return it->second;
}

DecisionResponse DecisionService::attachIdAndProcess(DecisionRequest request) {
  const auto request_start_time = std::chrono::steady_clock::now();

  if (request.request_id.empty()) {
    request.request_id = generateUuid4();
  }
  auto plan = getPlan(request.plan_name);

  OperatorData inputOperatorData = request.input;
  auto allOperatorData = runner_.run(plan->getOperators(), plan->getConstants(),
                                     request, inputOperatorData);
  auto outputDataIt = allOperatorData.find(plan->getOutputOperatorName());
  if (outputDataIt == allOperatorData.end()) {
    LOG_AND_THROW("Output op missing from plan "
                  << request.plan_name << " " << plan->getOutputOperatorName());
  }
  const RankedActionList outputData =
      std::get<RankedActionList>(outputDataIt->second);
  DecisionResponse response;
  response.request_id = request.request_id;
  response.plan_name = request.plan_name;
  response.actions = outputData;
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - request_start_time);
  response.duration = duration.count();
  logJoiner_->logDecision(request, response, allOperatorData);
  return response;
}

void DecisionService::computeRewardAndLogFeedback(Feedback feedback) {
  auto plan = getPlan(feedback.plan_name);

  // Turn metrics -> rewards
  double aggregateReward = 0.0;
  for (auto& it : feedback.actions) {
    double reward = 0.0;
    if (!plan->getConfig().reward_function.empty()) {
      reward = plan->evaluateRewardFunction(it.metrics);
    }

    it.computed_reward = reward;
    switch (plan->getConfig().reward_aggregator) {
      case DRA_MAX:
        aggregateReward = std::max(aggregateReward, reward);
        break;
      case DRA_SUM:
        aggregateReward = aggregateReward + reward;
        break;
      default:
        LOG_AND_THROW("Invalid reward aggregator");
    }
  }
  feedback.computed_reward = aggregateReward;

  logJoiner_->logFeedback(feedback);
}

void DecisionService::_giveFeedback(
    const DecisionWithFeedback& decisionWithFeedback) {
  if (!bool(decisionWithFeedback.decision_request) ||
      !bool(decisionWithFeedback.decision_response) ||
      !bool(decisionWithFeedback.operator_outputs) ||
      !bool(decisionWithFeedback.feedback)) {
    json j = decisionWithFeedback;
    LOG(FATAL) << "Decision with feedback is missing some elements: "
               << j.dump();
  }
  const auto& request = *(decisionWithFeedback.decision_request);
  const auto& operatorOutputs = *(decisionWithFeedback.operator_outputs);
  const auto& feedback = *(decisionWithFeedback.feedback);

  auto plan = getPlan(request.plan_name);

  // Update all operators with rewards
  for (const auto& op : plan->getOperators()) {
    StringOperatorDataMap namedInputs;
    for (const auto& inputDepEntry : op->getInputDepMap()) {
      const auto& inputName = inputDepEntry.first;
      const auto& depOperatorName = inputDepEntry.second;
      auto it = operatorOutputs.find(depOperatorName);
      if (it == operatorOutputs.end()) {
        LOG_AND_THROW("Could not find data of "
                      << depOperatorName
                      << " for finished operator: " << op->getName());
      }
      namedInputs[inputName] = it->second;
    }
    auto it = operatorOutputs.find(op->getName());
    if (it == operatorOutputs.end()) {
      LOG_AND_THROW(
          "Could not find data for finished operator: " << op->getName());
    }
    auto const& pastOutput = it->second;
    op->giveFeedback(feedback, namedInputs, pastOutput);
  }
}

std::shared_ptr<DecisionPlan> DecisionService::createPlan(
    const std::string& planName, const DecisionConfig& config) {
  std::vector<std::shared_ptr<Operator>> operators =
      OperatorFactory::getInstance()->createFromConfig(planName, config, this);
  if (operators.empty()) {
    LOG_AND_THROW("Plan has no operators");
  }
  // Create constants
  StringOperatorDataMap constants;
  for (const auto& it : config.constants) {
    const std::string& constantName = it.name;
    constants[constantName] = it.value;
  }
  auto plan = std::make_shared<DecisionPlan>(config, operators, constants);

  std::lock_guard<std::mutex> lock(planMutex_);
  planContainer_[planName] = plan;
  return plan;
}

}  // namespace reagent
