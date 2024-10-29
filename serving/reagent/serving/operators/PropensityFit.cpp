#include "reagent/serving/operators/PropensityFit.h"

#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/RealTimeCounter.h"
#include "reagent/serving/core/SharedParameterHandler.h"

#include <gflags/gflags.h>

namespace {
std::string createCounterKey(
    const std::string& configPath,
    const std::string& operatorName,
    const std::string& action) {
  return configPath + "/" + operatorName + "/" + action;
}

} // namespace

namespace reagent {

OperatorData PropensityFit::run(
    const DecisionRequest&,
    const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("input"));
  OperatorData ret;
  ret = run(input);
  return ret;
}

StringDoubleMap PropensityFit::run(const StringDoubleMap& input) {
  auto shifts = sharedParameterHandler_->getValues(getParameterName(planName_));
  StringDoubleMap retval;
  for (const auto& it : input) {
    // shift all parameters
    auto it2 = shifts.find(it.first);
    if (it2 != shifts.end()) {
      retval[it.first] = it.second + it2->second;
    } else {
      retval[it.first] = it.second;
    }
  }
  return retval;
}

void PropensityFit::giveFeedback(
    const Feedback& feedback,
    const StringOperatorDataMap& pastInputs,
    const OperatorData& pastOuptut) {
  const StringDoubleMap& targets =
      std::get<StringDoubleMap>(pastInputs.at("targets"));
  giveFeedbackInternal(
      feedback, pastInputs, std::get<StringDoubleMap>(pastOuptut), targets);
}

void PropensityFit::giveFeedbackInternal(
    const Feedback&,
    const StringOperatorDataMap&,
    const StringDoubleMap& propensities,
    const StringDoubleMap& targets) {
  for (const auto& actionFeedback : propensities) {
    auto it = targets.find(actionFeedback.first);
    if (it == targets.end()) {
      // This action has no target
      continue;
    }
    std::string counterKey =
        createCounterKey(planName_, name_, actionFeedback.first);
    realTimeCounter_->addValue(counterKey, actionFeedback.second);
  }

  // TODO: Implement pid controller to replace this fixed shift
  auto parameterName = getParameterName(planName_);
  if (sharedParameterHandler_->acquireLockToModifyParameter(parameterName)) {
    auto shifts = sharedParameterHandler_->getValues(parameterName);

    for (const auto& it : targets) {
      if (shifts.find(it.first) == shifts.end()) {
        // Missing shift, add it
        shifts[it.first] = 0;
      }
    }

    StringDoubleMap newShifts;
    for (const auto& it : targets) {
      std::string counterKey = createCounterKey(planName_, name_, it.first);
      int64_t numSamples = realTimeCounter_->getNumSamples(counterKey);

      if (numSamples == 0) {
        // No samples, skip
        newShifts[it.first] = shifts[it.first];
        continue;
      }

      double meanPropensity = realTimeCounter_->getMean(counterKey);

      if (meanPropensity > it.second) {
        // Average propensity is above target, begin reducing values
        newShifts[it.first] = shifts[it.first] - 0.01;
      } else {
        // Average propensity is below target, begin raising values
        newShifts[it.first] = shifts[it.first] + 0.01;
      }
    }

    sharedParameterHandler_->updateParameter(parameterName, newShifts);
  }
}

double PropensityFit::getShift(const std::string& actionName) {
  auto parameterName = getParameterName(planName_);
  auto shifts = sharedParameterHandler_->getValues(parameterName);
  return shifts.at(actionName);
}

REGISTER_OPERATOR(PropensityFit, "PropensityFit");

} // namespace reagent
