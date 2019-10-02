#include "reagent/serving/operators/Ucb.h"

#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/RealTimeCounter.h"

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

OperatorData Ucb::run(
    const DecisionRequest& request,
    const StringOperatorDataMap& namedInputs) {
  auto input = namedInputs.at("method");
  std::string* methodName = std::get_if<std::string>(&input);
  if (!methodName) {
    LOG_AND_THROW("Input parameter type unexcepted");
  }
  OperatorData ret;
  ret = runInternal(request, *methodName);
  return ret;
}

std::string Ucb::runInternal(
    const DecisionRequest& request,
    const std::string& method) {
  auto arms = Operator::getActionNamesFromRequest(request);

  int64_t totalPulls = 0;
  Eigen::ArrayXd armPulls(arms.size());
  Eigen::ArrayXd rewardMean(arms.size());
  Eigen::ArrayXd rewardVariance(arms.size());

  StringList armsWithoutPulls;

  auto rtc = realTimeCounter_;

  for (int a = 0; a < arms.size(); a++) {
    auto arm = arms[a];
    const auto key = createCounterKey(planName_, name_, arm);

    rewardMean[a] = rtc->getMean(key);
    rewardVariance[a] = rtc->getVariance(key);
    auto pulls = rtc->getNumSamples(key);
    armPulls[a] = pulls;
    totalPulls += pulls;

    if (pulls == 0) {
      armsWithoutPulls.emplace_back(arm);
    }
  }

  std::string armToPull;
  if (armsWithoutPulls.empty()) {
    double logTotalPulls = log(double(totalPulls));

    if (method != std::string("UCB1")) {
      LOG_AND_THROW("Only UCB1 is implemented at the moment");
    }
    // TODO: Implement CDF of t-distribution for bayesian UCB
    auto ucbScore = rewardMean + Eigen::sqrt((2 * logTotalPulls) / armPulls);
    Eigen::Array2d::Index maxIndex;
    ucbScore.maxCoeff(&maxIndex);
    if (arms.size() <= maxIndex) {
      LOG_AND_THROW("Invalid max index: " << arms.size() << " <= " << maxIndex);
    }
    armToPull = arms[maxIndex];
  } else {
    // Pick an empty arm at random
    armToPull =
        armsWithoutPulls[folly::Random::rand32() % armsWithoutPulls.size()];
  }
  return armToPull;
}

void Ucb::giveFeedback(
    const Feedback& feedback,
    const StringOperatorDataMap& /* pastInputs */,
    const OperatorData& pastOutput) {
  auto armName = std::get<std::string>(pastOutput);
  const auto key = createCounterKey(planName_, name_, armName);
  LOG(INFO) << "GIVING FEEDBACK: " << key << " " << *(feedback.computed_reward);
  realTimeCounter_->addValue(key, *(feedback.computed_reward));
}

double Ucb::getArmExpectation(const std::string& armName) {
  const auto key = createCounterKey(planName_, name_, armName);
  LOG(INFO) << "GETTING MEAN: " << key << " " << realTimeCounter_->getMean(key);
  return realTimeCounter_->getMean(key);
}

REGISTER_OPERATOR(Ucb, "Ucb");

} // namespace ml
