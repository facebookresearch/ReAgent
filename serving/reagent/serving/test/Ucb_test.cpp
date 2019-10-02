#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/LocalRealTimeCounter.h"
#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/operators/Ucb.h"

namespace reagent {

TEST(UcbTests, Simple) {
  const auto PLAN_NAME = std::string("/");
  auto service = makeTestDecisionService();

  StringOperatorDataMap namedInputs;
  namedInputs["method"] = ("UCB1");

  DecisionRequest request;
  request.request_id = "asdf";
  request.plan_name = PLAN_NAME;
  request.actions.names = {"Arm1", "Arm2"};

  Ucb ucbOp("ucb", PLAN_NAME, {}, service.get());

  std::mt19937 rng(1);

  std::map<std::string, std::bernoulli_distribution> reward_distributions = {
      {"Arm1", std::bernoulli_distribution(0.25)},
      {"Arm2", std::bernoulli_distribution(0.75)}};

  for (int trial = 0; trial < 1000; trial++) {
    std::string result = std::get<std::string>(ucbOp.run(request, namedInputs));
    EXPECT_TRUE(result == std::string("Arm1") || result == std::string("Arm2"));

    // Generate random feedback for chosen action
    Feedback feedback;
    feedback.plan_name = PLAN_NAME;
    feedback.computed_reward = double(reward_distributions.at(result)(rng));
    OperatorData pastOuptut;
    pastOuptut = (result);

    ucbOp.giveFeedback(feedback, {}, pastOuptut);
  }

  // The error on the worse arm can be pretty high if it is excluded early
  EXPECT_NEAR(ucbOp.getArmExpectation("Arm1"), 0.25, 0.5);

  EXPECT_NEAR(ucbOp.getArmExpectation("Arm2"), 0.75, 0.01);
}

TEST(UcbTests, UcbDecisionService) {
  const auto PLAN_NAME = std::string("/");
  auto service = makeTestDecisionService();

  DecisionConfig decisionConfig;

  {
    ConstantValue methodValueThrift;
    methodValueThrift = ("UCB1");

    Constant methodThrift;
    methodThrift.name = ("method");
    methodThrift.value = (methodValueThrift);

    OperatorDefinition ucbOpThrift;
    ucbOpThrift.name = ("ucb");
    ucbOpThrift.op_name = ("Ucb");
    ucbOpThrift.input_dep_map = {{"method", "method"}};

    decisionConfig.operators = {ucbOpThrift};
    decisionConfig.constants = {methodThrift};
    decisionConfig.reward_function = "score";
    decisionConfig.reward_aggregator = DRA_MAX;
  }

  auto decisionPlan = service->createPlan(PLAN_NAME, decisionConfig);
  auto ucbOp =
      std::dynamic_pointer_cast<Ucb>(decisionPlan->getOperators().at(0));

  DecisionRequest request;
  request.plan_name = PLAN_NAME;
  request.actions.names = {"Arm1", "Arm2"};

  std::mt19937 rng(1);

  std::map<std::string, std::bernoulli_distribution> reward_distributions = {
      {"Arm1", std::bernoulli_distribution(0.25)},
      {"Arm2", std::bernoulli_distribution(0.75)}};

  for (int trial = 0; trial < 1000; trial++) {
    request.request_id = std::string("Trial_") + std::to_string(trial);
    auto response = service->process(request);
    for (const auto& it : response.actions) {
      if (it.propensity > 0) {
        std::string result = it.name;
        EXPECT_TRUE(
            result == std::string("Arm1") || result == std::string("Arm2"));

        // Generate random feedback for chosen action
        ActionFeedback actionFeedback;
        actionFeedback.metrics["score"] =
            double(reward_distributions[result](rng));
        actionFeedback.name = result;

        Feedback feedback;
        feedback.request_id = request.request_id;
        feedback.plan_name = PLAN_NAME;
        feedback.actions.push_back(actionFeedback);

        service->computeRewardAndLogFeedback(feedback);
      }
    }
  }

  // The error on the worse arm can be pretty high if it is excluded early
  EXPECT_NEAR(ucbOp->getArmExpectation("Arm1"), 0.25, 0.5);

  EXPECT_NEAR(ucbOp->getArmExpectation("Arm2"), 0.75, 0.01);
}

} // namespace ml
