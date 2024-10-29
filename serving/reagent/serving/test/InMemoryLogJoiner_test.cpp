#include "reagent/serving/core/InMemoryLogJoiner.h"
#include "reagent/serving/test/TestHeaders.h"

namespace reagent {
TEST(InMemoryLogJoinerTests, Match) {
  auto service = makeTestDecisionService();
  InMemoryLogJoiner logJoiner;

  const auto PLAN_NAME = std::string("/");
  const auto REQUEST_ID = std::string("abc");

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
  service->createPlan(PLAN_NAME, decisionConfig);

  DecisionRequest request;
  request.request_id = REQUEST_ID;
  request.plan_name = PLAN_NAME;

  DecisionResponse response;
  response.request_id = REQUEST_ID;

  StringOperatorDataMap history;

  Feedback feedback;
  feedback.request_id = REQUEST_ID;
  feedback.plan_name = PLAN_NAME;

  logJoiner.logDecision(request, response, history);
  logJoiner.logFeedback(feedback);

  auto loggedData = logJoiner.getLoggedData();
  EXPECT_TRUE(loggedData.size() == 1);
  EXPECT_TRUE(loggedData.at(0).decision_response->request_id == REQUEST_ID);
}

TEST(InMemoryLogJoinerTests, NotMatch) {
  auto service = makeTestDecisionService();
  InMemoryLogJoiner logJoiner;

  const auto PLAN_NAME = std::string("/");
  const auto REQUEST_ID = std::string("abc");

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
  service->createPlan(PLAN_NAME, decisionConfig);

  DecisionRequest request;
  request.request_id = REQUEST_ID;
  request.plan_name = PLAN_NAME;

  DecisionResponse response;
  response.request_id = REQUEST_ID;

  StringOperatorDataMap history;

  Feedback feedback;
  feedback.request_id = REQUEST_ID;
  feedback.plan_name = PLAN_NAME;

  logJoiner.logDecision(request, response, history);
  logJoiner.logFeedback(feedback);

  auto loggedData = logJoiner.getLoggedData();
  EXPECT_TRUE(loggedData.size() == 1);
  EXPECT_TRUE(loggedData.at(0).decision_response->request_id == REQUEST_ID);
}

} // namespace reagent
