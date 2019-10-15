#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/ConfigProvider.h"
#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/OperatorRunner.h"

namespace reagent {
TEST(DecisionService, Simple) {
  auto service = makeTestDecisionService();

  DecisionConfig decisionConfig;

  {
    ConstantValue equationValueThrift;
    equationValueThrift = "input^2";

    Constant expressionThrift;
    expressionThrift.name = "equation_value";
    expressionThrift.value = equationValueThrift;

    OperatorDefinition expressionOpThrift;
    expressionOpThrift.name = "expression_op";
    expressionOpThrift.op_name = "Expression";
    expressionOpThrift.input_dep_map = {{"equation", "equation_value"},
                                        {"input", "input"}};

    ConstantValue epsilonValueThrift;
    epsilonValueThrift = 0.0;

    Constant epsilonThrift;
    epsilonThrift.name = "epsilon_value";
    epsilonThrift.value = epsilonValueThrift;

    OperatorDefinition eGreedyThrift;
    eGreedyThrift.name = "EpsilonGreedyRanker";
    eGreedyThrift.op_name = "EpsilonGreedyRanker";
    eGreedyThrift.input_dep_map = {{"epsilon","epsilon_value"},
				   {"values","expression_op"}};

    decisionConfig.operators = {expressionOpThrift, eGreedyThrift};
    decisionConfig.constants = {expressionThrift, epsilonThrift};
  }

  service->createPlan("/", decisionConfig);

  auto request = DecisionRequest();
  request.plan_name = "/";
  request.actions.names = {"x", "y", "z"};
  request.request_id = "asdf";
  StringDoubleMap input = {{"x", 2.0}, {"y", 3.0}, {"z", 4.0}};
  ConstantValue constantInput;
  constantInput = input;
  request.input = constantInput;

  auto response = service->attachIdAndProcess(request);

  EXPECT_EQ(response.request_id, request.request_id);

  RankedActionList expectedOutput = {{"z", 1.0}, {"y", 1.0}, {"x", 1.0}};

  EXPECT_RANKEDACTIONLIST_NEAR(response.actions, expectedOutput);
}

}  // namespace reagent
