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

    decisionConfig.operators = {expressionOpThrift};
    decisionConfig.constants = {expressionThrift};
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

  auto response = service->process(request);

  EXPECT_EQ(response.request_id, request.request_id);

  StringDoubleMap expectedOutput = {{"x", 4.0}, {"y", 9.0}, {"z", 16.0}};
  StringDoubleMap output;
  for (auto action : response.actions) {
    output[action.name] = action.propensity;
  }

  EXPECT_SYMBOLTABLE_NEAR(output, expectedOutput);
}

}  // namespace ml
