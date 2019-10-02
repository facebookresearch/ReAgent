#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/ConfigProvider.h"
#include "reagent/serving/core/OperatorRunner.h"

namespace reagent {

TEST(ConfigProvider, Simple) {
  auto service = makeTestDecisionService();
  auto provider = service->getConfigProvider();
  OperatorRunner runner;
  DecisionConfig decisionConfig;

  {
    ConstantValue equationValueThrift;
    equationValueThrift = ("input^2");

    Constant expressionThrift;
    expressionThrift.name = ("equation_value");
    expressionThrift.value = (equationValueThrift);

    OperatorDefinition expressionOpThrift;
    expressionOpThrift.name = ("expression_op");
    expressionOpThrift.op_name = ("Expression");
    expressionOpThrift.input_dep_map = {{"equation", "equation_value"},
                                        {"input", "input"}};

    decisionConfig.operators = {expressionOpThrift};
    decisionConfig.constants = {expressionThrift};
  }

  service->createPlan("/", decisionConfig);
  auto plan = service->getPlan("/");
  auto ops = plan->getOperators();
  auto constants = plan->getConstants();

  OperatorData input = StringDoubleMap({{"x", 2.0}, {"y", 3.0}, {"z", 4.0}});
  StringDoubleMap expectedOutput = {{"x", 4.0}, {"y", 9.0}, {"z", 16.0}};

  auto output = std::get<StringDoubleMap>(
      runner.run(ops, constants, DecisionRequest(), input).at("expression_op"));

  EXPECT_SYMBOLTABLE_NEAR(output, expectedOutput);
}

}  // namespace ml
