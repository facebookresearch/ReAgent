#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/operators/Expression.h"

namespace reagent {

TEST(ExpressionTests, Simple) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  namedInputs["equation"] = "(x*y)^z";

  StringDoubleMap x = {{"e1", 2.0}, {"e2", 2.0}, {"e3", 2.0}};
  namedInputs["x"] = x;
  StringDoubleMap y = {{"e1", 3.0}, {"e2", 3.0}, {"e3", 3.0}};
  namedInputs["y"] = y;
  StringDoubleMap z = {{"e1", 1.0}, {"e2", 2.0}, {"e3", 3.0}};
  namedInputs["z"] = z;

  Expression expression("expression", PLAN_NAME, {}, service.get());

  StringDoubleMap expectedOutput = {{"e1", 6.0}, {"e2", 36.0}, {"e3", 216.0}};
  EXPECT_SYMBOLTABLE_NEAR(
      std::get<StringDoubleMap>(expression.run(DecisionRequest(), namedInputs)),
      expectedOutput);
}

}  // namespace ml
