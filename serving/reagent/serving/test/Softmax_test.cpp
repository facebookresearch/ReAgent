#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/operators/Softmax.h"

namespace reagent {

TEST(SoftmaxTests, OneUniform) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  StringDoubleMap values = {{"1", 3.0}, {"2", 3.0}};
  namedInputs["values"] = (values);
  namedInputs["temperature"] = (1.0);

  StringDoubleMap expectedOutput = {{"1", 0.5}, {"2", 0.5}};
  EXPECT_SYMBOLTABLE_NEAR(
      std::get<StringDoubleMap>(Softmax("softmax", PLAN_NAME, {}, service.get())
                                    .run(DecisionRequest(), namedInputs)),
      expectedOutput);
}

}  // namespace ml
