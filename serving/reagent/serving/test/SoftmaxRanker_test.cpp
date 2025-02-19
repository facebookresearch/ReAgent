#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/operators/SoftmaxRanker.h"

namespace reagent {

TEST(SoftmaxRankerTests, SimpleSort) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  StringDoubleMap values = {{"1", 1.0}, {"2", 1000.0}};
  namedInputs["values"] = (values);
  namedInputs["temperature"] = (0.01);
  namedInputs["seed"] = (int64_t(1));

  auto result = std::get<RankedActionList>(
      SoftmaxRanker("softmaxranker", PLAN_NAME, {}, service.get())
          .run(DecisionRequest(), namedInputs));

  StringList expectedResult = {"2", "1"};
  for (int i = 0; i < int(expectedResult.size()); i++) {
    EXPECT_EQ(result[i].name, expectedResult[i]);
  }
}

} // namespace reagent
