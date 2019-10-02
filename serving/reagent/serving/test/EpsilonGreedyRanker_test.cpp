#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/operators/EpsilonGreedyRanker.h"

namespace reagent {

TEST(EpsilonGreedyRankerTests, EpsilonZero) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  StringDoubleMap values = {{"0", 0.0}, {"1", 20}};
  namedInputs["values"] = values;
  namedInputs["epsilon"] = 0.0;

  auto result = std::get<StringList>(
      EpsilonGreedyRanker("epsilongreedyranker", PLAN_NAME, {}, service.get())
          .run(DecisionRequest(), namedInputs));

  StringList expectedResult = {"1", "0"};
  for (int i = 0; i < expectedResult.size(); i++) {
    EXPECT_EQ(result[i], expectedResult[i]);
  }
}

TEST(EpsilonGreedyRankerTests, EpsilonGTZero) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  StringDoubleMap values = {{"0", 0.0}, {"1", 20}};
  namedInputs["values"] = values;
  namedInputs["epsilon"] = 0.3;

  // Run the ranker 1000 times, this test has a low false negative rate
  int sum = 0;
  for (int i = 0; i < 1000; ++i) {
    StringList result = std::get<StringList>(
        EpsilonGreedyRanker("epsilongreedyranker", PLAN_NAME, {}, service.get())
            .run(DecisionRequest(), namedInputs));
    if (result[0] == "0") {
      sum++;
    }
  }

  // 0 should be selected ~(N * E * 0.5) = 150 times
  EXPECT_TRUE(sum < 300);
  EXPECT_TRUE(sum > 50);
}

} // namespace ml
