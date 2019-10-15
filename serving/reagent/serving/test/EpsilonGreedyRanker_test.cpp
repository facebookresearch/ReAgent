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

  auto result = std::get<RankedActionList>(
      EpsilonGreedyRanker("epsilongreedyranker", PLAN_NAME, {}, service.get())
          .run(DecisionRequest(), namedInputs));

  RankedActionList expectedResult = {{"1",1.0}, {"0",1.0}};
  EXPECT_RANKEDACTIONLIST_NEAR(result, expectedResult);
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
    RankedActionList result = std::get<RankedActionList>(
        EpsilonGreedyRanker("epsilongreedyranker", PLAN_NAME, {}, service.get())
            .run(DecisionRequest(), namedInputs));
    if (result[0].name == "0") {
      sum++;
    }
  }

  // 0 should be selected ~(N * E * 0.5) = 150 times
  EXPECT_TRUE(sum < 300);
  EXPECT_TRUE(sum > 50);
}

} // namespace ml
