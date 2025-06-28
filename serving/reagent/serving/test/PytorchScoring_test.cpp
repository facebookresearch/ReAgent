#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/InMemoryLogJoiner.h"

namespace reagent {

TEST(PytorchActionValueScoringTest, SimpleDiscrete) {
#if 0 // Need to train a libtorch model before we can turn this test on
  const auto PLAN_NAME = std::string("/");
  auto service = std::make_shared<DecisionService>(
      std::make_shared<ConfigProvider>(),
      std::make_shared<PytorchActionValueScorer>(),
      std::make_shared<InMemoryLogJoiner>(),
      std::make_shared<LocalRealTimeCounter>(),
      std::make_shared<SharedParameterHandler>());

  FeatureSet context;

  // state == top-left cell in gridworld
  context[0] = 1.0;

  StringDoubleMap expectedOutput = {{"U", 0.49779078364372253},
                                    {"D", 0.50669991970062256},
                                    {"L", 0.4967217743396759},
                                    {"R", 0.56088292598724365}};

  // Discrete gridworld model
  StringOperatorDataMap namedInputs;

  namedInputs["model_id"] = (int64_t(98613017));
  namedInputs["snapshot_id"] = (int64_t(0));

  DecisionRequest request;
  request.context_features = context;
  request.actions.names = {"L", "R", "U", "D"};
  EXPECT_SYMBOLTABLE_NEAR(
      std::get<StringDoubleMap>(
          ActionValueScoring("test", PLAN_NAME, {}, service.get())
              .run(request, namedInputs)),
      expectedOutput);
#endif
}

} // namespace reagent
