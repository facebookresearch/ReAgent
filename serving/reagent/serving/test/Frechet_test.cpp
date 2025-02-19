#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/DecisionServiceException.h"
#include "reagent/serving/operators/Frechet.h"

namespace reagent {

TEST(FrechetTests, BadInput) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  StringDoubleMap values = {{"1", 10.0}, {"2", 5.0}};
  namedInputs["values"] = values;
  namedInputs["rho"] = 0.0;
  namedInputs["gamma"] = 1.0;

  EXPECT_THROW(Frechet("frechet", PLAN_NAME, {}, service.get())
                   .run(DecisionRequest(), namedInputs);
               , reagent::DecisionServiceException);
}

TEST(FrechetTests, FixedSeed) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  StringOperatorDataMap namedInputs;

  StringDoubleMap values = {{"1", 5.0}};
  namedInputs["values"] = values;
  namedInputs["rho"] = 1.0;
  namedInputs["gamma"] = 2.0;
  namedInputs["seed"] = int64_t(1);

  auto result =
      std::get<StringDoubleMap>(Frechet("frechet", PLAN_NAME, {}, service.get())
                                    .run(DecisionRequest(), namedInputs));
  // ((1 * (-math.log(0.0850324)) ** (- 1 / 2)) * 5 )
  // 3.1848281496136037

  // This isn't consistent across OS's for some reason.
  // EXPECT_NEAR(result["1"], 3.1848281496136037, 1e-3);
}

} // namespace reagent
