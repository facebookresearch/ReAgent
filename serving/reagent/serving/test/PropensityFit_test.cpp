#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/LocalRealTimeCounter.h"
#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/core/SharedParameterHandler.h"
#include "reagent/serving/operators/PropensityFit.h"

namespace reagent {

TEST(PropensityFit, Simple) {
  const auto PLAN_NAME = std::string("/");
  auto service = makeTestDecisionService();

  auto localRealTimeCounter = std::dynamic_pointer_cast<LocalRealTimeCounter>(
      service->getRealTimeCounter());

  // Shrink window size to make the test more responsive
  localRealTimeCounter->setWindowSize(1);

  StringOperatorDataMap namedInputs;

  StringDoubleMap input = {{"Action1", 0.2}};
  namedInputs["input"] = (input);

  StringDoubleMap targets = {{"Action1", 0.1}};
  namedInputs["targets"] = (targets);

  PropensityFit propensityFit("fit", PLAN_NAME, {}, service.get());

  StringDoubleMap result;

  DecisionRequest request;
  request.request_id = "asdf";
  request.plan_name = PLAN_NAME;
  request.actions.names = {"Action1"};
  for (int a = 0; a < 10000; a++) {
    result = std::get<StringDoubleMap>(propensityFit.run(request, namedInputs));

    // Generate empty feedback with the past Output
    Feedback feedback;
    OperatorData pastOutput;
    pastOutput = result;
    propensityFit.giveFeedback(feedback, namedInputs, pastOutput);
  }
  EXPECT_NEAR(
      propensityFit.getShift("Action1"),
      targets["Action1"] - input["Action1"],
      0.5);
}

} // namespace reagent
