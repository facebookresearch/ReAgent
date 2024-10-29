#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/OperatorFactory.h"
#include "reagent/serving/core/OperatorRunner.h"
#include "reagent/serving/operators/InputFromRequest.h"

namespace reagent {

TEST(InputFromRequestTests, Simple) {
  auto service = makeTestDecisionService();
  const auto PLAN_NAME = std::string("/");
  const auto INPUT_DATA = 100;

  DecisionRequest request;
  OperatorData input = int64_t(INPUT_DATA);
  request.input = input;

  EXPECT_EQ(
      std::get<int64_t>(
          InputFromRequest("input_from_request", PLAN_NAME, {}, service.get())
              .run(request, {})),
      INPUT_DATA);
}

} // namespace reagent
