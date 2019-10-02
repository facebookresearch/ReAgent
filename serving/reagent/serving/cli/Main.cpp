#include "reagent/serving/core/Headers.h"

#include <folly/init/Init.h>

#include "reagent/serving/core/ConfigProvider.h"
#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/InMemoryLogJoiner.h"
#include "reagent/serving/core/LocalRealTimeCounter.h"
#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {
int Main(int argc, char** argv) {
  folly::init(&argc, &argv);

  auto service = std::make_shared<DecisionService>(
      std::make_shared<ConfigProvider>(),
      std::shared_ptr<ActionValueScorer>(),
      std::make_shared<InMemoryLogJoiner>(),
      std::make_shared<LocalRealTimeCounter>(),
      std::make_shared<SharedParameterHandler>());

  auto request = DecisionRequest();
  request.plan_name = "reagent/servings/test/simple/softmax";
  request.actions.names = {"action1", "action2"};
  request.request_id = "asdf";
  const auto& response = service->process(request);
  for (auto action : response.actions) {
    LOG(INFO) << "ACTION " << action.name << " HAS PROPENSITY "
              << action.propensity;
  }

  while (true) {
    /* sleep override */ sleep(1);
  }
}
} // namespace reagent

int main(int argc, char** argv) {
  return reagent::Main(argc, argv);
}
