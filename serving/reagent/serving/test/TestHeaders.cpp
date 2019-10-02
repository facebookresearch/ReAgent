#include "reagent/serving/test/TestHeaders.h"

#include "reagent/serving/core/ConfigProvider.h"
#include "reagent/serving/core/InMemoryLogJoiner.h"
#include "reagent/serving/core/LocalRealTimeCounter.h"
#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {
std::shared_ptr<DecisionService> makeTestDecisionService() {
  return std::make_shared<DecisionService>(
      std::make_shared<ConfigProvider>(), std::shared_ptr<ActionValueScorer>(),
      std::make_shared<InMemoryLogJoiner>(),
      std::make_shared<LocalRealTimeCounter>(),
      std::make_shared<SharedParameterHandler>());
}

}  // namespace ml
