#include "reagent/serving/core/Headers.h"

#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/logging/xlog.h>

#include "reagent/serving/cli/Server.h"
#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/DiskConfigProvider.h"
#include "reagent/serving/core/InMemoryLogJoiner.h"
#include "reagent/serving/core/LocalRealTimeCounter.h"
#include "reagent/serving/core/PytorchActionValueScorer.h"
#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {
int Main(int argc, char** argv) {
  folly::init(&argc, &argv);

  auto service = std::make_shared<DecisionService>(
      std::make_shared<DiskConfigProvider>(
          "serving/examples/ecommerce/plans"),
      std::make_shared<PytorchActionValueScorer>(),
      std::make_shared<InMemoryLogJoiner>("/tmp/dsp_logging/log.txt"),
      std::make_shared<LocalRealTimeCounter>(),
      std::make_shared<SharedParameterHandler>());

  Server server(service, 3000);
  server.start();

  while (true) {
    sleep(1);
  }
}
}  // namespace reagent

int main(int argc, char** argv) { return reagent::Main(argc, argv); }
