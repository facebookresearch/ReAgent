#include "reagent/serving/core/Headers.h"

#include <glog/logging.h>

#include "reagent/serving/cli/Server.h"
#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/DiskConfigProvider.h"
#include "reagent/serving/core/InMemoryLogJoiner.h"
#include "reagent/serving/core/LocalRealTimeCounter.h"
#include "reagent/serving/core/PytorchActionValueScorer.h"
#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {
int Main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  auto service = std::make_shared<DecisionService>(
      std::make_shared<DiskConfigProvider>("serving/examples/ecommerce/plans"),
      std::make_shared<PytorchActionValueScorer>(),
      std::make_shared<InMemoryLogJoiner>("/tmp/rasp_logging/log.txt"),
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
