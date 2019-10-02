#include "reagent/serving/core/OperatorRunner.h"

namespace {
auto DAG_TIMEOUT = std::chrono::seconds(30);
}

namespace reagent {
StringOperatorDataMap OperatorRunner::run(
    const std::vector<std::shared_ptr<Operator>>& operators,
    const StringOperatorDataMap& constants,
    const DecisionRequest& request,
    const OperatorData& extraInput) {
  // Map of futures for finished ops
  std::unordered_map<
      std::string,
      std::shared_ptr<folly::SharedPromise<folly::Unit>>>
      operatorPromiseMap;

  StringOperatorDataMap finishedOperators;
  std::mutex finishedOperatorMutex;

  // Add special constant "input" equal to the extra input
  auto dummyPromise = std::make_shared<folly::SharedPromise<folly::Unit>>();
  dummyPromise->setWith([]() {});
  operatorPromiseMap["input"] = dummyPromise;
  finishedOperators["input"] = extraInput;

  // Add all constants to finished operators
  for (const auto& it : constants) {
    operatorPromiseMap[it.first] = dummyPromise;
    finishedOperators[it.first] = it.second;
  }

  // Create shared promises for all operators
  for (const auto& op : operators) {
    auto opPromise = std::make_shared<folly::SharedPromise<folly::Unit>>();
    operatorPromiseMap[op->getName()] = opPromise;
  }

  // List of futures for finished operators
  std::vector<folly::Future<folly::Unit>> opFinishedFutures;

  // Collect semifutures for each operator's dependencies and collect them, then
  // set the operator's promise
  for (auto op : operators) {
    // Create a list of futures for each dependency
    std::vector<folly::SemiFuture<folly::Unit>> depFutures;
    for (const auto& depName : op->getDeps()) {
      depFutures.push_back(operatorPromiseMap[depName]->getSemiFuture());
    }

    // Get the promise for the op we want to run
    auto operatorPromise = operatorPromiseMap.at(op->getName());

    // Add the following to our finished op list
    opFinishedFutures.push_back(
        // Collect the dependencies
        folly::collect(depFutures)
            // Assign an executor to run our futures
            .via(executor_.get())
            // All the deps are finished, it's go time!
            .thenValue([op,
                        operatorPromise,
                        &request,
                        &finishedOperators,
                        &finishedOperatorMutex](
                           std::vector<folly::Unit> /* unused */) {
              // Set the promise which will mark this op as finished for its
              // deps
              operatorPromise->setWith(
                  [op, &request, &finishedOperators, &finishedOperatorMutex]() {
                    // Resolve input symbols
                    StringOperatorDataMap namedInputs;
                    {
                      std::lock_guard<std::mutex> lock(finishedOperatorMutex);
                      for (const auto& inputDepEntry : op->getInputDepMap()) {
                        const auto& inputName = inputDepEntry.first;
                        const auto& depOperatorName = inputDepEntry.second;
                        auto it = finishedOperators.find(depOperatorName);
                        if (it == finishedOperators.end()) {
                          LOG(ERROR)
                              << "Could not find data for finished operator";
                        }
                        namedInputs[inputName] = it->second;
                      }
                    }

                    // Run the op
                    OperatorData outputData = op->run(request, namedInputs);

                    // Set output data
                    {
                      std::lock_guard<std::mutex> lock(finishedOperatorMutex);
                      finishedOperators[op->getName()] = outputData;
                    }
                  });
            }));
  }

  // Collect all finished ops
  auto allOpsFinishedFuture = folly::collect(opFinishedFutures);

  // Wait on ops to finish
  allOpsFinishedFuture.wait(DAG_TIMEOUT);
  if (!allOpsFinishedFuture.isReady()) {
    LOG_AND_THROW("DAG Timeout");
  }

  if (finishedOperators.size() != operators.size() + 1 + constants.size()) {
    LOG_AND_THROW("DAG Incomplete");
  }

  return finishedOperators;
}
} // namespace ml
