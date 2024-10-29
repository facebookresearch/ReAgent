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
  StringOperatorDataMap finishedOperators;
  std::mutex finishedOperatorMutex;
  tf::Taskflow taskflow;

  std::unordered_map<std::string, tf::Task> operatorTaskMap;

  // Add special constant "input" equal to the extra input
  finishedOperators["input"] = extraInput;
  operatorTaskMap["input"] = taskflow.emplace([]() {});

  // Add all constants to finished operators
  for (const auto& it : constants) {
    finishedOperators[it.first] = it.second;
    operatorTaskMap[it.first] = taskflow.emplace([]() {});
  }

  // Create tasks for all operators
  for (const auto& op : operators) {
    operatorTaskMap[op->getName()] = taskflow.emplace(
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
                LOG(ERROR) << "Could not find data for finished operator";
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
  }

  // Set dependencies
  for (const auto& op : operators) {
    auto& opTask = operatorTaskMap.at(op->getName());
    for (const auto& depName : op->getDeps()) {
      if (operatorTaskMap.find(depName) == operatorTaskMap.end()) {
        LOG_AND_THROW("Invalid Operator dep: " << depName);
      }
      operatorTaskMap.at(depName).precede(opTask);
      // depFutures.push_back(operatorPromiseMap.at(depName)->getSemiFuture());
    }
  }

  auto runStatus = taskExecutor_.run(taskflow).wait_for(DAG_TIMEOUT);
  if (runStatus == std::future_status::timeout) {
    LOG_AND_THROW("DAG Timeout");
  }
  if (runStatus != std::future_status::ready) {
    LOG_AND_THROW("Unknown error in DAG");
  }

  if (finishedOperators.size() != operators.size() + 1 + constants.size()) {
    LOG_AND_THROW("DAG Incomplete");
  }

  return finishedOperators;
}
} // namespace reagent
