#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
std::shared_ptr<OperatorFactory> OperatorFactory::getInstance() {
  static std::shared_ptr<OperatorFactory> operatorFactorySingleton =
      std::make_shared<OperatorFactory>();
  return operatorFactorySingleton;
}

void OperatorFactory::registerOperatorConstructor(
    const std::string& opName, OperatorConstructor constructor) {
  constructors_[opName] = constructor;
}

std::shared_ptr<Operator> OperatorFactory::createOp(
    const std::string& planName, const OperatorDefinition& definition,
    const DecisionService* const decisionService) {
  auto name = definition.name;
  auto inputDepMap = definition.input_dep_map;
  auto opName = definition.op_name;
  auto it = constructors_.find(opName);
  if (it == constructors_.end()) {
    LOG_AND_THROW("Tried to create an op type that does not exist: " << opName);
  }
  return it->second(name, planName, inputDepMap, decisionService);
}

std::vector<std::shared_ptr<Operator>> OperatorFactory::createFromConfig(
    const std::string& planName, const DecisionConfig& config,
    const DecisionService* const decisionService) {
  std::vector<std::shared_ptr<Operator>> operators;
  for (auto it : config.operators) {
    operators.emplace_back(
        OperatorFactory::createOp(planName, it, decisionService));
  }
  return operators;
}

}  // namespace ml
