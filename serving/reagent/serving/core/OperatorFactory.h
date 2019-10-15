#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {
class DecisionService;

#define REGISTER_OPERATOR(CLASS, OP_NAME)                        \
  std::shared_ptr<Operator> create##CLASS##Fn(                   \
      const std::string& name,                                   \
      const std::string& planName,                               \
      const StringStringMap& inputDepMap,                        \
      const DecisionService* const decisionService) {            \
    return std::make_shared<CLASS>(                              \
        name, planName, inputDepMap, decisionService);           \
  }                                                              \
                                                                 \
  bool register##CLASS##Fn() {                                   \
    OperatorFactory::getInstance()->registerOperatorConstructor( \
        OP_NAME, create##CLASS##Fn);                             \
    return true;                                                 \
  }                                                              \
  static bool register##CLASS##FnResult = register##CLASS##Fn();

class OperatorFactory {
 public:
  typedef std::function<std::shared_ptr<Operator>(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)>
      OperatorConstructor;

  static std::shared_ptr<OperatorFactory> getInstance();

  void registerOperatorConstructor(
      const std::string& opName,
      OperatorConstructor constructor);

  std::shared_ptr<Operator> createOp(
      const std::string& planName,
      const OperatorDefinition& definition,
      const DecisionService* const decisionService);

  std::vector<std::shared_ptr<Operator>> createFromConfig(
      const std::string& planName,
      const DecisionConfig& config,
      const DecisionService* const decisionService);

 protected:
  std::unordered_map<std::string, OperatorConstructor> constructors_;
};

} // namespace reagent
