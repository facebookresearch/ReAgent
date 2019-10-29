#include "reagent/serving/core/DecisionPlan.h"

#include <exprtk.hpp>

namespace reagent {
DecisionPlan::DecisionPlan(
    const DecisionConfig& config,
    const std::vector<std::shared_ptr<Operator>>& operators,
    const StringOperatorDataMap& constants)
    : config_(config), operators_(operators), constants_(constants) {
}

double DecisionPlan::evaluateRewardFunction(const StringDoubleMap& metrics) {
  exprtk::symbol_table<double> symbolTable;
  for (const auto& it : metrics) {
    symbolTable.add_constant(it.first, it.second);
  }
  exprtk::expression<double> expression;
  expression.register_symbol_table(symbolTable);

  exprtk::parser<double> parser;
  parser.compile(config_.reward_function, expression);

  double value = expression.value();
  return value;
}

}  // namespace reagent
