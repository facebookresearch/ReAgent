#include "reagent/serving/core/DecisionPlan.h"

#include <exprtk.hpp>

namespace reagent {
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

} // namespace reagent
