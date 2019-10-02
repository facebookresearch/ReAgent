#include "reagent/serving/operators/Expression.h"

#include "reagent/serving/core/OperatorFactory.h"

#include <exprtk.hpp>

namespace reagent {
OperatorData Expression::run(
    const DecisionRequest&,
    const StringOperatorDataMap& namedInputs) {
  const std::string& equation =
      std::get<std::string>(namedInputs.at("equation"));
  std::unordered_map<std::string, StringDoubleMap> equations;
  for (const auto& it : namedInputs) {
    const auto& inputName = it.first;
    if (inputName == std::string("equation")) {
      continue;
    }
    const StringDoubleMap& inputEquationValues =
        std::get<StringDoubleMap>(it.second);
    for (const auto& it2 : inputEquationValues) {
      const auto& equationName = it2.first;
      double inputValue = it2.second;
      if (equations.find(equationName) == equations.end()) {
        equations[equationName] = StringDoubleMap();
      }
      equations.at(equationName)[inputName] = inputValue;
    }
  }
  StringDoubleMap output;
  for (const auto& it : equations) {
    output[it.first] = runInternal(equation, it.second);
  }
  OperatorData ret = output;
  return ret;
}

double Expression::runInternal(
    const std::string& equation,
    const StringDoubleMap& symbolTableMap) {
  exprtk::symbol_table<double> symbolTable;
  for (const auto& it : symbolTableMap) {
    symbolTable.add_constant(it.first, it.second);
  }
  exprtk::expression<double> expression;
  expression.register_symbol_table(symbolTable);

  exprtk::parser<double> parser;
  parser.compile(equation, expression);

  double value = expression.value();
  return value;
}

REGISTER_OPERATOR(Expression, "Expression");

} // namespace reagent
