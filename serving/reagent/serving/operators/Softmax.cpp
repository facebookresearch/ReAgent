#include "reagent/serving/operators/Softmax.h"

#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData Softmax::run(const DecisionRequest&,
                          const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("values"));
  double temperature = std::get<double>(namedInputs.at("temperature"));
  OperatorData ret = run(input, temperature);
  return ret;
}

StringDoubleMap Softmax::run(const StringDoubleMap& input, double temperature) {
  Eigen::ArrayXd v(input.size());
  StringList names;
  for (auto& it : input) {
    v[names.size()] = it.second;
    names.emplace_back(it.first);
  }
  v -= v.maxCoeff();
  v /= temperature;
  v = v.exp();
  v /= v.sum();
  StringDoubleMap retval;
  for (int a = 0; a < int(names.size()); a++) {
    retval[names[a]] = v[a];
  }
  return retval;
}

REGISTER_OPERATOR(Softmax, "Softmax");

}  // namespace ml
