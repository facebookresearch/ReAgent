#include "reagent/serving/operators/EpsilonGreedyRanker.h"
#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData EpsilonGreedyRanker::run(
    const DecisionRequest&,
    const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("values"));
  double epsilon = std::get<double>(namedInputs.at("epsilon"));
  OperatorData ret = runInternal(input, epsilon);
  return ret;
}

RankedActionList EpsilonGreedyRanker::runInternal(
    const StringDoubleMap& input,
    double epsilon) {
  std::vector<double> values;
  StringList names;
  for (auto& it : input) {
    values.emplace_back(it.second);
    names.emplace_back(it.first);
  }

  RankedActionList rankedList;
  std::vector<double> tmpValues(values);
  double rollingPropensity = 1.0;

  while (names.size() > 0) {
    double r = (double(std::rand()) / RAND_MAX);
    int idx;

    if (r < epsilon) {
      idx = std::rand() % names.size();
      rollingPropensity *= epsilon;
    } else {
      idx = std::distance(
          tmpValues.begin(), max_element(tmpValues.begin(), tmpValues.end()));
      rollingPropensity *= (1.0 - epsilon);
    }

    rankedList.push_back({names[idx], rollingPropensity});
    names.erase(names.begin() + idx);
    tmpValues.erase(tmpValues.begin() + idx);
  }
  return rankedList;
}

REGISTER_OPERATOR(EpsilonGreedyRanker, "EpsilonGreedyRanker");

} // namespace ml
