#include "reagent/serving/operators/EpsilonGreedyRanker.h"
#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData EpsilonGreedyRanker::run(
    const DecisionRequest&,
    const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("values"));
  double epsilon = std::get<double>(namedInputs.at("epsilon"));
  int seed;
  if (namedInputs.count("seed") > 0) {
    seed = int(std::get<int64_t>(namedInputs.at("seed")));
  } else {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  }
  OperatorData ret = run(input, epsilon, seed);
  return ret;
}

StringList EpsilonGreedyRanker::run(
    const StringDoubleMap& input,
    double epsilon,
    int seed) {
  std::vector<double> values;
  StringList names;
  for (auto& it : input) {
    values.emplace_back(it.second);
    names.emplace_back(it.first);
  }

  StringList rankedList;
  std::vector<double> tmpValues(values);
  std::srand(seed);

  while (names.size() > 0) {
    double r = (double(std::rand()) / RAND_MAX);
    int idx;

    if (r < epsilon) {
      idx = std::rand() % names.size();
    } else {
      idx = std::distance(
          tmpValues.begin(), max_element(tmpValues.begin(), tmpValues.end()));
    }

    rankedList.emplace_back(names[idx]);
    names.erase(names.begin() + idx);
    tmpValues.erase(tmpValues.begin() + idx);
  }
  return rankedList;
}

REGISTER_OPERATOR(EpsilonGreedyRanker, "EpsilonGreedyRanker");

} // namespace ml
