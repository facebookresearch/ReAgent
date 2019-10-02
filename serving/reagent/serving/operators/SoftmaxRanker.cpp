#include "reagent/serving/operators/SoftmaxRanker.h"
#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData SoftmaxRanker::run(const DecisionRequest&,
                                const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("values"));
  double temperature = std::get<double>(namedInputs.at("temperature"));
  int seed;
  if (namedInputs.count("seed") > 0) {
    seed = int(std::get<int64_t>(namedInputs.at("seed")));
  } else {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  }
  OperatorData ret = run(input, temperature, seed);
  return ret;
}

StringList SoftmaxRanker::run(const StringDoubleMap& input,
                                            double temperature, int seed) {
  std::vector<double> values;
  StringList names;
  for (auto& it : input) {
    values.emplace_back(it.second);
    names.emplace_back(it.first);
  }

  StringList rankedList;
  std::vector<double> tmpValues;

  while (names.size() > 0) {
    tmpValues = values;
    // Calculate tempered softmax
    double maxV = *std::max_element(tmpValues.begin(), tmpValues.end());
    for (int i = 0; i < int(tmpValues.size()); i++) {
      tmpValues[i] = exp((tmpValues[i] - maxV) / temperature);
    }
    double sumV = std::accumulate(tmpValues.begin(), tmpValues.end(), 0.0);
    for (int i = 0; i < int(tmpValues.size()); i++) {
      tmpValues[i] /= sumV;
    }

    // Sample from weighted distribution
    std::discrete_distribution<int> dist(std::begin(tmpValues),
                                         std::end(tmpValues));
    std::mt19937 generator;
    generator.seed(seed);
    int sample = dist(generator);

    // Select and remove sampled item from list to rank
    rankedList.emplace_back(names[sample]);
    names.erase(names.begin() + sample);
    values.erase(values.begin() + sample);
  }
  return rankedList;
}

REGISTER_OPERATOR(SoftmaxRanker, "SoftmaxRanker");

}  // namespace ml
