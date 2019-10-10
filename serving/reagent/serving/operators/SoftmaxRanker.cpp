#include "reagent/serving/operators/SoftmaxRanker.h"
#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
SoftmaxRanker::SoftmaxRanker(const std::string& name,
                             const std::string& planName,
                             const StringStringMap& inputDepMap,
                             const DecisionService* const decisionService)
    : Operator(name, planName, inputDepMap, decisionService) {
  int seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator_.seed(seed);
}

OperatorData SoftmaxRanker::run(const DecisionRequest&,
                                const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("values"));
  double temperature = std::get<double>(namedInputs.at("temperature"));
  OperatorData ret = run(input, temperature);
  return ret;
}

RankedActionList SoftmaxRanker::run(const StringDoubleMap& input,
                                    double temperature) {
  std::vector<double> values;
  StringList names;
  for (auto& it : input) {
    values.emplace_back(it.second);
    names.emplace_back(it.first);
  }

  RankedActionList rankedList;
  std::vector<double> tmpValues;

  double rollingPropensity = 1.0;
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
    int sample = dist(generator_);
    rollingPropensity *= tmpValues[sample];

    // Select and remove sampled item from list to rank
    rankedList.push_back(ActionDetails({names[sample], rollingPropensity}));
    names.erase(names.begin() + sample);
    values.erase(values.begin() + sample);
  }
  return rankedList;
}

REGISTER_OPERATOR(SoftmaxRanker, "SoftmaxRanker");

}  // namespace reagent
