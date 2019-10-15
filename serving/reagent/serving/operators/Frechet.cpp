#include "reagent/serving/operators/Frechet.h"

#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData Frechet::run(const DecisionRequest&,
                          const StringOperatorDataMap& namedInputs) {
  const StringDoubleMap& input =
      std::get<StringDoubleMap>(namedInputs.at("values"));
  double rho = std::get<double>(namedInputs.at("rho"));
  double gamma = std::get<double>(namedInputs.at("gamma"));
  int seed;
  if (namedInputs.count("seed") > 0) {
    seed = int(std::get<int64_t>(namedInputs.at("seed")));
  } else {
    seed = std::chrono::system_clock::now().time_since_epoch().count();
  }
  OperatorData ret = run(input, rho, gamma, seed);
  return ret;
}

StringDoubleMap Frechet::run(const StringDoubleMap& input, double rho,
                             double gamma, int seed) {
  // Based on work from Leon & Badri outlined here:
  // https://fb.workplace.com/groups/121372781874504/permalink/159419624736486/

  if (rho <= 0 || gamma < 1) {
    LOG_AND_THROW(
        "Rho must be > 0 and gamma >= 1. Rho: " << rho << " Gamma: " << gamma);
  }

  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(1.0e-10, 1.0);

  StringDoubleMap retval;
  StringList names;
  for (auto& it : input) {
    double u = distribution(generator);
    VLOG(2) << "Sample from uniform distribution: " << u;
    retval[it.first] = it.second * (rho * pow((-log(u)), -1 / gamma));
  }
  return retval;
}

REGISTER_OPERATOR(Frechet, "Frechet");

}  // namespace ml
