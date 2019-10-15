#pragma once

#include "reagent/serving/core/Headers.h"

namespace reagent {

class ActionValueScorer {
 public:
  virtual ~ActionValueScorer() = default;

  virtual StringDoubleMap
  predict(const DecisionRequest& request, int model, int snapshot) = 0;
};

} // namespace reagent
