#pragma once

#include "reagent/serving/core/ActionValueScorer.h"

#include <torch/script.h>

namespace reagent {

class PytorchActionValueScorer : public ActionValueScorer {
 public:
  PytorchActionValueScorer();

  virtual ~PytorchActionValueScorer() override = default;

  StringDoubleMap
  predict(const DecisionRequest& request, int modelId, int snapshotId) override;

 protected:
  std::unordered_map<std::string, torch::jit::script::Module> models_;
};

} // namespace reagent
