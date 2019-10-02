#include "reagent/serving/operators/ActionValueScoring.h"

#include "reagent/serving/core/ActionValueScorer.h"
#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData ActionValueScoring::run(const DecisionRequest& request,
                                     const StringOperatorDataMap& namedInputs) {
  int modelId = int(std::get<int64_t>(namedInputs.at("model_id")));
  int snapshotId = int(std::get<int64_t>(namedInputs.at("snapshot_id")));
  OperatorData ret;
  ret = run(modelId, snapshotId, request);
  return ret;
}

StringDoubleMap ActionValueScoring::run(int modelId, int snapshotId,
                                        const DecisionRequest& request) {
  return actionValueScorer_->predict(request, modelId, snapshotId);
}

REGISTER_OPERATOR(ActionValueScoring, "ActionValueScoring");

}  // namespace ml
