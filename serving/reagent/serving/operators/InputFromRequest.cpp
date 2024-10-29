#include "reagent/serving/operators/InputFromRequest.h"

#include "reagent/serving/core/OperatorFactory.h"

namespace reagent {
OperatorData InputFromRequest::run(
    const DecisionRequest& request,
    const StringOperatorDataMap&) {
  return request.input;
}

REGISTER_OPERATOR(InputFromRequest, "InputFromRequest");

} // namespace reagent
