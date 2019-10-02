#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {
class PropensityFit : public Operator {
 public:
  PropensityFit(
      const std::string& name,
      const std::string& planName,
      const StringStringMap& inputDepMap,
      const DecisionService* const decisionService)
      : Operator(name, planName, inputDepMap, decisionService) {}

  virtual ~PropensityFit() override = default;

  virtual OperatorData run(
      const DecisionRequest& request,
      const StringOperatorDataMap& namedInputs) override;

  virtual StringDoubleMap run(const StringDoubleMap& input);

  void giveFeedback(
      const Feedback& feedback,
      const StringOperatorDataMap& pastInputs,
      const OperatorData& pastOuptut) override;

  void giveFeedbackInternal(
      const Feedback& feedback,
      const StringOperatorDataMap& pastInputs,
      const StringDoubleMap& pastOuptut,
      const StringDoubleMap& targets);

  double getShift(const std::string& actionName);

 protected:
  inline std::string getParameterName(const std::string& configeratorPath) {
    return configeratorPath + std::string("/") + name_;
  }
};
} // namespace ml
