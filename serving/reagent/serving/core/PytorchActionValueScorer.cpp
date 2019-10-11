#include "reagent/serving/core/PytorchActionValueScorer.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {

PytorchActionValueScorer::PytorchActionValueScorer() {}

StringDoubleMap PytorchActionValueScorer::predict(
    const DecisionRequest& request, int modelId, int snapshotId) {
  std::string path =
      "/tmp/" + std::to_string(modelId) + "/" + std::to_string(snapshotId);

  if (models_.find(path) == models_.end()) {
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      torch::jit::script::Module module = torch::jit::load(path);
      models_[path] = std::move(module);
    } catch (const c10::Error& e) {
      LOG(ERROR) << "Error loading the model: " << e.what();
      return StringDoubleMap();
    }
  }
  auto model = models_.find(path)->second;

  bool discreteActions = !request.actions.names.empty();

  StringList actionNames = Operator::getActionNamesFromRequest(request);

  StringDoubleMap retval;

  if (discreteActions) {
    int64_t input_size = 0;
    for (auto it : request.context_features) {
      input_size = std::max(input_size, int64_t(std::stoll(it.first)));
    }
    auto input = torch::zeros({input_size});
    for (auto it : request.context_features) {
      input[std::stoll(it.first)] = it.second;
    }
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    auto result = model.forward(inputs).toTensor();
    for (int a = 0; a < actionNames.size(); a++) {
      retval[actionNames[a]] = result[a].item().to<double>();
    }
  } else {
    LOG(FATAL) << "Not supported yet";
  }

  return retval;
}

}  // namespace reagent
