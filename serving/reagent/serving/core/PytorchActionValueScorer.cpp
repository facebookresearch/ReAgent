#include "reagent/serving/core/PytorchActionValueScorer.h"

#include "reagent/serving/core/Operator.h"

namespace reagent {

PytorchActionValueScorer::PytorchActionValueScorer() : ActionValueScorer() {}

StringDoubleMap PytorchActionValueScorer::predict(
    const DecisionRequest& request, int modelId, int snapshotId) {
  try {
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
    std::set<std::string> actionNameSet(actionNames.begin(), actionNames.end());

    StringDoubleMap retval;

    if (discreteActions) {
      int input_size = 1;
      for (auto it : request.context_features) {
        input_size = std::max(input_size, 1 + std::stoi(it.first));
      }
      auto input = torch::zeros({1, input_size});
      auto inputMask = torch::zeros({1, input_size});
      for (auto it : request.context_features) {
        VLOG(1) << "FEATURE SCORE: " << it.second;
        input[0][std::stoi(it.first)] = it.second;
        inputMask[0][std::stoi(it.first)] = 1.0;
      }
      // Create a vector of inputs.
      std::vector<torch::jit::IValue> inputs;
      auto stateWithPresence = c10::ivalue::Tuple::create({input, inputMask});
      inputs.push_back(stateWithPresence);
      auto result = model.forward(inputs);
      auto tupleResult = result.toTuple();
      auto outputActionNames = tupleResult->elements()[0];
      auto outputActionNameList = outputActionNames.toGenericList();
      auto actionScores = tupleResult->elements()[1];
      auto actionScoresTensor = actionScores.toTensor();
      for (int a = 0; a < outputActionNameList.size(); a++) {
        std::string scoredActionName =
            outputActionNameList.get(a).toStringRef();
        if (actionNameSet.find(scoredActionName) == actionNameSet.end()) {
          VLOG(1) << "Skipping action that wasn't possible";
          continue;
        }
        VLOG(1) << "SCORING: " << scoredActionName << " -> "
                << actionScoresTensor[0][a].item().to<double>();
        retval[scoredActionName] = actionScoresTensor[0][a].item().to<double>();
      }
    } else {
      LOG(FATAL) << "Not supported yet";
    }

    VLOG(1) << "SCORED " << retval.size() << " ITEMS";
    return retval;
  } catch (const c10::Error& e) {
    LOG(FATAL) << "TORCH ERROR: " << e.what();
  } catch (...) {
    LOG(FATAL) << "UNKNOWN ERROR";
  }
}

}  // namespace reagent
