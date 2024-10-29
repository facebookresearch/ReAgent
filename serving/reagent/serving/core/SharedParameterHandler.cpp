#include "reagent/serving/core/SharedParameterHandler.h"

namespace reagent {
SharedParameterHandler::SharedParameterHandler() {}

StringDoubleMap SharedParameterHandler::getValues(const std::string& name) {
  if (parameters_.find(name) == parameters_.end()) {
    // Add the parameter
    parameters_[name] = std::make_shared<SharedParameterInfo>(name);
  }

  auto parameter = parameters_.find(name)->second;
  return parameter->getValues();
}

bool SharedParameterHandler::acquireLockToModifyParameter(const std::string&) {
  return true;
}

void SharedParameterHandler::updateParameter(
    const std::string& name,
    const StringDoubleMap& values) {
  auto it = parameters_.find(name);
  if (it == parameters_.end()) {
    LOG_AND_THROW("Tried to update a parameter that doesn't exist: " << name);
  }
  it->second->updateValues(values);
}

} // namespace reagent
