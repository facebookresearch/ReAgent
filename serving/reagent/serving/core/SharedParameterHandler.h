#pragma once

#include "reagent/serving/core/Headers.h"

namespace reagent {
class SharedParameterInfo {
 public:
  explicit SharedParameterInfo(std::string name) : name_(name) {}

  time_t getLastFetchTime() {
    return lastFetchTime_;
  }

  StringDoubleMap getValues() {
    return values_;
  }

  void updateValues(StringDoubleMap values) {
    values_ = values;
    lastFetchTime_ = time(nullptr);
  }

 protected:
  std::string name_;
  time_t lastFetchTime_;
  StringDoubleMap values_;
};

class SharedParameterHandler {
 public:
  SharedParameterHandler();

  virtual ~SharedParameterHandler() = default;

  virtual StringDoubleMap getValues(const std::string& name);

  virtual bool acquireLockToModifyParameter(const std::string& name);

  // This doesn't guarantee that we acquired the lock, maybe there's a better
  // architecture?
  virtual void updateParameter(
      const std::string& name,
      const StringDoubleMap& values);

 protected:
  std::unordered_map<std::string, std::shared_ptr<SharedParameterInfo>>
      parameters_;

  inline std::string get_parameter_store_name(
      const std::string& parameter_name) {
    return std::string("Parameter_Store_") + parameter_name;
  }
};
} // namespace ml
