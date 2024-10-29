#pragma once

#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/ConfigProvider.h"

namespace reagent {
class DiskConfigProvider : public ConfigProvider {
 public:
  explicit DiskConfigProvider(std::string config_dir) {
    configDir_ = config_dir;
  }

 protected:
  std::string configDir_;

  void initialize(DecisionService* decisionService) override;
  void readConfig(const std::string& path);
};
} // namespace reagent
