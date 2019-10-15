#include "reagent/serving/core/DiskConfigProvider.h"
#include <folly/FileUtil.h>
#include <boost/filesystem.hpp>
#include "reagent/serving/core/DecisionService.h"

namespace reagent {

void DiskConfigProvider::initialize(DecisionService* decisionService) {
  decisionService_ = decisionService;
  LOG(INFO) << "READING CONFIGS FROM " << configDir_;
  for (auto& file :
       boost::filesystem::recursive_directory_iterator(configDir_)) {
    readConfig(file.path().string());
  }
}

void DiskConfigProvider::readConfig(const std::string& path) {
  auto planName =
      boost::filesystem::relative(boost::filesystem::path(path), configDir_)
          .string();
  DecisionConfig decisionConfig;

  std::string contents;

  if (!folly::readFile(path.c_str(), contents)) {
    // Read of config failed
    LOG(INFO) << "Reading config: " << path << " failed";
    return;
  }

  try {
    decisionConfig = nlohmann::json::parse(contents).get<DecisionConfig>();
  } catch (const std::exception& e) {
    // Parsing of JSON failed. Super sad panda :(
    LOG(ERROR) << "Config was not a valid thrift object: " << contents
               << std::endl
               << "Error: " << e.what();
    return;
  }
  // Now do stuff with the newly-populated `cfg` object

  LOG(INFO) << "GOT CONFIG " << planName << " AT " << path;

  try {
    decisionService_->createPlan(planName, decisionConfig);
    LOG(INFO) << "Registered decision config: " << planName;
  } catch (const std::runtime_error& er) {
    LOG(ERROR) << er.what();
  }
}
}  // namespace reagent
