#include "reagent/serving/core/DiskConfigProvider.h"
#include <boost/filesystem.hpp>  // We have to use boost until OSX 10.15 :-(
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

inline std::string ReadFile(const std::string& fileName) {
  std::ifstream ifs(fileName.c_str(),
                    std::ios::in | std::ios::binary | std::ios::ate);

  std::ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  std::vector<char> bytes(fileSize);
  ifs.read(bytes.data(), fileSize);

  return std::string(bytes.data(), fileSize);
}

void DiskConfigProvider::readConfig(const std::string& path) {
  auto planName =
      boost::filesystem::relative(boost::filesystem::path(path), configDir_)
          .string();
  DecisionConfig decisionConfig;

  std::string contents = ReadFile(path);

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
