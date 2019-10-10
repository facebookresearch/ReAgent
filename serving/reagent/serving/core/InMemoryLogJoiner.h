#pragma once

#include <boost/filesystem.hpp>
#include <fstream>
#include "reagent/serving/core/LogJoiner.h"

namespace reagent {

class InMemoryLogJoiner : public LogJoiner {
 public:
  InMemoryLogJoiner(const std::string& log_path = "") {
    if (log_path.empty()) {
      LOG(INFO) << "No log file given. Logging will be in memory only";
      return;
    }
    auto log_dir = boost::filesystem::path(log_path).parent_path();
    if (!boost::filesystem::exists(log_dir)) {
      boost::filesystem::create_directory(log_dir.string());
      LOG(INFO) << "Log directory : " << log_dir
                << " does not exist. Created";
    }

    logStream_ = std::make_unique<std::ofstream>(
        log_path, std::ofstream::out | std::ofstream::app);
  }
  virtual ~InMemoryLogJoiner() override {
    if (logStream_ != nullptr) {
      logStream_->close();
    }
  }

  void logDecision(
      const DecisionRequest& request,
      const DecisionResponse& decisionResponse,
      const StringOperatorDataMap& operator_outputs) override;

  void logFeedback(Feedback feedback) override;

  virtual DecisionWithFeedback deserializeAndJoinDecisionAndFeedback(
      StringList decisionAndFeedback) override;

  std::unordered_map<std::string, DecisionWithFeedback> getLoggedData();

 protected:
  std::unique_ptr<std::ofstream> logStream_;
  std::unordered_map<std::string, DecisionWithFeedback> loggedData_;
  std::unordered_map<std::string, DecisionWithFeedback> unjoinedData_;
};

} // namespace reagent
