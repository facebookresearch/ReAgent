#pragma once

#include <cassert>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <glog/logging.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <taskflow/taskflow.hpp>
#if __has_include("filesystem")
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#endif

#include "reagent/serving/core/Containers.h"
#include "reagent/serving/core/DecisionServiceException.h"

#define LOG_AND_THROW(MSG_STREAM)                               \
  {                                                             \
    std::ostringstream errorStream;                             \
    errorStream << MSG_STREAM;                                  \
    LOG(ERROR) << errorStream.str();                            \
    throw reagent::DecisionServiceException(errorStream.str()); \
  }

namespace reagent {
std::string generateUuid4();

}  // namespace reagent
