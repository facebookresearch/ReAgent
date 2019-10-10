#pragma once

#include <cassert>
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

#include <folly/Random.h>
#include <folly/Singleton.h>
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#include <folly/futures/Future.h>
#include <folly/futures/SharedPromise.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

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
StringDoubleMap operatorDataToPropensity(const OperatorData& value);

std::string generateUuid4();

}  // namespace reagent
