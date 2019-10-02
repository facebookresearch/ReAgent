#pragma once

#include <stdexcept>

namespace reagent {
class DecisionServiceException : public std::runtime_error {
 public:
  explicit DecisionServiceException(const std::string& what)
      : std::runtime_error(what) {}

  explicit DecisionServiceException(const char* what)
      : std::runtime_error(what) {}
};
} // namespace reagent
