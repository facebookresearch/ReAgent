#pragma once

#include <gtest/gtest.h>

#include "reagent/serving/core/DecisionService.h"
#include "reagent/serving/core/Headers.h"
#include "reagent/serving/core/Operator.h"

namespace reagent {

inline void EXPECT_SYMBOLTABLE_NEAR(
    const StringDoubleMap& st1,
    const StringDoubleMap& st2) {
  std::set<std::string> keys1;
  std::transform(
      st1.begin(), st1.end(), std::inserter(keys1, keys1.end()), [](auto pair) {
        return pair.first;
      });

  std::set<std::string> keys2;
  std::transform(
      st2.begin(), st2.end(), std::inserter(keys2, keys2.end()), [](auto pair) {
        return pair.first;
      });

  EXPECT_EQ(keys1, keys2);

  for (auto& it : st1) {
    EXPECT_NEAR(it.second, st2.find(it.first)->second, 1e-3);
  }
}

std::shared_ptr<DecisionService> makeTestDecisionService();

} // namespace ml
