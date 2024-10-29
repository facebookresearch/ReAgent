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

inline void EXPECT_RANKEDACTIONLIST_NEAR(
    const RankedActionList& st1,
    const RankedActionList& st2) {
  EXPECT_EQ(st1.size(), st2.size());
  for (int a = 0; a < int(st1.size()); a++) {
    EXPECT_EQ(st1[a].name, st2[a].name);
    EXPECT_NEAR(st1[a].propensity, st2[a].propensity, 1e-3);
  }
}

std::shared_ptr<DecisionService> makeTestDecisionService();

} // namespace reagent
