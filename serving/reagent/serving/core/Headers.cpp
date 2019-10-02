#include "reagent/serving/core/Headers.h"

namespace reagent {
StringDoubleMap operatorDataToPropensity(const OperatorData& value) {
  return std::visit(
      [](auto&& arg) -> StringDoubleMap {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
          return StringDoubleMap({{arg, 1.0}});
        } else if constexpr (std::is_same_v<T, StringList>) {
          StringDoubleMap sdm;
          for (int i = 0; i < arg.size(); i++) {
            sdm[arg[i]] = -i;
          }
          return sdm;
        } else if constexpr (std::is_same_v<T, StringIntMap>) {
          StringDoubleMap sdm;
          for (const auto& it : arg) {
            sdm[it.first] = double(it.second);
          }
          return sdm;
        } else if constexpr (std::is_same_v<T, StringDoubleMap>) {
          return arg;
        } else {
          LOG_AND_THROW("Invalid output operator");
        }
      },
      value);
}
} // namespace reagent
