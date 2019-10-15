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

std::string generateUuid4() {
  static std::random_device rd;
  static std::uniform_int_distribution<uint64_t> dist(0, (uint64_t)(~0));

  uint64_t ab = dist(rd);
  uint64_t cd = dist(rd);

  ab = (ab & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;
  cd = (cd & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;

  std::stringstream ss;
  ss << std::hex << std::nouppercase << std::setfill('0');

  uint32_t a = (ab >> 32);
  uint32_t b = (ab & 0xFFFFFFFF);
  uint32_t c = (cd >> 32);
  uint32_t d = (cd & 0xFFFFFFFF);

  ss << std::setw(8) << (a) << '-';
  ss << std::setw(4) << (b >> 16) << '-';
  ss << std::setw(4) << (b & 0xFFFF) << '-';
  ss << std::setw(4) << (c >> 16) << '-';
  ss << std::setw(4) << (c & 0xFFFF);
  ss << std::setw(8) << d;

  return ss.str();
}
} // namespace reagent
