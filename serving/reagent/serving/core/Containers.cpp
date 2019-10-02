#include "reagent/serving/core/Headers.h"

#include "reagent/serving/core/Containers.h"

namespace reagent {
ConstantValue json_to_constant_value(const json& j) {
  if (j.count("string_value")) {
    return j.at("string_value").get<std::string>();
  }
  if (j.count("int_value")) {
    return j.at("int_value").get<int64_t>();
  }
  if (j.count("double_value")) {
    return j.at("double_value").get<double>();
  }
  if (j.count("list_string_value")) {
    return j.at("list_string_value").get<StringList>();
  }
  if (j.count("list_int_value")) {
    return j.at("list_int_value").get<std::vector<int64_t>>();
  }
  if (j.count("list_double_value")) {
    return j.at("list_double_value").get<std::vector<double>>();
  }
  if (j.count("map_string_value")) {
    return j.at("map_string_value").get<StringStringMap>();
  }
  if (j.count("map_int_value")) {
    return j.at("map_int_value").get<StringIntMap>();
  }
  if (j.count("map_double_value")) {
    return j.at("map_double_value").get<StringDoubleMap>();
  }
  if (j.count("map_map_double_value")) {
    return j.at("map_map_double_value")
        .get<std::unordered_map<std::string, StringDoubleMap>>();
  }
  std::string what = "Invalid constant: " + j.dump();
  throw std::runtime_error(what.c_str());
}

void constant_value_to_json(const ConstantValue& value, json& j) {
  j = std::visit(
      [j](auto&& arg) -> json {
        using T = std::decay_t<decltype(arg)>;
        json j;
        if (std::is_same_v<T, std::string>) {
          j["string_value"] = arg;
        } else if (std::is_same_v<T, int64_t>) {
          j["int_value"] = arg;
        } else if (std::is_same_v<T, double>) {
          j["double_value"] = arg;
        } else if (std::is_same_v<T, StringList>) {
          j["list_string_value"] = arg;
        } else if (std::is_same_v<T, std::vector<int64_t>>) {
          j["list_int_value"] = arg;
        } else if (std::is_same_v<T, std::vector<double>>) {
          j["list_double_value"] = arg;
        } else if (std::is_same_v<T, StringStringMap>) {
          j["map_string_value"] = arg;
        } else if (std::is_same_v<T, StringIntMap>) {
          j["map_int_value"] = arg;
        } else if (std::is_same_v<T, StringDoubleMap>) {
          j["map_double_value"] = arg;
        } else if (std::is_same_v<
                       T,
                       std::unordered_map<std::string, StringDoubleMap>>) {
          j["map_map_double_value"] = arg;
        } else {
          LOG_AND_THROW("Invalid output operator");
        }
        return j;
      },
      value);
}

} // namespace reagent
