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
  if (j.count("ranked_action_list")) {
    return j.at("ranked_action_list").get<RankedActionList>();
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
        json result;
        if (std::is_same_v<T, std::string>) {
          result["string_value"] = arg;
        } else if (std::is_same_v<T, int64_t>) {
          result["int_value"] = arg;
        } else if (std::is_same_v<T, double>) {
          result["double_value"] = arg;
        } else if (std::is_same_v<T, StringList>) {
          result["list_string_value"] = arg;
        } else if (std::is_same_v<T, std::vector<int64_t>>) {
          result["list_int_value"] = arg;
        } else if (std::is_same_v<T, std::vector<double>>) {
          result["list_double_value"] = arg;
        } else if (std::is_same_v<T, StringStringMap>) {
          result["map_string_value"] = arg;
        } else if (std::is_same_v<T, StringIntMap>) {
          result["map_int_value"] = arg;
        } else if (std::is_same_v<T, StringDoubleMap>) {
          result["map_double_value"] = arg;
        } else if (std::is_same_v<T, RankedActionList>) {
          result["ranked_action_list"] = arg;
        } else if (std::is_same_v<
                       T,
                       std::unordered_map<std::string, StringDoubleMap>>) {
          result["map_map_double_value"] = arg;
        } else {
          LOG(FATAL) << "INVALID OUTPUT OPERATOR";
          LOG_AND_THROW("Invalid output operator");
        }
        return result;
      },
      value);
}

} // namespace reagent
