#pragma once

#include <string>
#include <nlohmann/json.hpp>
#include "cmd_line_parser.h"

class ConfigLoader {
public:
    static nlohmann::json loadConfig(const CommandLineArguments& args);
    
private:
    static nlohmann::json loadFromFile(const std::string& configFile);
    static void mergeConfigs(nlohmann::json& base, const nlohmann::json& overrides);
    static std::string expandVariables(const std::string& value, const nlohmann::json& config);
};