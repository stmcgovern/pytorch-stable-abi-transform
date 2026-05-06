#pragma once

#include "Reporter.h"
#include <clang/Tooling/Transformer/RewriteRule.h>

namespace stable_abi {

clang::transformer::RewriteRule buildTransformerRules(
    Reporter &reporter, bool rewrite_mode,
    const std::string &projectRoot = "");

} // namespace stable_abi
