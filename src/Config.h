#pragma once

#include <llvm/Support/YAMLTraits.h>
#include <string>
#include <vector>

namespace stable_abi {

struct Config {
    std::string mode = "audit";
    std::string format = "text";
    std::string pytorch_root;
    std::string project_root;
    std::vector<std::string> compiler_flags;
    std::vector<std::string> include_paths;
    std::vector<std::string> extra_includes;
    std::vector<std::string> sources;
    std::string verify_method = "compile";
    std::string cuda_include;
};

bool loadConfig(const std::string &path, Config &out, std::string &error);
void printExampleConfig();

} // namespace stable_abi

namespace llvm {
namespace yaml {

template <> struct MappingTraits<stable_abi::Config> {
    static void mapping(IO &io, stable_abi::Config &c) {
        io.mapOptional("mode", c.mode);
        io.mapOptional("format", c.format);
        io.mapOptional("pytorch_root", c.pytorch_root);
        io.mapOptional("project_root", c.project_root);
        io.mapOptional("compiler_flags", c.compiler_flags);
        io.mapOptional("include_paths", c.include_paths);
        io.mapOptional("extra_includes", c.extra_includes);
        io.mapOptional("sources", c.sources);
        io.mapOptional("verify_method", c.verify_method);
        io.mapOptional("cuda_include", c.cuda_include);
    }
};

} // namespace yaml
} // namespace llvm
