#pragma once

#include "AstCallbacks.h"
#include "PreprocessorCallbacks.h"
#include "Reporter.h"
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Refactoring/AtomicChange.h>
#include <clang/Tooling/Transformer/Transformer.h>
#include <clang/Tooling/Tooling.h>

namespace stable_abi {

class StableAbiConsumer : public clang::ASTConsumer {
public:
    StableAbiConsumer(FileReplacements &fileRepls, Reporter &rep,
                      bool rewrite, const std::string &projectRoot,
                      PreprocessorCallbacks *ppCallbacks = nullptr);
    void HandleTranslationUnit(clang::ASTContext &Context) override;

private:
    FileReplacements &file_repls_;
    bool rewrite_mode_;
    PreprocessorCallbacks *pp_callbacks_;
    std::vector<clang::tooling::AtomicChange> changes_;
    clang::ast_matchers::MatchFinder finder_;
    clang::tooling::Transformer transformer_;
    DeviceGuardCallback guardCallback_;
    CudaStreamCallback streamCallback_;
};

class StableAbiFrontendAction : public clang::ASTFrontendAction {
public:
    StableAbiFrontendAction(Reporter &reporter, bool rewrite, bool json,
                            const std::string &projectRoot, bool dry_run = false)
        : reporter_(reporter), rewrite_mode_(rewrite), json_mode_(json),
          project_root_(projectRoot), dry_run_(dry_run) {}

    std::unique_ptr<clang::ASTConsumer>
    CreateASTConsumer(clang::CompilerInstance &CI,
                      llvm::StringRef InFile) override;
    void EndSourceFileAction() override;

private:
    clang::Rewriter rewriter_;
    Reporter &reporter_;
    FileReplacements file_repls_;
    bool rewrite_mode_;
    bool json_mode_;
    std::string project_root_;
    bool dry_run_;
};

class StableAbiActionFactory
    : public clang::tooling::FrontendActionFactory {
public:
    StableAbiActionFactory(bool rewrite, bool json,
                           const std::string &projectRoot = "",
                           bool dry_run = false)
        : rewrite_mode_(rewrite), json_mode_(json),
          project_root_(projectRoot), dry_run_(dry_run) {}

    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<StableAbiFrontendAction>(
            reporter_, rewrite_mode_, json_mode_, project_root_, dry_run_);
    }

    Reporter &getReporter() { return reporter_; }

private:
    bool rewrite_mode_;
    bool json_mode_;
    std::string project_root_;
    bool dry_run_;
    Reporter reporter_;
};

class ParseDiagConsumer : public clang::DiagnosticConsumer {
public:
    explicit ParseDiagConsumer(Reporter &reporter) : reporter_(reporter) {}

    void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                          const clang::Diagnostic &info) override {
        DiagnosticConsumer::HandleDiagnostic(level, info);
        if (level < clang::DiagnosticsEngine::Error)
            return;
        std::string file = "<unknown>";
        if (info.hasSourceManager()) {
            auto &SM = info.getSourceManager();
            auto loc = info.getLocation();
            if (loc.isValid()) {
                auto ploc = SM.getPresumedLoc(loc);
                if (ploc.isValid())
                    file = ploc.getFilename();
            }
        }
        reporter_.recordParseError(file);
    }

private:
    Reporter &reporter_;
};

} // namespace stable_abi
