#include "AstCallbacks.h"
#include "Helpers.h"
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Lex/Lexer.h>

using namespace clang;
using namespace clang::ast_matchers;

namespace stable_abi {

static void addReplacement(FileReplacements &fileRepls,
                           const SourceManager &SM,
                           CharSourceRange range, StringRef text,
                           const LangOptions &LO) {
    clang::tooling::Replacement R(SM, range, text, LO);
    auto &repls = fileRepls[R.getFilePath().str()];
    if (auto err = repls.add(R))
        llvm::consumeError(std::move(err));
}

// ---- DeviceGuardCallback ----

void DeviceGuardCallback::run(const MatchFinder::MatchResult &Result) {
    const auto *DS = Result.Nodes.getNodeAs<DeclStmt>("guardDeclStmt");
    const auto *VD = Result.Nodes.getNodeAs<VarDecl>("guardDecl");
    if (!VD)
        return;

    const auto &SM = *Result.SourceManager;
    const auto &LO = Result.Context->getLangOpts();
    auto loc = VD->getBeginLoc();

    if (!isInProjectScope(SM, SM.getSpellingLoc(loc), project_root_))
        return;
    if (SM.isMacroBodyExpansion(loc)) {
        auto text = getSourceText(VD->getSourceRange(), SM, LO);
        if (text.find("CUDAGuard") != std::string::npos) {
            reporter_.addFinding(FindingKind::DeviceGuard, SM,
                                 SM.getSpellingLoc(loc), text,
                                 "DeviceGuard usage inside macro body", true);
        }
        return;
    }

    auto text = getSourceText(VD->getSourceRange(), SM, LO);

    if (text.find("CUDAGuard") == std::string::npos)
        return;

    auto varName = VD->getNameAsString();

    std::string deviceExpr;

    if (const auto *init = VD->getInit()) {
        if (const auto *CE = dyn_cast<CXXConstructExpr>(
                init->IgnoreImplicit())) {
            if (CE->getNumArgs() > 0) {
                auto argText =
                    getSourceText(CE->getArg(0)->getSourceRange(), SM, LO);
                if (const auto *callArg = dyn_cast<CallExpr>(
                        CE->getArg(0)->IgnoreUnlessSpelledInSource())) {
                    if (auto *callee = callArg->getDirectCallee()) {
                        if (callee->getName() == "device_of" &&
                            callArg->getNumArgs() > 0) {
                            auto tensorText = getSourceText(
                                callArg->getArg(0)->getSourceRange(), SM, LO);
                            deviceExpr = tensorText + ".get_device_index()";
                        }
                    }
                    if (deviceExpr.empty()) {
                        if (const auto *memberCall =
                                dyn_cast<CXXMemberCallExpr>(callArg)) {
                            if (auto *method = memberCall->getMethodDecl()) {
                                if (method->getName() == "device") {
                                    auto objText = getSourceText(
                                        memberCall->getImplicitObjectArgument()
                                            ->getSourceRange(), SM, LO);
                                    deviceExpr = objText + ".get_device_index()";
                                }
                            }
                        }
                    }
                }
                if (deviceExpr.empty()) {
                    deviceExpr = argText;
                }
            }
        }
    }

    if (deviceExpr.empty()) {
        auto deviceOfPos = text.find("device_of(");
        if (deviceOfPos != std::string::npos) {
            auto argStart = deviceOfPos + 10;
            int depth = 1;
            auto argEnd = argStart;
            for (; argEnd < text.size() && depth > 0; ++argEnd) {
                if (text[argEnd] == '(')
                    ++depth;
                else if (text[argEnd] == ')')
                    --depth;
            }
            if (depth == 0 && argEnd > argStart + 1) {
                auto tensorName = text.substr(argStart, argEnd - argStart - 1);
                deviceExpr = tensorName + ".get_device_index()";
            }
        }
    }

    if (deviceExpr.empty()) {
        reporter_.addFinding(
            FindingKind::DeviceGuard, SM, loc, text,
            "torch::stable::accelerator::DeviceGuard " + varName +
                "(tensor.get_device_index())",
            true);
        return;
    }

    last_device_expr_ = deviceExpr;

    std::string replacement = "const torch::stable::accelerator::DeviceGuard " +
                              varName + "(" + deviceExpr + ");";

    reporter_.addFinding(FindingKind::DeviceGuard, SM, loc, text, replacement);
    if (rewrite_mode_) {
        SourceRange replaceRange = DS ? DS->getSourceRange() : VD->getSourceRange();
        auto stmtEnd = Lexer::findLocationAfterToken(
            replaceRange.getEnd(), tok::semi, SM, LO, false);
        if (stmtEnd.isValid()) {
            addReplacement(file_repls_, SM,
                CharSourceRange::getCharRange(replaceRange.getBegin(), stmtEnd),
                replacement, LO);
        } else {
            addReplacement(file_repls_, SM,
                CharSourceRange::getTokenRange(replaceRange),
                replacement, LO);
        }
    }
}

// ---- CudaStreamCallback ----

void CudaStreamCallback::run(const MatchFinder::MatchResult &Result) {
    const auto *DS = Result.Nodes.getNodeAs<DeclStmt>("streamDeclStmt");
    const auto *CE = Result.Nodes.getNodeAs<CallExpr>("streamCall");
    if (!CE)
        return;

    const auto &SM = *Result.SourceManager;
    const auto &LO = Result.Context->getLangOpts();
    auto loc = CE->getBeginLoc();

    if (!isInProjectScope(SM, SM.getSpellingLoc(loc), project_root_))
        return;
    if (SM.isMacroBodyExpansion(loc)) {
        auto text = getSourceText(CE->getSourceRange(), SM, LO);
        bool isStream =
            text.find("getCurrentCUDAStream") != std::string::npos ||
            text.find("getCurrentStream") != std::string::npos;
        if (isStream) {
            reporter_.addFinding(FindingKind::CudaStream, SM,
                                 SM.getSpellingLoc(loc), text,
                                 "CUDA stream usage inside macro body", true);
        }
        return;
    }

    auto text = getSourceText(CE->getSourceRange(), SM, LO);

    bool isCurrentStream =
        text.find("getCurrentCUDAStream") != std::string::npos ||
        text.find("getCurrentStream") != std::string::npos;

    if (!isCurrentStream)
        return;

    if (DS && DS->isSingleDecl()) {
        if (const auto *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
            auto varName = VD->getNameAsString();
            auto stmtText = getSourceText(DS->getSourceRange(), SM, LO);
            auto indent = getIndent(DS->getBeginLoc(), SM);

            const auto &device_expr = guard_cb_.getLastDeviceExpr();
            std::string deviceExpr = device_expr.empty() ? "-1" : device_expr;

            std::string replacement =
                "void* " + varName + "_ptr = nullptr;\n" +
                indent + "aoti_torch_get_current_cuda_stream(" + deviceExpr +
                ", &" + varName + "_ptr);\n" +
                indent + "const cudaStream_t " + varName +
                " = reinterpret_cast<cudaStream_t>(" + varName + "_ptr);";

            reporter_.addFinding(FindingKind::CudaStream, SM, loc, stmtText,
                                 replacement);
            if (rewrite_mode_) {
                auto stmtEnd = Lexer::findLocationAfterToken(
                    DS->getEndLoc(), tok::semi, SM, LO, false);
                if (stmtEnd.isValid()) {
                    addReplacement(file_repls_, SM,
                        CharSourceRange::getCharRange(DS->getBeginLoc(),
                                                      stmtEnd),
                        replacement, LO);
                } else {
                    addReplacement(file_repls_, SM,
                        CharSourceRange::getTokenRange(DS->getSourceRange()),
                        replacement, LO);
                }
            }
            return;
        }
    }

    reporter_.addFinding(
        FindingKind::CudaStream, SM, loc, text,
        "aoti_torch_get_current_cuda_stream(device_index, &stream_ptr)",
        true);
}

// ---- Matcher Registration (manual callbacks only) ----

void registerManualMatchers(MatchFinder &finder,
                            CudaStreamCallback &streamCallback,
                            DeviceGuardCallback &guardCallback) {
    // DeviceGuard
    finder.addMatcher(
        declStmt(containsDeclaration(
                     0, varDecl(hasType(hasUnqualifiedDesugaredType(recordType(
                                    hasDeclaration(cxxRecordDecl(hasAnyName(
                                        "OptionalCUDAGuard", "CUDAGuard",
                                        "InferenceMode")))))))
                            .bind("guardDecl")))
            .bind("guardDeclStmt"),
        &guardCallback);

    // CudaStream — with enclosing DeclStmt
    finder.addMatcher(
        declStmt(containsDeclaration(
                     0, varDecl(hasInitializer(hasDescendant(
                            callExpr(callee(functionDecl(hasAnyName(
                                         "getCurrentCUDAStream",
                                         "getCurrentStream"))))
                                .bind("streamCall"))))))
            .bind("streamDeclStmt"),
        &streamCallback);

    // CudaStream — standalone (not in a VarDecl)
    finder.addMatcher(
        callExpr(callee(functionDecl(
                     hasAnyName("getCurrentCUDAStream", "getCurrentStream"))),
                 unless(hasAncestor(declStmt())))
            .bind("streamCall"),
        &streamCallback);
}

} // namespace stable_abi
