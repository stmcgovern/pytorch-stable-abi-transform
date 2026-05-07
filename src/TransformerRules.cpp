#include "TransformerRules.h"
#include "Helpers.h"
#include "Rules.h"
#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchers.h>
#include <clang/Lex/Lexer.h>
#include <clang/Tooling/Transformer/RangeSelector.h>
#include <clang/Tooling/Transformer/RewriteRule.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Regex.h>

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::transformer;

namespace stable_abi {

struct LocFilter {
    ast_matchers::internal::Matcher<Stmt> stmt;
    ast_matchers::internal::Matcher<TypeLoc> typeLoc;
    ast_matchers::internal::Matcher<Decl> decl;
    std::string projectRoot;
};

static LocFilter buildLocFilter(const std::string &projectRoot) {
    if (projectRoot.empty())
        return {isExpansionInMainFile(), isExpansionInMainFile(),
                isExpansionInMainFile(), ""};
    auto regex = "^" + llvm::Regex::escape(projectRoot);
    return {isExpansionInFileMatching(regex),
            isExpansionInFileMatching(regex),
            isExpansionInFileMatching(regex),
            projectRoot};
}

static bool shouldUseConstDataPtr(const CXXMemberCallExpr *CE,
                                  ASTContext &Ctx) {
    auto parents = Ctx.getParents(*CE);
    while (!parents.empty()) {
        const auto &parent = parents[0];

        if (const auto *cast = parent.get<CastExpr>()) {
            const auto resultType = cast->getType();
            if (resultType->isPointerType())
                return resultType->getPointeeType().isConstQualified();
        }

        if (const auto *call = parent.get<CallExpr>()) {
            if (const auto *FD = call->getDirectCallee()) {
                for (unsigned i = 0; i < call->getNumArgs(); ++i) {
                    if (call->getArg(i)->IgnoreParenImpCasts() == CE) {
                        if (i < FD->getNumParams()) {
                            auto paramType = FD->getParamDecl(i)->getType();
                            if (paramType->isPointerType())
                                return paramType->getPointeeType()
                                    .isConstQualified();
                        }
                        break;
                    }
                }
            }
            break;
        }

        if (const auto *VD = parent.get<VarDecl>()) {
            const auto varType = VD->getType();
            if (varType->isPointerType())
                return varType->getPointeeType().isConstQualified();
            if (varType->getContainedAutoType())
                break;
            return false;
        }

        if (const auto *RCE = parent.get<CXXReinterpretCastExpr>()) {
            const auto resultType = RCE->getType();
            if (resultType->isPointerType())
                return resultType->getPointeeType().isConstQualified();
            break;
        }
        if (const auto *SCE = parent.get<CXXStaticCastExpr>()) {
            const auto resultType = SCE->getType();
            if (resultType->isPointerType())
                return resultType->getPointeeType().isConstQualified();
            break;
        }

        parents = Ctx.getParents(parent);
    }

    const auto *objArg = CE->getImplicitObjectArgument();
    if (objArg) {
        const auto objType = objArg->getType();
        if (objType.isConstQualified() ||
            (objType->isReferenceType() &&
             objType.getNonReferenceType().isConstQualified()))
            return true;
    }
    return false;
}

static llvm::Expected<SmallVector<Edit, 1>> noEdits() {
    return SmallVector<Edit, 1>{};
}

static ast_matchers::internal::Matcher<NamedDecl>
hasAnyNameFromVec(std::vector<std::string> names) {
    return ast_matchers::internal::Matcher<NamedDecl>(
        new ast_matchers::internal::HasNameMatcher(std::move(names)));
}

enum class MacroPolicy { RejectAll, AllowArgs };

static std::optional<SourceLocation> getRewritableLoc(
    SourceLocation loc, const SourceManager &SM,
    llvm::StringRef projectRoot = "",
    MacroPolicy policy = MacroPolicy::RejectAll) {
    auto spelling = SM.getSpellingLoc(loc);
    if (!isInProjectScope(SM, spelling, projectRoot))
        return std::nullopt;
    if (SM.isMacroBodyExpansion(loc))
        return std::nullopt;
    if (policy == MacroPolicy::RejectAll && SM.isMacroArgExpansion(loc))
        return std::nullopt;
    return spelling;
}

static llvm::Expected<SmallVector<Edit, 1>>
singleEdit(SourceLocation start, unsigned len, std::string replacement) {
    Edit E;
    E.Kind = EditKind::Range;
    E.Range = CharSourceRange::getCharRange(
        start, start.getLocWithOffset(static_cast<int>(len)));
    E.Replacement = std::move(replacement);
    return SmallVector<Edit, 1>{std::move(E)};
}

static llvm::Expected<SmallVector<Edit, 1>>
singleEdit(SourceRange range, std::string replacement) {
    Edit E;
    E.Kind = EditKind::Range;
    E.Range = CharSourceRange::getTokenRange(range);
    E.Replacement = std::move(replacement);
    return SmallVector<Edit, 1>{std::move(E)};
}

// ---------------------------------------------------------------------------
// Type rewrite rules (from kTypeRules)
// ---------------------------------------------------------------------------

static bool isEnumQualifierType(std::string_view typeName) {
    return typeName == "at::ScalarType" || typeName == "c10::ScalarType" ||
           typeName == "at::DeviceType" || typeName == "c10::DeviceType";
}

static void addTypeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                         bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *TL = R.Nodes.getNodeAs<TypeLoc>("tl");
        if (!TL)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(TL->getBeginLoc(), *R.SourceManager,
                                          projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto text = getSourceText(TL->getSourceRange(), SM,
                                  R.Context->getLangOpts());

        for (const auto &rule : kTypeRules) {
            auto pos = text.find(rule.old_text);
            if (pos == std::string::npos)
                continue;

            auto endPos = pos + rule.old_text.size();
            if (endPos < text.size()) {
                char next = text[endPos];
                if (next == ':' && isEnumQualifierType(rule.old_text))
                    continue;
                if (std::isalnum(static_cast<unsigned char>(next)) ||
                    next == '_')
                    continue;
            }

            const bool flag_only = rule.new_text.empty();
            std::string_view suggestion = flag_only
                ? "decompose into explicit scalar_type, layout, device args"
                : rule.new_text;

            rep.addFinding(FindingKind::Type, SM, TL->getBeginLoc(),
                           rule.old_text, suggestion, flag_only);
            if (!rewrite || flag_only)
                return noEdits();

            auto replaceStart = rewriteLoc->getLocWithOffset(
                static_cast<int>(pos));
            return singleEdit(replaceStart,
                              static_cast<unsigned>(rule.old_text.size()),
                              std::string(suggestion));
        }
        return noEdits();
    };

    rules.push_back(makeRule(
        typeLoc(loc.typeLoc,
                ast_matchers::loc(qualType(hasDeclaration(namedDecl(hasAnyName(
            "at::Tensor", "torch::Tensor",
            "c10::Device", "at::Device", "torch::Device",
            "at::ScalarType", "c10::ScalarType", "torch::Dtype",
            "at::DeviceType", "c10::DeviceType",
            "at::Layout", "c10::Layout",
            "at::MemoryFormat", "c10::MemoryFormat",
            "at::Half", "c10::Half",
            "at::BFloat16", "c10::BFloat16",
            "c10::Float8_e4m3fn", "c10::Float8_e4m3fnuz",
            "c10::Float8_e5m2", "c10::Float8_e5m2fnuz",
            "c10::CppTypeToScalarType",
            "torch::TensorOptions", "at::TensorOptions", "c10::TensorOptions"
        ))))))
            .bind("tl"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Enum shorthand rules (shared by ScalarType and DeviceType)
// ---------------------------------------------------------------------------

// Macro-expanded arguments can produce multiple AST matches at the same source
// offset. shared_ptr is needed because std::function (inside EditGenerator)
// requires a copyable lambda.
template <typename ShorthandTable>
static llvm::Expected<SmallVector<Edit, 1>> rewriteEnumRef(
    const MatchFinder::MatchResult &R,
    Reporter &rep, bool rewrite,
    llvm::DenseSet<unsigned> &dedup,
    llvm::StringRef bindName,
    const ShorthandTable &shorthands,
    std::string_view typePrefix1, std::string_view typePrefix2,
    std::string_view macroBodyDesc,
    llvm::StringRef projectRoot = "") {

    const auto *DRE = R.Nodes.getNodeAs<DeclRefExpr>(bindName);
    if (!DRE)
        return noEdits();

    SourceLocation loc = DRE->getBeginLoc();
    const auto &SM = *R.SourceManager;
    const auto &LO = R.Context->getLangOpts();
    SourceLocation spellingLoc = SM.getSpellingLoc(loc);

    if (!isInProjectScope(SM, spellingLoc, projectRoot))
        return noEdits();
    if (!dedup.insert(SM.getFileOffset(spellingLoc)).second)
        return noEdits();
    if (SM.isMacroBodyExpansion(loc)) {
        rep.addFinding(FindingKind::ScalarType, SM, spellingLoc,
                       macroBodyDesc, "(inside macro body)", true);
        return noEdits();
    }

    const bool in_macro_arg = SM.isMacroArgExpansion(loc);
    SourceRange textRange = in_macro_arg
        ? SourceRange(SM.getSpellingLoc(DRE->getBeginLoc()),
                       SM.getSpellingLoc(DRE->getEndLoc()))
        : DRE->getSourceRange();
    std::string text = getSourceText(textRange, SM, LO);
    SourceLocation rewriteLoc = in_macro_arg ? spellingLoc : loc;

    for (const auto &rule : shorthands) {
        if (text == rule.old_text) {
            rep.addFinding(FindingKind::ScalarType, SM, spellingLoc,
                           rule.old_text, rule.new_text);
            if (!rewrite)
                return noEdits();
            return singleEdit(rewriteLoc,
                              static_cast<unsigned>(rule.old_text.size()),
                              std::string(rule.new_text));
        }
    }

    for (const auto &rule : kTypeRules) {
        if (rule.old_text != typePrefix1 && rule.old_text != typePrefix2)
            continue;
        if (text.starts_with(rule.old_text)) {
            std::string newText = std::string(rule.new_text) +
                                  text.substr(rule.old_text.size());
            rep.addFinding(FindingKind::ScalarType, SM, spellingLoc,
                           text, newText);
            if (!rewrite)
                return noEdits();
            return singleEdit(rewriteLoc,
                              static_cast<unsigned>(text.size()),
                              newText);
        }
    }

    return noEdits();
}

static void addScalarTypeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                               bool rewrite, const LocFilter &loc) {
    auto shorthandMatcher = declRefExpr(loc.stmt,
        to(namedDecl(hasAnyName(
        "kFloat", "kFloat16", "kFloat32", "kFloat64",
        "kDouble", "kHalf", "kBFloat16", "kBool",
        "kByte", "kChar", "kShort", "kInt", "kInt8",
        "kInt16", "kInt32", "kInt64", "kLong",
        "kFloat8_e4m3fn", "kFloat8_e5m2", "kFloat8_e4m3fnuz",
        "kFloat8_e5m2fnuz", "kFloat8_e8m0fnu",
        "kFloat4_e2m1fn_x2",
        "kComplexHalf", "kComplexFloat",
        "kComplexDouble", "kUInt8"))))
        .bind("scalarRef");
    auto enumMatcher = declRefExpr(loc.stmt,
        to(enumConstantDecl(hasDeclContext(enumDecl(hasName("ScalarType"))))))
        .bind("scalarRef");

    auto dedup = std::make_shared<llvm::DenseSet<unsigned>>();
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, dedup, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        return rewriteEnumRef(R, rep, rewrite, *dedup, "scalarRef",
                              kScalarTypeShorthands,
                              "at::ScalarType", "c10::ScalarType",
                              "scalar type shorthand", projectRoot);
    };

    rules.push_back(makeRule(shorthandMatcher, EditGenerator(editGen)));
    rules.push_back(makeRule(enumMatcher, EditGenerator(editGen)));
}

static void addDeviceTypeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                               bool rewrite, const LocFilter &loc) {
    auto shorthandMatcher = declRefExpr(loc.stmt,
        to(namedDecl(hasAnyName(
        "kCPU", "kCUDA", "kMKLDNN", "kOPENGL", "kOPENCL",
        "kIDEEP", "kHIP", "kFPGA", "kMAIA", "kXLA",
        "kVulkan", "kMetal", "kXPU", "kMPS", "kMeta",
        "kHPU", "kVE", "kLazy", "kIPU", "kMTIA",
        "kPrivateUse1"))))
        .bind("deviceRef");
    auto enumMatcher = declRefExpr(loc.stmt,
        to(enumConstantDecl(hasDeclContext(enumDecl(hasName("DeviceType"))))))
        .bind("deviceRef");

    auto dedup = std::make_shared<llvm::DenseSet<unsigned>>();
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, dedup, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        return rewriteEnumRef(R, rep, rewrite, *dedup, "deviceRef",
                              kDeviceTypeShorthands,
                              "at::DeviceType", "c10::DeviceType",
                              "device type", projectRoot);
    };

    rules.push_back(makeRule(shorthandMatcher, EditGenerator(editGen)));
    rules.push_back(makeRule(enumMatcher, EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// data_ptr → mutable_data_ptr / const_data_ptr
// ---------------------------------------------------------------------------

static void addDataPtrRule(std::vector<RewriteRule> &rules, Reporter &rep,
                           bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("dataPtrCall");
        if (!CE || !isTensorType(CE->getImplicitObjectArgument()))
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto spellingLoc = getRewritableLoc(CE->getBeginLoc(), SM,
                                            projectRoot, MacroPolicy::AllowArgs);
        if (!spellingLoc)
            return noEdits();

        const auto *ME = R.Nodes.getNodeAs<MemberExpr>("dataPtrMember");
        if (!ME)
            return noEdits();

        if (ME->hasExplicitTemplateArgs()) {
            const auto tArgs = ME->template_arguments();
            if (!tArgs.empty()) {
                const auto argType = tArgs[0].getArgument().getAsType();
                if (argType->isDependentType()) {
                    rep.addFinding(FindingKind::DataPtr, SM, *spellingLoc,
                                   "data_ptr<dependent>",
                                   "mutable_data_ptr<T> (dependent type)",
                                   true);
                    return noEdits();
                }
            }
        }

        const bool use_const = shouldUseConstDataPtr(CE, *R.Context);
        const std::string replacement =
            use_const ? "const_data_ptr" : "mutable_data_ptr";

        auto loc = CE->getBeginLoc();
        const bool in_macro = SM.isMacroArgExpansion(loc);
        auto memberLoc = in_macro ? SM.getSpellingLoc(ME->getMemberLoc())
                                  : ME->getMemberLoc();

        rep.addFinding(FindingKind::DataPtr, SM, *spellingLoc, "data_ptr",
                       replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(memberLoc, 8, replacement);
    };

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(memberExpr(member(hasName("data_ptr"))).bind("dataPtrMember")))
            .bind("dataPtrCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Method-to-free-function rules (from kMethodToFreeFuncRules)
// ---------------------------------------------------------------------------

static void addMethodToFuncRules(std::vector<RewriteRule> &rules,
                                 Reporter &rep, bool rewrite,
                                 const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("methodCall");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();

        const auto *ME = R.Nodes.getNodeAs<MemberExpr>("methodMember");
        if (!ME)
            return noEdits();
        auto methodName = ME->getMemberDecl()->getNameAsString();

        const auto *obj = CE->getImplicitObjectArgument();
        if (!isTensorType(obj))
            return noEdits();

        for (const auto &rule : kMethodToFreeFuncRules) {
            if (methodName != rule.method_name)
                continue;

            auto objText = getSourceText(obj->getSourceRange(), SM, LO);
            std::string argsText;
            auto argsRange = callArgs("methodCall")(R);
            if (argsRange)
                argsText = Lexer::getSourceText(*argsRange, SM, LO).str();
            else
                llvm::consumeError(argsRange.takeError());

            std::string replacement =
                std::string(rule.free_func) + "(" + objText;
            if (!argsText.empty())
                replacement += ", " + argsText;
            replacement += ")";

            auto fullText = getSourceText(CE->getSourceRange(), SM, LO);
            rep.addFinding(FindingKind::MethodToFunc, SM, CE->getBeginLoc(),
                           fullText, replacement);
            if (!rewrite)
                return noEdits();

            return singleEdit(CE->getSourceRange(), replacement);
        }
        return noEdits();
    };

    std::vector<std::string> methodNames;
    for (const auto &rule : kMethodToFreeFuncRules)
        methodNames.push_back(std::string(rule.method_name));

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(memberExpr(member(cxxMethodDecl(
                hasAnyNameFromVec(std::move(methodNames)))))
                .bind("methodMember")))
            .bind("methodCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Method rename rules (dtype→scalar_type, sizes()[i]→size(i))
// ---------------------------------------------------------------------------

static void addMethodRenameRules(std::vector<RewriteRule> &rules,
                                 Reporter &rep, bool rewrite,
                                 const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;

    // TensorOptions constructor — flag only
    auto tensorOptsGen = [&rep, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *Ctor =
            R.Nodes.getNodeAs<CXXConstructExpr>("tensorOptionsConstruct");
        if (!Ctor)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(Ctor->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        auto text = getSourceText(Ctor->getSourceRange(), *R.SourceManager,
                                  R.Context->getLangOpts());
        rep.addFinding(FindingKind::Type, *R.SourceManager, Ctor->getBeginLoc(),
                       text,
                       "decompose into explicit scalar_type, layout, device args",
                       true);
        return noEdits();
    };

    rules.push_back(makeRule(
        cxxConstructExpr(loc.stmt,
            hasType(hasUnqualifiedDesugaredType(recordType(
            hasDeclaration(cxxRecordDecl(hasName("TensorOptions")))))))
            .bind("tensorOptionsConstruct"),
        EditGenerator(tensorOptsGen)));

    // .sizes()[i] → .size(i), .strides()[i] → .stride(i)
    auto rewriteSizes = [&rep, rewrite](
            const MatchFinder::MatchResult &R,
            const CXXMemberCallExpr *sizesCall,
            const Expr *idx, SourceRange fullRange)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();

        const auto *calleeME = dyn_cast<MemberExpr>(sizesCall->getCallee());
        if (!calleeME)
            return noEdits();

        auto methodName = calleeME->getMemberDecl()->getName();
        const char *newMethod;
        if (methodName == "sizes")
            newMethod = "size";
        else if (methodName == "strides")
            newMethod = "stride";
        else
            return noEdits();

        const auto *obj = sizesCall->getImplicitObjectArgument();
        if (!obj)
            return noEdits();

        auto objText = getSourceText(obj->getSourceRange(), SM, LO);
        auto idxText = getSourceText(idx->getSourceRange(), SM, LO);
        auto fullText = getSourceText(fullRange, SM, LO);

        std::string replacement =
            objText + "." + newMethod + "(" + idxText + ")";
        rep.addFinding(FindingKind::MethodToFunc, SM, fullRange.getBegin(),
                       fullText, replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(fullRange, replacement);
    };

    auto sizesGen = [rewriteSizes, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *ASE =
            R.Nodes.getNodeAs<ArraySubscriptExpr>("sizesSubscript");
        if (!ASE)
            return noEdits();
        if (!getRewritableLoc(ASE->getBeginLoc(), *R.SourceManager, projectRoot))
            return noEdits();
        const auto *sizesCall =
            R.Nodes.getNodeAs<CXXMemberCallExpr>("sizesCall");
        if (!sizesCall)
            return noEdits();
        return rewriteSizes(R, sizesCall, ASE->getIdx(),
                            ASE->getSourceRange());
    };

    rules.push_back(makeRule(
        arraySubscriptExpr(loc.stmt,
            hasBase(hasDescendant(
            cxxMemberCallExpr(callee(cxxMethodDecl(
                hasAnyName("sizes", "strides"))))
                .bind("sizesCall"))))
            .bind("sizesSubscript"),
        EditGenerator(sizesGen)));

    // CXXOperatorCallExpr version (overloaded operator[])
    auto sizesOpGen = [rewriteSizes, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *Sub =
            R.Nodes.getNodeAs<CXXOperatorCallExpr>("sizesSubscriptOp");
        if (!Sub || Sub->getNumArgs() < 2)
            return noEdits();
        if (!getRewritableLoc(Sub->getBeginLoc(), *R.SourceManager, projectRoot))
            return noEdits();
        const auto *sizesCall = dyn_cast<CXXMemberCallExpr>(
            Sub->getArg(0)->IgnoreImplicit());
        if (!sizesCall)
            return noEdits();
        return rewriteSizes(R, sizesCall, Sub->getArg(1),
                            Sub->getSourceRange());
    };

    rules.push_back(makeRule(
        cxxOperatorCallExpr(loc.stmt,
            hasOverloadedOperatorName("[]"),
            hasArgument(0, ignoringImplicit(cxxMemberCallExpr(
                callee(cxxMethodDecl(hasAnyName("sizes", "strides")))))))
            .bind("sizesSubscriptOp"),
        EditGenerator(sizesOpGen)));

    // .dtype() → .scalar_type()
    auto dtypeGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("methodRename");
        if (!CE || !isTensorType(CE->getImplicitObjectArgument()))
            return noEdits();

        auto exprLoc = CE->getBeginLoc();
        const auto &SM = *R.SourceManager;
        auto spellingLoc = SM.getSpellingLoc(exprLoc);
        if (!isInProjectScope(SM, spellingLoc, projectRoot))
            return noEdits();
        if (SM.isMacroBodyExpansion(exprLoc)) {
            rep.addFinding(FindingKind::MethodToFunc, SM, spellingLoc,
                           "dtype", "scalar_type (inside macro body)", true);
            return noEdits();
        }
        const bool in_macro_arg = SM.isMacroArgExpansion(exprLoc);

        const auto *ME = R.Nodes.getNodeAs<MemberExpr>("dtypeMember");
        if (!ME)
            return noEdits();

        auto methodName = ME->getMemberDecl()->getNameAsString();
        for (const auto &rule : kMethodRenameRules) {
            if (methodName != rule.old_name)
                continue;

            auto memberLoc = in_macro_arg
                ? SM.getSpellingLoc(ME->getMemberLoc())
                : ME->getMemberLoc();
            rep.addFinding(FindingKind::MethodToFunc, SM, memberLoc,
                           rule.old_name, rule.new_name);
            if (!rewrite)
                return noEdits();

            return singleEdit(memberLoc,
                              static_cast<unsigned>(rule.old_name.size()),
                              std::string(rule.new_name));
        }
        return noEdits();
    };

    std::vector<std::string> renameNames;
    for (const auto &rule : kMethodRenameRules)
        renameNames.push_back(std::string(rule.old_name));

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(memberExpr(member(hasAnyNameFromVec(std::move(renameNames))))
                .bind("dtypeMember")))
            .bind("methodRename"),
        EditGenerator(dtypeGen)));
}

// ---------------------------------------------------------------------------
// Free function rules (from kFreeFuncRules)
// ---------------------------------------------------------------------------

static void addElementSizeRules(std::vector<RewriteRule> &rules, Reporter &rep,
                                bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    // at::elementSize(tensor.scalar_type/dtype()) → tensor.element_size()
    auto exactGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CallExpr>("elemSizeExact");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();
        const auto *obj = R.Nodes.getNodeAs<Expr>("elemObj");
        if (!obj)
            return noEdits();

        auto objText = getSourceText(obj->getSourceRange(), SM, LO);
        std::string replacement = objText + ".element_size()";
        auto fullText = getSourceText(CE->getSourceRange(), SM, LO);

        rep.addFinding(FindingKind::FreeFunc, SM, CE->getBeginLoc(),
                       fullText, replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(CE->getSourceRange(), replacement);
    };

    rules.push_back(makeRule(
        callExpr(loc.stmt,
            callee(functionDecl(hasAnyName("at::elementSize", "c10::elementSize"))),
            hasArgument(0, ignoringImplicit(
                cxxMemberCallExpr(
                    callee(cxxMethodDecl(hasAnyName("scalar_type", "dtype"))),
                    on(expr().bind("elemObj"))))))
            .bind("elemSizeExact"),
        EditGenerator(exactGen)));

    // at::elementSize(...) — flag only (can't auto-rewrite)
    auto fallbackGen = [&rep, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CallExpr>("elemSizeFallback");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        auto text = getSourceText(CE->getSourceRange(), *R.SourceManager,
                                  R.Context->getLangOpts());
        rep.addFinding(FindingKind::FreeFunc, *R.SourceManager,
                       CE->getBeginLoc(), text,
                       "tensor.element_size()", true);
        return noEdits();
    };

    rules.push_back(makeRule(
        callExpr(loc.stmt,
            callee(functionDecl(hasAnyName("at::elementSize", "c10::elementSize"))))
            .bind("elemSizeFallback"),
        EditGenerator(fallbackGen)));
}

static void addFreeFuncRules(std::vector<RewriteRule> &rules, Reporter &rep,
                             bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CallExpr>("freeFuncCall");
        if (!CE)
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();

        const auto *calleeExpr = CE->getCallee()->IgnoreParenImpCasts();
        const auto *DRE = dyn_cast<DeclRefExpr>(calleeExpr);
        if (!DRE)
            return noEdits();
        const auto *FD = dyn_cast<FunctionDecl>(DRE->getDecl());
        if (!FD)
            return noEdits();

        auto canonicalName = FD->getQualifiedNameAsString();
        auto writtenText = getSourceText(DRE->getSourceRange(), SM, LO);

        for (const auto &rule : kFreeFuncRules) {
            if (canonicalName != rule.old_qualified &&
                writtenText != rule.old_qualified)
                continue;

            auto oldFunc = rule.old_qualified.substr(
                rule.old_qualified.rfind(':') + 1);
            auto newFunc = rule.new_qualified.substr(
                rule.new_qualified.rfind(':') + 1);
            bool name_changes = (oldFunc != newFunc);

            if (name_changes) {
                std::string suggestion = std::string(rule.new_qualified) +
                    " (function name changes — adjust arguments manually)";
                rep.addFinding(FindingKind::FreeFunc, SM, CE->getBeginLoc(),
                               writtenText, suggestion, true);
                return noEdits();
            }

            rep.addFinding(FindingKind::FreeFunc, SM, CE->getBeginLoc(),
                           writtenText, rule.new_qualified);
            if (!rewrite)
                return noEdits();

            return singleEdit(DRE->getSourceRange(),
                              std::string(rule.new_qualified));
        }
        return noEdits();
    };

    std::vector<std::string> funcNames;
    for (const auto &rule : kFreeFuncRules)
        funcNames.push_back(std::string(rule.old_qualified));

    rules.push_back(makeRule(
        callExpr(loc.stmt,
            callee(functionDecl(hasAnyNameFromVec(std::move(funcNames)))))
            .bind("freeFuncCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// .nbytes() → .numel() * .element_size()
// ---------------------------------------------------------------------------

static void addNbytesRule(std::vector<RewriteRule> &rules, Reporter &rep,
                          bool rewrite, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, rewrite, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *CE = R.Nodes.getNodeAs<CXXMemberCallExpr>("nbytesCall");
        if (!CE || !isTensorType(CE->getImplicitObjectArgument()))
            return noEdits();

        auto rewriteLoc = getRewritableLoc(CE->getBeginLoc(),
                                           *R.SourceManager, projectRoot);
        if (!rewriteLoc)
            return noEdits();

        const auto &SM = *R.SourceManager;
        const auto &LO = R.Context->getLangOpts();
        const auto *obj = CE->getImplicitObjectArgument();
        auto objText = getSourceText(obj->getSourceRange(), SM, LO);
        auto fullText = getSourceText(CE->getSourceRange(), SM, LO);

        std::string replacement = objText + ".numel() * " +
                                  objText + ".element_size()";

        const bool sideEffectFree =
            isa<DeclRefExpr>(obj->IgnoreParenImpCasts()) ||
            isa<MemberExpr>(obj->IgnoreParenImpCasts());

        if (!sideEffectFree) {
            rep.addFinding(FindingKind::MethodToFunc, SM, CE->getBeginLoc(),
                           fullText, replacement, true);
            return noEdits();
        }

        rep.addFinding(FindingKind::MethodToFunc, SM, CE->getBeginLoc(),
                       fullText, replacement);
        if (!rewrite)
            return noEdits();

        return singleEdit(CE->getSourceRange(), replacement);
    };

    rules.push_back(makeRule(
        cxxMemberCallExpr(loc.stmt,
            callee(cxxMethodDecl(hasName("nbytes"))))
            .bind("nbytesCall"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Catch-all detectors for unstable API usage
// ---------------------------------------------------------------------------

static bool isUnstableNamespace(llvm::StringRef qualName) {
    if (qualName.starts_with("at::") || qualName.starts_with("c10::"))
        return true;
    if (qualName.starts_with("torch::") &&
        !qualName.starts_with("torch::stable::") &&
        !qualName.starts_with("torch::headeronly::"))
        return true;
    return false;
}

static void addUnstableTypeCatchAll(std::vector<RewriteRule> &rules,
                                     Reporter &rep, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto dedup = std::make_shared<
        llvm::DenseSet<std::pair<const NamedDecl *, unsigned>>>();
    auto editGen = [&rep, projectRoot, dedup](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *TL = R.Nodes.getNodeAs<TypeLoc>("catchallType");
        if (!TL)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto spelling = SM.getSpellingLoc(TL->getBeginLoc());
        if (!isInProjectScope(SM, spelling, projectRoot))
            return noEdits();

        const auto *ND = R.Nodes.getNodeAs<NamedDecl>("catchallTypeDecl");
        if (!ND)
            return noEdits();

        auto qualName = ND->getQualifiedNameAsString();
        if (!isUnstableNamespace(qualName))
            return noEdits();

        unsigned line = SM.getSpellingLineNumber(spelling);
        if (!dedup->insert({ND, line}).second)
            return noEdits();

        bool inMacroBody = SM.isMacroBodyExpansion(TL->getBeginLoc());
        std::string text = inMacroBody
            ? qualName
            : getSourceText(TL->getSourceRange(), SM, R.Context->getLangOpts());
        std::string msg = "no stable equivalent — rewrite or remove " + qualName;
        if (inMacroBody) msg += " (inside macro body)";
        rep.addFinding(FindingKind::Flag, SM, spelling,
                       text, msg, true);
        return noEdits();
    };

    rules.push_back(makeRule(
        typeLoc(loc.typeLoc,
                ast_matchers::loc(qualType(hasDeclaration(
                    namedDecl().bind("catchallTypeDecl")))))
            .bind("catchallType"),
        EditGenerator(editGen)));
}

static void addUnstableRefCatchAll(std::vector<RewriteRule> &rules,
                                    Reporter &rep, const LocFilter &loc) {
    auto projectRoot = loc.projectRoot;
    auto editGen = [&rep, projectRoot](const MatchFinder::MatchResult &R)
            -> llvm::Expected<SmallVector<Edit, 1>> {
        const auto *DRE = R.Nodes.getNodeAs<DeclRefExpr>("catchallRef");
        if (!DRE)
            return noEdits();

        const auto &SM = *R.SourceManager;
        auto spelling = SM.getSpellingLoc(DRE->getBeginLoc());
        if (!isInProjectScope(SM, spelling, projectRoot))
            return noEdits();

        const auto *ND = R.Nodes.getNodeAs<NamedDecl>("catchallRefDecl");
        if (!ND)
            return noEdits();
        if (isa<CXXMethodDecl>(ND))
            return noEdits();
        if (const auto *FD = dyn_cast<FunctionDecl>(ND))
            if (FD->isOverloadedOperator())
                return noEdits();

        auto qualName = ND->getQualifiedNameAsString();
        if (!isUnstableNamespace(qualName))
            return noEdits();

        bool inMacroBody = SM.isMacroBodyExpansion(DRE->getBeginLoc());
        std::string text = inMacroBody
            ? qualName
            : getSourceText(DRE->getSourceRange(), SM, R.Context->getLangOpts());
        std::string msg = "no stable equivalent — rewrite or remove " + qualName;
        if (inMacroBody) msg += " (inside macro body)";
        rep.addFinding(FindingKind::Flag, SM, spelling,
                       text, msg, true);
        return noEdits();
    };

    rules.push_back(makeRule(
        declRefExpr(loc.stmt,
                    to(namedDecl().bind("catchallRefDecl")))
            .bind("catchallRef"),
        EditGenerator(editGen)));
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

RewriteRule buildTransformerRules(Reporter &rep, bool rewrite_mode,
                                  const std::string &projectRoot) {
    auto loc = buildLocFilter(projectRoot);
    std::vector<RewriteRule> rules;
    addTypeRules(rules, rep, rewrite_mode, loc);
    addScalarTypeRules(rules, rep, rewrite_mode, loc);
    addDeviceTypeRules(rules, rep, rewrite_mode, loc);
    addDataPtrRule(rules, rep, rewrite_mode, loc);
    addMethodToFuncRules(rules, rep, rewrite_mode, loc);
    addMethodRenameRules(rules, rep, rewrite_mode, loc);
    addElementSizeRules(rules, rep, rewrite_mode, loc);
    addFreeFuncRules(rules, rep, rewrite_mode, loc);
    addNbytesRule(rules, rep, rewrite_mode, loc);
    addUnstableTypeCatchAll(rules, rep, loc);
    addUnstableRefCatchAll(rules, rep, loc);
    return applyFirst(rules);
}

} // namespace stable_abi
