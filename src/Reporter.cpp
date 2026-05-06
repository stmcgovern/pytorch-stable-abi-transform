#include "Reporter.h"
#include <llvm/Support/raw_ostream.h>

namespace stable_abi {

void Reporter::addFinding(FindingKind kind, const clang::SourceManager &SM,
                          clang::SourceLocation loc, std::string_view old_text,
                          std::string_view new_text, bool is_flag) {
    if (loc.isInvalid())
        return;

    auto ploc = SM.getPresumedLoc(loc);
    if (ploc.isInvalid())
        return;

    addFinding(kind, ploc.getFilename(), ploc.getLine(), ploc.getColumn(),
               old_text, new_text, is_flag);
}

void Reporter::addFinding(FindingKind kind, std::string_view file,
                           unsigned line, unsigned col,
                           std::string_view old_text,
                           std::string_view new_text, bool is_flag) {
    if (!seen_.emplace(std::string(file), line, col, std::string(old_text)).second)
        return;

    findings_.push_back(
        {kind, std::string(file), line, col, std::string(old_text),
         std::string(new_text), is_flag});

    if (is_flag)
        ++flag_count_;
    else
        ++rewrite_count_;
}

void Reporter::printReport() const {
    for (const auto &f : findings_) {
        if (f.is_flag) {
            llvm::outs() << "[FLAG]  ";
        } else {
            llvm::outs() << "[" << kindLabel(f.kind) << "] ";
        }
        llvm::outs() << f.file << ":" << f.line << ":" << f.col << "  "
                      << f.old_text << " -> "
                      << (f.is_flag ? "(manual review) " : "") << f.new_text
                      << "\n";
    }
}

void Reporter::printSummary() const {
    llvm::outs() << "\nSummary: " << rewrite_count_ << " auto-rewritable, "
                 << flag_count_ << " flagged for manual review\n";
}

static std::string jsonEscape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c;
        }
    }
    return out;
}

void Reporter::printJson() const {
    llvm::outs() << "{\n  \"findings\": [\n";
    for (size_t i = 0; i < findings_.size(); ++i) {
        const auto &f = findings_[i];
        llvm::outs() << "    {";
        llvm::outs() << "\"kind\": \"" << kindLabel(f.kind) << "\", ";
        llvm::outs() << "\"file\": \"" << jsonEscape(f.file) << "\", ";
        llvm::outs() << "\"line\": " << f.line << ", ";
        llvm::outs() << "\"col\": " << f.col << ", ";
        llvm::outs() << "\"old\": \"" << jsonEscape(f.old_text) << "\", ";
        llvm::outs() << "\"new\": \"" << jsonEscape(f.new_text) << "\", ";
        llvm::outs() << "\"flag\": " << (f.is_flag ? "true" : "false");
        llvm::outs() << "}";
        if (i + 1 < findings_.size())
            llvm::outs() << ",";
        llvm::outs() << "\n";
    }
    llvm::outs() << "  ],\n";
    llvm::outs() << "  \"rewrites\": " << rewrite_count_ << ",\n";
    llvm::outs() << "  \"flags\": " << flag_count_ << ",\n";
    llvm::outs() << "  \"parse_errors\": " << parse_error_count_ << "\n";
    llvm::outs() << "}\n";
}

void Reporter::recordParseError(const std::string &file) {
    ++parse_error_count_;
    ++parse_errors_by_file_[file];
}

void Reporter::printParseWarnings() const {
    if (parse_error_count_ == 0)
        return;
    llvm::errs() << "\nwarning: parse errors in "
                 << parse_errors_by_file_.size()
                 << " file(s) — results may be incomplete\n";
    for (const auto &[file, count] : parse_errors_by_file_)
        llvm::errs() << "  " << file << ": " << count << " error(s)\n";
}

void Reporter::suppressRedundantFlags() {
    std::set<std::pair<std::string, unsigned>> covered;
    for (const auto &f : findings_)
        if (!f.is_flag)
            covered.insert({f.file, f.line});

    size_t removed = 0;
    auto it = std::remove_if(findings_.begin(), findings_.end(),
        [&covered, &removed](const Finding &f) {
            if (f.is_flag && covered.count({f.file, f.line})) {
                ++removed;
                return true;
            }
            return false;
        });
    findings_.erase(it, findings_.end());
    flag_count_ -= removed;
}

bool Reporter::hasNonIncludeFindingsForFile(std::string_view filename) const {
    for (const auto &f : findings_) {
        if (f.kind != FindingKind::Include && f.file == filename)
            return true;
    }
    return false;
}

const char *Reporter::kindLabel(FindingKind kind) {
    switch (kind) {
    case FindingKind::Include:
        return "INCL ";
    case FindingKind::Macro:
        return "MACRO";
    case FindingKind::Type:
        return "TYPE ";
    case FindingKind::ScalarType:
        return "STYPE";
    case FindingKind::DataPtr:
        return "DPTR ";
    case FindingKind::CudaStream:
        return "STRM ";
    case FindingKind::DeviceGuard:
        return "GUARD";
    case FindingKind::MethodToFunc:
        return "M2F  ";
    case FindingKind::FreeFunc:
        return "FUNC ";
    case FindingKind::Flag:
        return "FLAG ";
    }
    return "?????";
}

} // namespace stable_abi
