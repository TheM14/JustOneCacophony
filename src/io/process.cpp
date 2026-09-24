#include "io/process.h"
#include "foundation/fs_utf8.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/wait.h>
#endif

namespace joc::io {

namespace {

std::string quote_argument(const std::string& argument) {
    if (!argument.empty() && argument.find_first_of(" \t\"") == std::string::npos) {
        return argument;
    }
    std::string quoted = "\"";
    unsigned backslashes = 0;
    for (const char c : argument) {
        if (c == '\\') {
            ++backslashes;
            continue;
        }
        if (c == '"') {
            quoted.append(backslashes * 2 + 1, '\\');
            quoted.push_back('"');
            backslashes = 0;
            continue;
        }
        quoted.append(backslashes, '\\');
        backslashes = 0;
        quoted.push_back(c);
    }
    quoted.append(backslashes * 2, '\\');
    quoted.push_back('"');
    return quoted;
}

std::string tail_of(const std::string& text, std::size_t limit) {
    if (text.size() <= limit) {
        return text;
    }
    return text.substr(text.size() - limit);
}

}  // namespace

Status run_process(const std::vector<std::string>& argv, ProcessResult* out) {
    if (out == nullptr || argv.empty()) {
        return Status::fail(JOC_ERR_INVALID_ARGUMENT, stage::kOutput, "empty command");
    }
    out->output.clear();
    out->exit_code = 0;

    const std::filesystem::path log_path =
        std::filesystem::temp_directory_path() /
        ("joc_process_" + std::to_string(std::random_device{}()) + ".log");

    auto read_log = [&]() {
#if defined(_WIN32)
        return;  // the Windows branch reads the handle it opened
#else
        std::ifstream log = fs_utf8::open_input(fs_utf8::from_path(log_path));
        if (log) {
            std::string text((std::istreambuf_iterator<char>(log)),
                             std::istreambuf_iterator<char>());
            out->output = tail_of(text, 4096);
        }
#endif
    };

#if defined(_WIN32)
    std::string command;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) {
            command.push_back(' ');
        }
        command += quote_argument(argv[i]);
    }
    auto widen = [](const std::string& text) {
        if (text.empty()) {
            return std::wstring();
        }
        const int size = MultiByteToWideChar(CP_UTF8, 0, text.c_str(),
                                             static_cast<int>(text.size()), nullptr, 0);
        std::wstring wide(static_cast<std::size_t>(size), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), wide.data(),
                            size);
        return wide;
    };
    const std::wstring wide_command = widen(command);
    const std::wstring wide_log = widen(fs_utf8::from_path(log_path));

    SECURITY_ATTRIBUTES attributes{};
    attributes.nLength = sizeof(attributes);
    attributes.bInheritHandle = TRUE;
    // DELETE access plus FILE_FLAG_DELETE_ON_CLOSE means the log disappears when
    // the last handle goes away - including when this process is killed, which
    // would otherwise leave joc_process_*.log litter in the temp directory.
    HANDLE log_handle = CreateFileW(
        wide_log.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, &attributes, CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (log_handle == INVALID_HANDLE_VALUE) {
        return Status::fail(JOC_ERR_IO, stage::kOutput, "cannot create the process log file");
    }

    // A delete-on-close file cannot be reopened by name (it is delete-pending), so
    // the child's output is read back through the handle it wrote to.
    auto read_log_handle = [&]() {
        LARGE_INTEGER start{};
        start.QuadPart = 0;
        if (!SetFilePointerEx(log_handle, start, nullptr, FILE_BEGIN)) {
            return;
        }
        std::string text;
        char buffer[1024];
        DWORD count = 0;
        while (ReadFile(log_handle, buffer, sizeof(buffer), &count, nullptr) && count > 0) {
            text.append(buffer, count);
        }
        out->output = tail_of(text, 4096);
    };

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = log_handle;
    startup.hStdError = log_handle;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};

    std::vector<wchar_t> mutable_command(wide_command.begin(), wide_command.end());
    mutable_command.push_back(L'\0');
    const BOOL started = CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    if (!started) {
        CloseHandle(log_handle);  // delete-on-close removes the file
        return Status::fail(JOC_ERR_LIBRARY_MISSING, stage::kOutput,
                            "cannot start " + argv[0] + " (is it on PATH?)");
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 0;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    // Read the log before the delete-on-close handle goes away.
    read_log_handle();
    CloseHandle(log_handle);
    out->exit_code = static_cast<std::uint32_t>(exit_code);
#else
    std::string command;
    for (std::size_t i = 0; i < argv.size(); ++i) {
        if (i != 0) {
            command.push_back(' ');
        }
        command += quote_argument(argv[i]);
    }
    command += " > " + quote_argument(fs_utf8::from_path(log_path)) + " 2>&1";
    const int status = std::system(command.c_str());
    // system() reports a wait status, not the child's exit code.
    out->exit_code = status == -1 ? 127u
                     : WIFEXITED(status) ? static_cast<std::uint32_t>(WEXITSTATUS(status))
                                         : 128u;
    read_log();
    std::error_code ignored;
    std::filesystem::remove(log_path, ignored);
#endif

    if (out->exit_code != 0) {
        return Status::fail(JOC_ERR_INPUT_FORMAT, stage::kOutput,
                            argv[0] + " failed with exit code " + std::to_string(out->exit_code) +
                                (out->output.empty() ? "" : ": " + tail_of(out->output, 400)));
    }
    return Status::success();
}

}  // namespace joc::io
