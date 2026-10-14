#include "foundation/fs_utf8.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace joc::fs_utf8 {

namespace fs = std::filesystem;

namespace {

#if defined(_WIN32)
constexpr char kPathListSeparator = ';';
#else
constexpr char kPathListSeparator = ':';
#endif

// PATH (and PATHEXT) are separator-delimited lists.  An empty entry is kept: on both
// platforms it means the current directory.
std::vector<std::string> split_list(const char* value, char separator) {
    std::vector<std::string> parts;
    if (value == nullptr) {
        return parts;
    }
    std::string current;
    for (const char c : std::string(value)) {
        if (c == separator) {
            parts.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    parts.push_back(current);
    return parts;
}

bool is_runnable_file(const std::string& utf8_path) {
    std::error_code error;
    if (!fs::is_regular_file(to_path(utf8_path), error) || error) {
        return false;
    }
#if defined(_WIN32)
    return true;
#else
    // Existence is not enough: a file without the execute bit cannot be started, so
    // deciding here names the real problem instead of a later "exit code 126".
    return ::access(utf8_path.c_str(), X_OK) == 0;
#endif
}

bool carries_directory(const std::string& utf8_path) {
#if defined(_WIN32)
    return utf8_path.find_first_of("/\\") != std::string::npos;
#else
    return utf8_path.find('/') != std::string::npos;
#endif
}

#if defined(_WIN32)
// A bare name only resolves through PATHEXT on Windows.  The environment is the
// authority; the classic four are the fallback when it is unset.
std::vector<std::string> executable_extensions() {
    std::vector<std::string> extensions = split_list(std::getenv("PATHEXT"), ';');
    if (extensions.empty()) {
        extensions = {".COM", ".EXE", ".BAT", ".CMD"};
    }
    return extensions;
}
#endif

}  // namespace

fs::path to_path(const std::string& utf8) {
    return fs::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string from_path(const fs::path& path) {
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char*>(text.data()), text.size());
}

std::FILE* fopen(const std::string& utf8_path, const char* mode) {
#if defined(_WIN32)
    const std::wstring wide_mode(mode, mode + std::strlen(mode));
    return ::_wfopen(to_path(utf8_path).c_str(), wide_mode.c_str());
#else
    return std::fopen(utf8_path.c_str(), mode);
#endif
}

std::FILE* fopen_spool(const std::string& utf8_path) {
#if defined(_WIN32)
    // Delete-on-close handed to the CRT: if the process is killed the file goes with
    // it, which is what stops an aborted run from leaving hundreds of gigabytes.
    HANDLE handle = ::CreateFileW(
        to_path(utf8_path).c_str(), GENERIC_READ | GENERIC_WRITE | DELETE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_ALWAYS,
        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    const int descriptor = ::_open_osfhandle(reinterpret_cast<std::intptr_t>(handle), 0);
    if (descriptor == -1) {
        ::CloseHandle(handle);
        return nullptr;
    }
    return ::_fdopen(descriptor, "wb+");
#else
    return std::fopen(utf8_path.c_str(), "wb+");
#endif
}

int remove(const std::string& utf8_path) {
#if defined(_WIN32)
    return ::_wremove(to_path(utf8_path).c_str());
#else
    return std::remove(utf8_path.c_str());
#endif
}

bool exists(const std::string& utf8_path) {
    std::error_code error;
    return fs::exists(to_path(utf8_path), error);
}

bool is_directory(const std::string& utf8_path) {
    std::error_code error;
    return fs::is_directory(to_path(utf8_path), error);
}

std::uintmax_t file_size(const std::string& utf8_path, std::error_code& error) {
    return fs::file_size(to_path(utf8_path), error);
}

std::string temp_directory() {
    std::error_code error;
    const fs::path directory = fs::temp_directory_path(error);
    return error ? std::string(".") : from_path(directory);
}

std::string executable_path() {
#if defined(_WIN32)
    std::vector<wchar_t> buffer(MAX_PATH);
    while (true) {
        const DWORD written =
            ::GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (written == 0) {
            return std::string();
        }
        if (written < buffer.size()) {
            return from_path(fs::path(std::wstring(buffer.data(), written)));
        }
        buffer.resize(buffer.size() * 2u);
    }
#elif defined(__linux__)
    std::vector<char> buffer(4096u, '\0');
    const ssize_t written = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1u);
    return written > 0 ? std::string(buffer.data(), static_cast<std::size_t>(written))
                       : std::string();
#else
    return std::string();
#endif
}

std::string find_executable(const std::string& reference) {
    if (reference.empty()) {
        return std::string();
    }
    if (carries_directory(reference)) {
        // A reference that already names a location is the caller's own decision:
        // keep it verbatim rather than searching for a same-named program elsewhere.
        return is_runnable_file(reference) ? reference : std::string();
    }
#if defined(_WIN32)
    const bool named_with_extension = to_path(reference).has_extension();
#endif
    for (const std::string& entry : split_list(std::getenv("PATH"), kPathListSeparator)) {
        const fs::path directory = to_path(entry.empty() ? std::string(".") : entry);
        const std::string candidate = from_path(directory / to_path(reference));
        if (is_runnable_file(candidate)) {
            return candidate;
        }
#if defined(_WIN32)
        if (!named_with_extension) {
            for (const std::string& extension : executable_extensions()) {
                if (is_runnable_file(candidate + extension)) {
                    return candidate + extension;
                }
            }
        }
#endif
    }
    return std::string();
}

std::ifstream open_input(const std::string& utf8_path) {
    return std::ifstream(to_path(utf8_path), std::ios::binary);
}

std::ofstream open_output(const std::string& utf8_path) {
    return std::ofstream(to_path(utf8_path), std::ios::binary);
}

std::vector<std::string> command_line_arguments(int argc, char** argv) {
#if defined(_WIN32)
    (void)argc;
    (void)argv;
    int count = 0;
    LPWSTR* wide = ::CommandLineToArgvW(::GetCommandLineW(), &count);
    std::vector<std::string> arguments;
    if (wide == nullptr) {
        return arguments;
    }
    arguments.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        const std::wstring_view text(wide[index]);
        const int size = ::WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                               static_cast<int>(text.size()), nullptr, 0, nullptr,
                                               nullptr);
        std::string utf8(static_cast<std::size_t>(size), '\0');
        if (size > 0) {
            ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                                  utf8.data(), size, nullptr, nullptr);
        }
        arguments.push_back(std::move(utf8));
    }
    ::LocalFree(wide);
    return arguments;
#else
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }
    return arguments;
#endif
}

void configure_console() {
#if defined(_WIN32)
    ::SetConsoleOutputCP(CP_UTF8);
#endif
}

}  // namespace joc::fs_utf8
