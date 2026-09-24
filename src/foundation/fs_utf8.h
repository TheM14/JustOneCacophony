#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

// Paths inside this library are always UTF-8, on every platform.  std::filesystem
// stores UTF-16 on Windows and bytes elsewhere, and the narrow CRT uses the ANSI
// code page on Windows, so every path crosses into the OS through this shim: that
// is what makes non-ASCII names (Japanese, Chinese, ...) work.
namespace joc::fs_utf8 {

std::filesystem::path to_path(const std::string& utf8);
std::string from_path(const std::filesystem::path& path);

// File handles and queries take a UTF-8 path: _wfopen on Windows, plain calls
// elsewhere.  Nothing else in the library may call the narrow CRT with a path.
std::FILE* fopen(const std::string& utf8_path, const char* mode);
// Temporary spool handle: on Windows the file is opened delete-on-close, so killing
// the process removes it instead of leaving a multi-gigabyte leftover behind.
std::FILE* fopen_spool(const std::string& utf8_path);
int remove(const std::string& utf8_path);
bool exists(const std::string& utf8_path);
bool is_directory(const std::string& utf8_path);
std::uintmax_t file_size(const std::string& utf8_path, std::error_code& error);
std::string temp_directory();

// The running executable's own path, UTF-8, or empty when the platform cannot
// report it.  Defaults are anchored here so they never depend on the CWD.
std::string executable_path();

// Streams: std::ifstream/ofstream accept a std::filesystem::path, which is the
// portable way to open a UTF-8 path.
std::ifstream open_input(const std::string& utf8_path);
std::ofstream open_output(const std::string& utf8_path);

// Command line arguments as UTF-8.  Windows hands the process UTF-16 and the
// narrow CRT would convert it through the ANSI code page, so the wide command
// line is re-parsed there; on POSIX argv is already bytes in the user's locale.
std::vector<std::string> command_line_arguments(int argc, char** argv);
void configure_console();

}  // namespace joc::fs_utf8
