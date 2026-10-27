/* Product version of joc_core and joc_cli.
 *
 * This file is the single source of truth: CMakeLists.txt reads these numbers for
 * project(VERSION ...), joc_version_string() returns JOC_VERSION_STRING, and the
 * release workflow refuses a tag that does not match it.  Bump the three numbers
 * and the string together, then tag the commit with v<JOC_VERSION_STRING>.
 */
#ifndef JOC_VERSION_H
#define JOC_VERSION_H

#define JOC_VERSION_MAJOR 1
#define JOC_VERSION_MINOR 1
#define JOC_VERSION_PATCH 0
#define JOC_VERSION_STRING "1.1.0"

#endif /* JOC_VERSION_H */
