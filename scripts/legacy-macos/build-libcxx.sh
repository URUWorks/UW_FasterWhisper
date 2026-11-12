#!/usr/bin/env bash
# libc++ / libc++abi (LLVM) for macOS 10.11 x86_64, as dylibs (@rpath).
# The system libc++ of 10.11 lacks parts of C++17 that CTranslate2 uses
# (std::optional, std::variant... are 10.13/10.14 in the SDK).
# Usage: build-libcxx.sh <LLVM version> <install prefix>
set -euo pipefail
VER="$1"; PREFIX="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
export MACOSX_DEPLOYMENT_TARGET=10.11
WORK="$(mktemp -d)"
cd "$WORK"
curl -fsSL -o llvm.tar.xz "https://github.com/llvm/llvm-project/releases/download/llvmorg-$VER/llvm-project-$VER.src.tar.xz"
tar xf llvm.tar.xz
# libc++ uses utimensat (10.13+): below 10.13, its alternative with utimes
sed -i '' 's/^#if defined(UTIME_OMIT)$/#if defined(UTIME_OMIT) \&\& !(defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) \&\& __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < 101300)/' "llvm-project-$VER.src/libcxx/src/filesystem/time_utils.h"
grep -n 'defined(UTIME_OMIT)' "llvm-project-$VER.src/libcxx/src/filesystem/time_utils.h"
# our clock_gettime (10.12+ in the system), inside each dylib
mkdir -p "$PREFIX/lib"
clang -c -O2 -arch x86_64 -mmacosx-version-min=10.11 "$HERE/clock_gettime-compat.c" -o "$PREFIX/lib/clock_gettime-compat.o"
cmake -S "llvm-project-$VER.src/runtimes" -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=10.11 \
  -DCMAKE_SHARED_LINKER_FLAGS="$PREFIX/lib/clock_gettime-compat.o" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
  -DLIBCXXABI_USE_LLVM_UNWINDER=OFF \
  -DLIBCXX_ENABLE_SHARED=ON -DLIBCXX_ENABLE_STATIC=OFF \
  -DLIBCXXABI_ENABLE_SHARED=ON -DLIBCXXABI_ENABLE_STATIC=OFF \
  -DLIBCXX_INCLUDE_TESTS=OFF -DLIBCXX_INCLUDE_BENCHMARKS=OFF -DLIBCXXABI_INCLUDE_TESTS=OFF \
  -DLIBCXX_INSTALL_LIBRARY_DIR=lib -DLIBCXXABI_INSTALL_LIBRARY_DIR=lib \
  -DCMAKE_INSTALL_NAME_DIR=@rpath \
  -DLIBCXX_ENABLE_VENDOR_AVAILABILITY_ANNOTATIONS=OFF
cmake --build build
cmake --install build
otool -L "$PREFIX/lib/libc++.1.dylib"
rm -rf "$WORK"
