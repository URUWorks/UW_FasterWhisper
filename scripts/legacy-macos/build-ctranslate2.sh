#!/usr/bin/env bash
# CTranslate2 (library + Python module) for macOS 10.11 x86_64, with our
# libc++ (build-libcxx.sh), Apple Accelerate (float32) and Ruy (int8).
# The official wheels need macOS 10.13 or 11, and use Intel MKL.
# Usage: build-ctranslate2.sh <version> <libc++ prefix> <python> <wheel folder>
set -euo pipefail
VER="$1"; LCXX="$2"; PY="$3"; OUT="$4"
export MACOSX_DEPLOYMENT_TARGET=10.11
export ARCHFLAGS="-arch x86_64"
WORK="$(mktemp -d)"
cd "$WORK"

git clone -q --depth 1 -b "v$VER" https://github.com/OpenNMT/CTranslate2 ct2
git -C ct2 submodule update -q --init --depth 1 third_party/cpu_features third_party/spdlog
git -C ct2 submodule update -q --init --recursive --depth 1 third_party/ruy
# CTranslate2 fixes 10.13 and its Python module 10.14: we use 10.11
sed -i '' 's/set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)/# the deployment target of the environment/' ct2/CMakeLists.txt
sed -i '' 's/-mmacosx-version-min=10.14/-mmacosx-version-min=10.11/' ct2/python/setup.py
grep -n 'CMAKE_OSX_DEPLOYMENT_TARGET\|deployment target' ct2/CMakeLists.txt
grep -n 'macosx-version-min' ct2/python/setup.py

CXXF="-nostdinc++ -isystem $LCXX/include/c++/v1"
LDF="-nostdlib++ -L$LCXX/lib -lc++ -lc++abi $LCXX/lib/clock_gettime-compat.o -Wl,-rpath,@loader_path -Wl,-rpath,$LCXX/lib"
cmake -S ct2 -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_OSX_DEPLOYMENT_TARGET=10.11 \
  -DCMAKE_INSTALL_PREFIX="$WORK/install" -DCMAKE_INSTALL_NAME_DIR=@rpath \
  -DCMAKE_CXX_FLAGS="$CXXF" -DCMAKE_SHARED_LINKER_FLAGS="$LDF" -DCMAKE_EXE_LINKER_FLAGS="$LDF" \
  -DWITH_MKL=OFF -DWITH_DNNL=OFF -DWITH_ACCELERATE=ON -DWITH_RUY=ON -DOPENMP_RUNTIME=NONE \
  -DENABLE_CPU_DISPATCH=ON -DBUILD_CLI=OFF -DBUILD_TESTS=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build
cmake --install build

# the Python module, with the same libc++
cd ct2/python
"$PY" -m pip install -q -r install_requirements.txt
CTRANSLATE2_ROOT="$WORK/install" \
CFLAGS="$CXXF" CXXFLAGS="$CXXF" LDFLAGS="$LDF" \
  "$PY" -m pip wheel --no-deps --no-build-isolation -w "$WORK/dist" .
# the libraries inside the package, next to _ext (as the official wheels,
# but by hand: delocate doesn't follow the @rpath of our libc++)
"$PY" -m pip install -q wheel
"$PY" -m wheel unpack -d "$WORK/unpacked" "$WORK"/dist/*.whl
PKG=$(ls -d "$WORK"/unpacked/ctranslate2-*/ctranslate2)
CT2LIB=$(otool -D "$WORK"/install/lib/libctranslate2.*.*.dylib | tail -1 | sed 's|@rpath/||')
cp -L "$WORK/install/lib/$CT2LIB" "$LCXX/lib/libc++.1.dylib" "$LCXX/lib/libc++abi.1.dylib" "$PKG/"
chmod u+w "$PKG"/*.dylib
for f in "$PKG"/_ext*.so "$PKG"/*.dylib; do
  # only @loader_path (the folder of the package) to find the others
  for r in $(otool -l "$f" | awk '/LC_RPATH/{getline; getline; print $2}'); do
    install_name_tool -delete_rpath "$r" "$f"
  done
  install_name_tool -add_rpath @loader_path "$f"
  codesign -f -s - "$f"
  echo "== $(basename "$f")"; otool -L "$f" | tail -n +2; otool -l "$f" | grep -A2 LC_RPATH | grep path
done
mkdir -p "$OUT"
"$PY" -m wheel pack -d "$OUT" "$WORK"/unpacked/ctranslate2-*
ls -la "$OUT"
rm -rf "$WORK"
