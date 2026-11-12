#!/usr/bin/env bash
# Checks that the build runs on macOS 10.11:
# - every Mach-O asks for macOS 10.11 or older (minos)
# - the binaries we build (bootloader, CTranslate2, libc++) don't import, as
#   "weak", functions of the system that 10.11 doesn't have (they would be
#   NULL there). Python, numpy... check those at runtime (__builtin_available).
# - nothing of ours uses the libc++ of the system
# Usage: check-macho.sh <folder>
set -euo pipefail
DIR="$1"
fail=0
NEWER=' _(clock_gettime|clock_gettime_nsec_np|clock_getres|clock_settime|getentropy|os_unfair_lock_[a-z_]*|aligned_alloc|utimensat|futimens|mkostemp|mkostemps|timingsafe_bcmp|os_log_[a-z_]*|os_signpost_[a-z_]*) '
while IFS= read -r -d '' f; do
  file -b "$f" | grep -q 'Mach-O' || continue
  min=$(otool -l "$f" | awk '/LC_VERSION_MIN_MACOSX/{v=1} v&&/ version/{print $2; exit} /LC_BUILD_VERSION/{b=1} b&&/minos/{print $2; exit}')
  name="${f#$DIR/}"
  if [ -n "$min" ] && [ "$(printf '%s\n10.11\n' "$min" | sort -V | tail -1)" != "10.11" ]; then
    echo "ERROR: $name needs macOS $min"; fail=1
  fi
  # functions of macOS 10.12+ imported as normal (not weak): dyld refuses to
  # load the file on 10.11 (Rust 1.74+, for example, uses some of them)
  newer=$(nm -m "$f" 2>/dev/null | grep '(undefined) external' | grep -E "$NEWER" || true)
  if [ -n "$newer" ]; then echo "ERROR: $name needs functions of macOS 10.12+:"; echo "$newer"; fail=1; fi
  case "$(basename "$f")" in
    whisper-faster|libctranslate2*|_ext*.so|libc++*)
      weak=$(nm -m "$f" | grep '(undefined) weak external' | grep -v '(from libc++' | grep -v '(from libc++abi' || true)
      if [ -n "$weak" ]; then echo "ERROR: $name imports functions newer than macOS 10.11:"; echo "$weak"; fail=1; fi
      if otool -L "$f" | grep -q '/usr/lib/libc++'; then echo "ERROR: $name uses the libc++ of the system"; fail=1; fi
      echo "$name: minos=$min (checked)";;
  esac
done < <(find "$DIR" -type f -print0)
exit $fail
