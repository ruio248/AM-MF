#!/usr/bin/env bash
# Install only the generic EGL loader into a user-owned prefix when a shared
# server image ships an NVIDIA EGL vendor library but omits libEGL.so.1.
set -euo pipefail

prefix="${AM_MF_EGL_PREFIX:-$HOME/.local/am-mf-egl}"
loader="$prefix/usr/lib/x86_64-linux-gnu/libEGL.so.1"

if [[ -e /usr/lib/x86_64-linux-gnu/libEGL.so.1 ]]; then
  printf 'system EGL loader already present: %s\n' /usr/lib/x86_64-linux-gnu/libEGL.so.1
  exit 0
fi

if [[ -e "$loader" ]]; then
  printf 'user EGL loader already present: %s\n' "$loader"
  exit 0
fi

if ! command -v apt-get >/dev/null || ! command -v dpkg-deb >/dev/null; then
  echo "apt-get and dpkg-deb are required to install a user-local libegl1 package." >&2
  exit 1
fi

mkdir -p "$prefix/packages"
package_dir="$(mktemp -d "$prefix/packages/libegl1.XXXXXX")"

(
  cd "$package_dir"
  apt-get download libegl1
)

package_file="$(find "$package_dir" -maxdepth 1 -type f -name 'libegl1_*.deb' -print -quit)"
if [[ -z "$package_file" ]]; then
  echo "apt-get download did not produce a libegl1 package." >&2
  exit 1
fi

dpkg-deb -x "$package_file" "$prefix"

if [[ ! -e "$loader" ]]; then
  echo "libegl1 extraction completed but did not provide $loader." >&2
  exit 1
fi

printf 'installed user EGL loader: %s\n' "$loader"
