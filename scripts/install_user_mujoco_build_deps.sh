#!/usr/bin/env bash
# Install the OSMesa headers/runtime used by mujoco_py into a user-writable
# prefix.  This deliberately uses apt download + dpkg-deb extraction rather
# than sudo apt install, so it never alters the host operating system.
set -euo pipefail

prefix="${AM_MF_MUJOCO_BUILD_DEPS:-$HOME/.local/am-mf-mujoco-build-deps}"
header="$prefix/usr/include/GL/osmesa.h"
runtime="$prefix/usr/lib/x86_64-linux-gnu/libOSMesa.so.8"

if [[ -f "$header" && -e "$runtime" ]]; then
  echo "User-local MuJoCo build dependencies already available at $prefix"
  exit 0
fi

command -v apt-get >/dev/null || {
  echo "apt-get is required to fetch user-local OSMesa packages." >&2
  exit 1
}
command -v dpkg-deb >/dev/null || {
  echo "dpkg-deb is required to unpack user-local OSMesa packages." >&2
  exit 1
}

mkdir -p "$prefix"
stage_dir="$(mktemp -d "$prefix/.mujoco-debs.XXXXXX")"
cleanup() {
  rm -rf "$stage_dir"
}
trap cleanup EXIT

(
  cd "$stage_dir"
  # Together these provide GL/osmesa.h, GL/gl.h, the OSMesa runtime, and the
  # link-time symlinks on Ubuntu 22.04.  Extracting into the prefix avoids
  # changing the host package database or requiring administrator access.
  apt-get download libosmesa6 libosmesa6-dev libgl-dev libglx-dev mesa-common-dev
  for deb in ./*.deb; do
    dpkg-deb -x "$deb" "$prefix"
  done
)

test -f "$header"
test -e "$runtime"
echo "Installed user-local MuJoCo build dependencies at $prefix"
