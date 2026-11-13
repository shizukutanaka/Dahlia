#!/bin/sh
# Verify that every place carrying the app version agrees (W55).
# The single source of truth is apps/studio/src-tauri/tauri.conf.json;
# the app shows the studio crate's CARGO_PKG_VERSION in its footer.
set -eu
cd "$(dirname "$0")/.."

want=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' apps/studio/src-tauri/tauri.conf.json | head -1)
fail=0
check() {
    got=$(sed -n "s/.*$1.*\"\([0-9][^\"]*\)\".*/\1/p" "$2" | head -1)
    if [ "$got" != "$want" ]; then
        echo "version mismatch: $2 has '$got', expected '$want'" >&2
        fail=1
    fi
}
check 'version' apps/studio/src-tauri/Cargo.toml
check 'version' core/Cargo.toml
check 'version' apps/studio/package.json

lock_check() {
    # version of one named package inside a Cargo.lock
    got=$(awk -v name="$2" '
        $0 == "name = \"" name "\"" { found = 1; next }
        found && /^version = / { gsub(/[" ]/, "", $3); print $3; exit }
    ' "$1")
    if [ "$got" != "$want" ]; then
        echo "version mismatch: $1 package $2 has '$got', expected '$want'" >&2
        fail=1
    fi
}
lock_check core/Cargo.lock dahlia-core
lock_check apps/studio/src-tauri/Cargo.lock dahlia-core
lock_check apps/studio/src-tauri/Cargo.lock studio

if [ "$fail" -ne 0 ]; then
    exit 1
fi
echo "version OK: $want"
