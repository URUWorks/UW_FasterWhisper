#!/usr/bin/env python3
#
# Monthly update (.github/workflows/update.yml): requirements.txt to the
# newest versions on PyPI, and the next version number (1.0.0 -> 1.0.1) in
# whisper_faster.py.
#
# - The macOS lines (sys_platform == "darwin") stay as they are: the newer
#   onnxruntime/numpy builds need a recent macOS, these work from 11.
# - CTranslate2 stays in its major version: the GPU zip has the CUDA 12 /
#   cuDNN 9 libraries that CTranslate2 4.x needs.
# - Lines without "==" (numpy<2.4) are not touched.
#
# Writes changed, tag and summary to $GITHUB_OUTPUT, and the commit message
# to $RUNNER_TEMP/commit.txt.
#

import json
import os
import re
import subprocess
import urllib.request

from packaging.version import InvalidVersion, Version

REQUIREMENTS = "requirements.txt"
SOURCE = "whisper_faster.py"
SAME_MAJOR = {"ctranslate2"}

PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.\-]+)==(?P<ver>[^\s;]+)(?P<rest>\s*;.*)?$")
DARWIN = re.compile(r"sys_platform\s*==\s*[\"']darwin[\"']")
VERSION = re.compile(r'^__version__ = "[^"]*"', re.M)


def newest(name, major=None):
    """The newest stable (not yanked) version of a PyPI package."""
    with urllib.request.urlopen("https://pypi.org/pypi/%s/json" % name, timeout=60) as r:
        releases = json.load(r)["releases"]
    best = None
    for v, files in releases.items():
        try:
            ver = Version(v)
        except InvalidVersion:
            continue
        if ver.is_prerelease or ver.is_devrelease:
            continue
        if not files or all(f.get("yanked") for f in files):
            continue
        if major is not None and ver.major != major:
            continue
        if best is None or ver > best:
            best = ver
    return best


def next_version():
    tags = subprocess.run(["git", "tag", "--list", "v*"], capture_output=True,
                          text=True, check=True).stdout.split()
    versions = []
    for t in tags:
        try:
            versions.append(Version(t[1:]))
        except InvalidVersion:
            pass
    last = max(versions) if versions else Version("1.0.0")
    return "%d.%d.%d" % (last.major, last.minor, last.micro + 1)


def output(key, value):
    print("%s=%s" % (key, value))
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("%s=%s\n" % (key, value))


def main():
    with open(REQUIREMENTS, encoding="utf-8", newline="") as f:
        lines = f.read().split("\n")

    changes = []
    for i, line in enumerate(lines):
        m = PIN.match(line.strip())
        if not m or DARWIN.search(m.group("rest") or ""):
            continue
        name, old = m.group("name"), Version(m.group("ver"))
        new = newest(name, old.major if name.lower() in SAME_MAJOR else None)
        if new and new > old:
            lines[i] = "%s==%s%s" % (name, new, m.group("rest") or "")
            changes.append("%s %s -> %s" % (name, old, new))

    if not changes:
        print("No new versions.")
        output("changed", "false")
        return

    version = next_version()
    with open(REQUIREMENTS, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines))
    with open(SOURCE, encoding="utf-8", newline="") as f:
        source = f.read()
    with open(SOURCE, "w", encoding="utf-8", newline="") as f:
        f.write(VERSION.sub('__version__ = "%s"' % version, source, count=1))

    summary = ", ".join(changes)
    with open(os.path.join(os.environ.get("RUNNER_TEMP", "."), "commit.txt"), "w", encoding="utf-8") as f:
        f.write("UW_FasterWhisper %s\n\n%s\n" % (version, "\n".join("- " + c for c in changes)))

    output("changed", "true")
    output("tag", "v" + version)
    output("summary", "Updated: " + summary)


if __name__ == "__main__":
    main()
