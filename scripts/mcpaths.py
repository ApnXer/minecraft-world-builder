"""Locate a Minecraft client jar (the source of truth for block ids, models and textures)."""
import glob
import os
import sys


def find_jar(version="1.21.11"):
    """MC_JAR env var wins; otherwise search the default .minecraft locations for versions/<version>/<version>.jar."""
    env = os.environ.get("MC_JAR")
    if env and os.path.isfile(env):
        return env
    roots = []
    if sys.platform.startswith("win"):
        roots.append(os.path.join(os.environ.get("APPDATA", ""), ".minecraft"))
    elif sys.platform == "darwin":
        roots.append(os.path.expanduser("~/Library/Application Support/minecraft"))
    roots.append(os.path.expanduser("~/.minecraft"))
    for root in roots:
        p = os.path.join(root, "versions", version, f"{version}.jar")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Minecraft {version} jar not found. Start that version once in the launcher or set MC_JAR=/path/to/{version}.jar")
