# Shared by the launchers: run from the repository root with the active environment's Python, and hide the noisy
# macOS AVFoundation and deprecation warnings that LeRobot's video stack prints.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
quiet() { grep --line-buffered -v -E "objc\[|Warning|NOTE: Redirects|deprecated|Svt\[|SVT|Map:|moov atom|^\s*$" >&2; }
