#!/usr/bin/env bash
# Build + run the model-free adaptive-width self-test with plain clang++/g++.
# No cmake, no llama.cpp libraries, no model required: the AdaptiveWidth policy is
# header-only integer logic and is verified in isolation here.
#
#   ./build_selftest.sh          # compile and run
#
# Exit 0 == all policy checks pass.
set -euo pipefail
cd "$(dirname "$0")"
CXX="${CXX:-c++}"
echo ">> $CXX -std=c++17 -O2 -Wall -Wextra -I. -I../edgeml-lookup adaptive_selftest.cpp -o adaptive_selftest"
"$CXX" -std=c++17 -O2 -Wall -Wextra -I. -I../edgeml-lookup adaptive_selftest.cpp -o adaptive_selftest
echo ">> ./adaptive_selftest"
./adaptive_selftest
