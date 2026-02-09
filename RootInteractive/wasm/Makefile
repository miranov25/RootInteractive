# Phase 0.1.D — WASM Compilation
# Requires: emsdk installed and activated (source emsdk_env.sh)
#
# Pin emscripten version for reproducibility:
#   emsdk install 3.1.51
#   emsdk activate 3.1.51

SRC = src/wasm_functions.cpp
OUT = build/functions.wasm

# -O2: standard optimization, no FMA. DO NOT use -O3 or -ffast-math.
EMFLAGS = -O2 -s STANDALONE_WASM --no-entry \
          -s EXPORTED_FUNCTIONS="['_fun1','_fun2','_fun3','_fun4','_fun5',\
          '_fun1_v','_fun2_v','_fun3_v','_fun4_v','_fun5_v','_malloc','_free']" \
          -s EXPORTED_RUNTIME_METHODS="[]" \
          -s ALLOW_MEMORY_GROWTH=1 \
          -s INITIAL_MEMORY=1048576

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p build
	em++ $(EMFLAGS) $(SRC) -o $(OUT)
	@echo "Built: $(OUT) ($$(wc -c < $(OUT)) bytes)"

clean:
	rm -f build/functions.wasm
