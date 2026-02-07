/**
 * Phase 0.1.D — Convolution functions for WASM benchmark
 *
 * 1D convolution with clamp-to-edge boundary handling.
 * Separate from wasm_functions.cpp to keep the invariance test WASM small.
 *
 * Compile:
 *   em++ -O2 -s STANDALONE_WASM --no-entry \
 *        -s EXPORTED_FUNCTIONS="['_convolve1d','_malloc','_free']" \
 *        -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=1048576 \
 *        wasm_conv.cpp -o wasm_conv.wasm
 */
#include <cstdlib>

extern "C" {

/**
 * 1D convolution with clamp-to-edge boundary.
 *
 * @param data   Input array (n elements)
 * @param kernel Convolution kernel (klen elements, must be odd)
 * @param out    Output array (n elements)
 * @param n      Data length
 * @param klen   Kernel length
 */
void convolve1d(const double* data, const double* kernel, double* out, int n, int klen) {
    const int half = (klen - 1) / 2;

    // Boundary: left edge
    for (int i = 0; i < half; i++) {
        double sum = 0.0;
        for (int j = 0; j < klen; j++) {
            int idx = i - half + j;
            if (idx < 0) idx = 0;
            sum += data[idx] * kernel[j];
        }
        out[i] = sum;
    }

    // Inner region: no boundary checks
    for (int i = half; i < n - half; i++) {
        double sum = 0.0;
        for (int j = 0; j < klen; j++) {
            sum += data[i - half + j] * kernel[j];
        }
        out[i] = sum;
    }

    // Boundary: right edge
    for (int i = n - half; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < klen; j++) {
            int idx = i - half + j;
            if (idx >= n) idx = n - 1;
            sum += data[idx] * kernel[j];
        }
        out[i] = sum;
    }
}

} // extern "C"
