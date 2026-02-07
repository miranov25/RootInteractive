/**
 * Phase 0.1.D — Convolution functions for WASM benchmark
 *
 * 1D, 2D, 3D convolution with clamp-to-edge boundary handling.
 * Use cases: signal smoothing (1D), histogram smoothing (2D, 3D).
 * Separate from wasm_functions.cpp to keep the invariance test WASM small.
 *
 * Data layout:
 *   1D: data[i]                          (n elements)
 *   2D: data[y * nx + x]                 (ny * nx elements, row-major)
 *   3D: data[(z * ny + y) * nx + x]      (nz * ny * nx elements, row-major)
 *
 * Kernel layout:
 *   1D: kernel[j]                        (klen elements)
 *   2D: kernel[ky * klen + kx]           (klen * klen elements, square)
 *   3D: kernel[(kz * klen + ky) * klen + kx] (klen^3 elements, cube)
 *
 * Compile:
 *   em++ -O2 -s STANDALONE_WASM --no-entry \
 *        -s EXPORTED_FUNCTIONS="['_convolve1d','_convolve2d','_convolve3d','_malloc','_free']" \
 *        -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=1048576 \
 *        wasm_conv.cpp -o wasm_conv.wasm
 */
#include <cstdlib>

extern "C" {

// Helper: clamp index to [0, max-1]
static inline int clamp(int val, int max) {
    if (val < 0) return 0;
    if (val >= max) return max - 1;
    return val;
}

/**
 * 1D convolution with clamp-to-edge boundary.
 */
void convolve1d(const double* data, const double* kernel, double* out, int n, int klen) {
    const int half = (klen - 1) / 2;

    // Boundary: left edge
    for (int i = 0; i < half; i++) {
        double sum = 0.0;
        for (int j = 0; j < klen; j++) {
            sum += data[clamp(i - half + j, n)] * kernel[j];
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
            sum += data[clamp(i - half + j, n)] * kernel[j];
        }
        out[i] = sum;
    }
}

/**
 * 2D convolution with clamp-to-edge boundary.
 * Data: row-major ny x nx. Kernel: square klen x klen.
 */
void convolve2d(const double* data, const double* kernel, double* out,
                int nx, int ny, int klen) {
    const int half = (klen - 1) / 2;

    for (int y = 0; y < ny; y++) {
        const int y_inner = (y >= half && y < ny - half) ? 1 : 0;

        for (int x = 0; x < nx; x++) {
            double sum = 0.0;

            if (y_inner && x >= half && x < nx - half) {
                // Inner region: no boundary checks
                for (int ky = 0; ky < klen; ky++) {
                    const int row = (y - half + ky) * nx;
                    const int krow = ky * klen;
                    for (int kx = 0; kx < klen; kx++) {
                        sum += data[row + (x - half + kx)] * kernel[krow + kx];
                    }
                }
            } else {
                // Boundary region: clamp
                for (int ky = 0; ky < klen; ky++) {
                    const int iy = clamp(y - half + ky, ny);
                    const int krow = ky * klen;
                    for (int kx = 0; kx < klen; kx++) {
                        const int ix = clamp(x - half + kx, nx);
                        sum += data[iy * nx + ix] * kernel[krow + kx];
                    }
                }
            }
            out[y * nx + x] = sum;
        }
    }
}

/**
 * 3D convolution with clamp-to-edge boundary.
 * Data: row-major nz x ny x nx. Kernel: cube klen x klen x klen.
 */
void convolve3d(const double* data, const double* kernel, double* out,
                int nx, int ny, int nz, int klen) {
    const int half = (klen - 1) / 2;
    const int slice = ny * nx;

    for (int z = 0; z < nz; z++) {
        const int z_inner = (z >= half && z < nz - half) ? 1 : 0;

        for (int y = 0; y < ny; y++) {
            const int y_inner = (y >= half && y < ny - half) ? 1 : 0;

            for (int x = 0; x < nx; x++) {
                double sum = 0.0;

                if (z_inner && y_inner && x >= half && x < nx - half) {
                    // Inner region: no boundary checks
                    for (int kz = 0; kz < klen; kz++) {
                        const int iz = z - half + kz;
                        for (int ky = 0; ky < klen; ky++) {
                            const int iy = y - half + ky;
                            const int base_data = iz * slice + iy * nx + (x - half);
                            const int base_kern = (kz * klen + ky) * klen;
                            for (int kx = 0; kx < klen; kx++) {
                                sum += data[base_data + kx] * kernel[base_kern + kx];
                            }
                        }
                    }
                } else {
                    // Boundary region: clamp
                    for (int kz = 0; kz < klen; kz++) {
                        const int iz = clamp(z - half + kz, nz);
                        for (int ky = 0; ky < klen; ky++) {
                            const int iy = clamp(y - half + ky, ny);
                            const int base_kern = (kz * klen + ky) * klen;
                            for (int kx = 0; kx < klen; kx++) {
                                const int ix = clamp(x - half + kx, nx);
                                sum += data[iz * slice + iy * nx + ix] * kernel[base_kern + kx];
                            }
                        }
                    }
                }
                out[z * slice + y * nx + x] = sum;
            }
        }
    }
}

} // extern "C"
