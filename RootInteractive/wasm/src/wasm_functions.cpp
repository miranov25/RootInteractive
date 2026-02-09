/**
 * WASM Invariance Test Functions — Phase 0.1.D v1.1
 *
 * Five test functions covering three operation tiers:
 *   Trivial:  fun1 (a+b)/c, fun2 a*b+1, fun4 a>b?c:-c
 *   Complex:  fun3 sqrt(a²+1), fun5 sin(a)*cos(b)+exp(-c)
 *
 * Each function has scalar (single-value) and vector (array) variants.
 * Vector functions take float64 pointers + length; caller manages memory.
 *
 * Compile with emscripten:
 *   em++ -O2 -s STANDALONE_WASM --no-entry \
 *        -s EXPORTED_FUNCTIONS="['_fun1','_fun2','_fun3','_fun4','_fun5',\
 *        '_fun1_v','_fun2_v','_fun3_v','_fun4_v','_fun5_v','_malloc','_free']" \
 *        wasm_functions.cpp -o functions.wasm
 *
 * IMPORTANT: Use -O2, NOT -O3 (may enable FMA, causing ULP differences).
 *            NEVER use -ffast-math (breaks IEEE-754 NaN/Inf semantics).
 */
#include <cmath>
#include <cstdlib>

extern "C" {

// ============================================================
// Scalar functions: operate on single values
// ============================================================

// Tier: Trivial — basic arithmetic (f64.add, f64.div)
double fun1(double a, double b, double c) {
    return (a + b) / c;
}

// Tier: Trivial — multiply-add (f64.mul, f64.add). FMA-sensitive under -O3.
double fun2(double a, double b) {
    return a * b + 1.0;
}

// Tier: Complex — transcendental (f64.sqrt, IEEE-754 correctly-rounded)
double fun3(double a) {
    return sqrt(a * a + 1.0);
}

// Tier: Trivial — conditional branch (f64.gt, f64.neg, if/else)
double fun4(double a, double b, double c) {
    return a > b ? c : -c;
}

// Tier: Complex — multi-transcendental (sin, cos, exp). NOT correctly-rounded
// by IEEE-754; may differ 1-2 ULP between implementations. Tolerance: 1e-12.
double fun5(double a, double b, double c) {
    return sin(a) * cos(b) + exp(-c);
}

// ============================================================
// Vector functions: operate on arrays
// Pointers to Float64 arrays, n = number of elements.
// Caller is responsible for malloc() and free().
// ============================================================

void fun1_v(const double* a, const double* b, const double* c, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (a[i] + b[i]) / c[i];
    }
}

void fun2_v(const double* a, const double* b, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i] + 1.0;
    }
}

void fun3_v(const double* a, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = sqrt(a[i] * a[i] + 1.0);
    }
}

void fun4_v(const double* a, const double* b, const double* c, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] > b[i] ? c[i] : -c[i];
    }
}

void fun5_v(const double* a, const double* b, const double* c, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = sin(a[i]) * cos(b[i]) + exp(-c[i]);
    }
}

} // extern "C"
