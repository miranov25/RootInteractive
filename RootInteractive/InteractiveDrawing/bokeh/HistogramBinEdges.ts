/**
 * Canonical uniform-bin edges and exact represented-edge classification.
 *
 * RI-HIST-BINEDGE-20261115-01.  Binary64 edge construction follows NumPy
 * linspace(xmin, xmax, nbins + 1) for ordinary finite nondegenerate ranges:
 * compute step once; then edges[i] = xmin + i*step; pin endpoints.
 *
 * This routine does NOT apply a tolerance.  A representable value directly
 * below an edge must remain in its left bin.
 */
export function uniformBinEdges(xmin: number, xmax: number, nbins: number): Float64Array {
  if (!Number.isSafeInteger(nbins) || nbins < 1) {
    throw new RangeError("Histogram nbins must be a positive safe integer")
  }
  const width = xmax - xmin
  if (!Number.isFinite(xmin) || !Number.isFinite(xmax) || !Number.isFinite(width) || !(width > 0)) {
    throw new RangeError("Histogram range must be finite, ordered, and have finite width")
  }
  const step = width / nbins
  if (!(step > 0)) {
    throw new RangeError("Histogram step must be representable")
  }
  const edges = new Float64Array(nbins + 1)
  for (let i = 0; i <= nbins; i++) {
    edges[i] = xmin + i * step
  }
  edges[0] = xmin
  edges[nbins] = xmax
  for (let i = 1; i <= nbins; i++) {
    if (!(edges[i] > edges[i - 1])) {
      throw new RangeError("Histogram bin edges are not strictly increasing in binary64")
    }
  }
  return edges
}

/** Classify an already finite, in-range value; xmax belongs to the last bin.
 *
 * Use the legacy fast arithmetic to estimate, then correct the estimate by
 * comparing the *represented* bin edges.  No epsilons and no 32-bit coercion.
 * Binary search is reserved for pathological estimates more than one bin off.
 */
export function uniformInRangeBin(
  value: number, scale: number, origin: number, edges: Float64Array
): number {
  const nbins = edges.length - 1
  if (value === edges[nbins]) return nbins - 1
  let guess = Math.trunc(value * scale + origin)
  if (!Number.isFinite(guess)) guess = 0
  guess = Math.max(0, Math.min(nbins - 1, guess))

  if (value >= edges[guess] && value < edges[guess + 1]) return guess
  if (guess > 0 && value >= edges[guess - 1] && value < edges[guess]) return guess - 1
  if (guess + 1 < nbins && value >= edges[guess + 1] && value < edges[guess + 2]) return guess + 1

  // Rare fallback: locate the last edge <= value, including exact equality.
  let low = 0
  let high = nbins
  while (low < high) {
    const mid = Math.floor((low + high) / 2)
    if (value < edges[mid]) high = mid
    else low = mid + 1
  }
  return Math.max(0, Math.min(nbins - 1, low - 1))
}
