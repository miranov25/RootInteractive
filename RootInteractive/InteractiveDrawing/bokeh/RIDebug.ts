// PHASE_5 group 6 (the Architect's answer G4 b): the timing and debug output of the models is off by default.
// For performance work, type `window.RootInteractiveDebug = true` in the browser console; the models then print
// their timings (console.time / console.timeEnd) and debug lines again.
export function ri_debug(): boolean {
  const g = globalThis as any
  return g.RootInteractiveDebug === true || (g.window != null && g.window.RootInteractiveDebug === true)
}
