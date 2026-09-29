# Reserved: optional compiled kernel

Kept for a narrowly scoped compiled extension, used only if a future performance gate
(implementation plan, section 4) fails. At the time of this implementation the gate was met with
NumPy/SciPy, pyamg, and an optional Numba streamline kernel; see
`docs/validation_and_performance.md`. No C++ code is needed now.
