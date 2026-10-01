# Reserved: optional compiled kernel

Kept for a narrowly scoped compiled extension, used only if a future performance gate
(`docs/validation_and_performance.md`, section 4) fails. At the time of this implementation the
gate was met with NumPy/SciPy, pyamg, and an optional Numba streamline kernel. No C++ code is
needed now.
