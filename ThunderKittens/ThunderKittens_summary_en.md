# ThunderKittens: Simple, Fast, and Adorable AI Kernels

**Paper**: [arXiv:2410.20399](https://arxiv.org/abs/2410.20399)
**Authors**: Benjamin F. Spector, Simran Arora, Aaryan Singhal, Daniel Y. Fu, Christopher Ré
**Affiliation**: Stanford University (Hazy Research Group)
**Published**: ICLR 2025
**Code**: [github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)

---

## 1. Problem Statement

Mapping AI architectures to GPU hardware efficiently is a critical bottleneck in AI progress. Despite substantial engineering efforts, hand-written custom CUDA kernels frequently fail to reach their theoretical performance ceilings—even for well-established operations like matrix multiplication and attention. Existing approaches (raw CUDA, CUTLASS, Triton) either demand deep hardware expertise or sacrifice performance due to abstraction overhead. There is a pressing need for a framework that is simultaneously **simple to use** and **high-performance**.

## 2. Key Insight

Modern GPUs are not monolithic "big matrix multiply" machines. They are manycore processors where each core efficiently runs small (~16×16) matrix multiplies via tensor cores. On H100, BF16 tensor cores provide 15–16× the FLOPs of general-purpose BF16/FP32 compute, making tensor core utilization the dominant factor for performance. ThunderKittens is built from this hardware reality upward: **the 16×16 matrix tile is the fundamental unit of computation**.

## 3. Framework Design: Three-Level GPU Hierarchy

ThunderKittens provides abstractions that map directly to the three levels of the GPU memory/compute hierarchy:

### 3.1 Warp Level (Registers)

- **Data Structure**: The 16×16 matrix tile (`kittens::rt`) is the basic unit, stored across the registers of a single warp (32 threads). Tiles can be composed into larger tiles (e.g., 64×32).
- **Operations**: PyTorch-like operations over tiles—`mma` (matrix multiply-accumulate), `exp`, `cumsum`, `pointwise_multiply`, etc.
- **Layout Management**: TK automatically selects optimal memory layouts (row-major vs. column-major) and performs compile-time layout checking. For example, `mma_AB` requires A in row-major and B in column-major, and TK raises compile-time errors if violated.
- **Data Types**: Register tiles (`rt`), register vectors (`rv`), parameterized by layout, type (bf16, fp16, fp32), and size.

### 3.2 Thread-Block Level (Shared Memory)

- **Shared Tiles**: 2D tensors in shared memory (`kittens::st`), accessible by all warps in a block.
- **Bank Conflict Elimination**: A critical innovation. GPU shared memory suffers from bank conflicts when multiple threads access the same bank simultaneously. TK automatically selects from three specialized "swizzled" layouts (strided at 32, 64, and 128-byte intervals) to minimize conflicts. This eliminates up to 9.6-way bank conflicts observed in competing implementations like FlashAttention-3.
- **LCSF Template (Load-Compute-Store-Finish)**: A general program template for coordinating asynchronous parallel workers within a thread block. Dedicated "loader" warps handle HBM-to-SRAM data movement while "compute" warps perform operations in registers, overlapping memory latency with computation.
- **TMA Integration**: ThunderKittens leverages the Tensor Memory Accelerator (TMA) on Hopper GPUs for asynchronous tile transfers between global and shared memory, with automatic TMA descriptor initialization.

### 3.3 Grid Level (Global Memory)

- **Global Tiles**: Abstractions for global memory operations.
- **Block Launch/Teardown**: TK helps amortize the cost of block initialization and cleanup.
- **TMA Descriptor Management**: TK handles TMA descriptor setup at the grid level, abstracting away one of the most complex aspects of Hopper programming.

## 4. Asynchronous Execution Model

The H100 GPU's asynchronous features allow a single instruction stream to keep multiple hardware units busy simultaneously, reducing the need for high occupancy. TK exploits this through:

- **Warp Specialization**: Different warps within a block perform different roles (loading, computing, storing).
- **Async WGMMA**: Warpgroup-level (4 warps) asynchronous matrix multiply-accumulate instructions on H100.
- **Pipeline Overlapping**: The LCSF template naturally overlaps memory transfers with computation, maximizing hardware utilization.

## 5. Experimental Results

All benchmarks are conducted on an **NVIDIA H100 80GB SXM GPU**.

### 5.1 GEMM (Matrix Multiplication)

| Metric | Result |
|--------|--------|
| Code Size | ~40 lines of TK code |
| Performance | Matches CuBLAS (NVIDIA's hand-optimized library) |
| Significance | Demonstrates TK can match vendor-optimized code with dramatically less effort |

### 5.2 Attention (FlashAttention Comparison)

| Operation | vs. FlashAttention-3 |
|-----------|---------------------|
| Forward Pass | Competitive / matches FA-3 |
| Backward Pass | **10–40% faster** than FA-3 |
| Code Size | ~100 lines |
| vs. FlashAttention-2 | ~30% faster on H100 |

**Root Cause Analysis (via NVIDIA NCU Profiler)**:
- Tensor core utilization: comparable to FA-3
- Issue slot utilization: higher than FA-3
- HBM throughput: higher, with 10% fewer stalled cycles on HBM waits
- Shared memory: **85% fewer stalled cycles** (TK has zero bank conflicts vs. up to 9.6-way conflicts in FA-3)

### 5.3 Mamba-2 (State Space Models)

| Metric | Result |
|--------|--------|
| vs. Triton (prior work) | **>3× faster** |
| Key Advantage | Ease of fusing complex operations in TK |

### 5.4 Linear Attention

| Variant | Speedup vs. Flash Linear Attention |
|---------|-------------------------------------|
| Polynomial-based | Up to **14×** |
| Learned feature maps | Up to **6.5×** |

### 5.5 Long Convolutions (FlashFFTConv)

| Sequence Length | Speedup vs. FlashFFTConv CUDA |
|-----------------|-------------------------------|
| 1024 | **7.9×** |
| 4096 | **4.7×** |
| vs. PyTorch FFT | Up to **8.7×** |

## 6. Code Simplicity

A key contribution is that TK achieves these results with remarkably compact code:

| Kernel | Lines of Code |
|--------|---------------|
| GEMM | ~40 lines |
| Attention (forward) | ~100 lines |
| Mamba-2 | Compact (exact LOC not specified) |

For comparison, FlashAttention-3 and CuBLAS are thousands of lines of highly specialized CUDA code.

## 7. Design Philosophy

1. **Hardware-first**: Built from the silicon up—abstractions reflect what the GPU hardware actually wants.
2. **Tile-centric**: The 16×16 tile is the atom of computation, not individual scalars or vectors.
3. **Implicit optimization**: Bank conflict elimination, layout selection, and TMA descriptor management are handled automatically.
4. **Familiar API**: Operations mirror PyTorch/NumPy semantics, lowering the learning curve for ML researchers.
5. **Compile-time safety**: Layout and type mismatches are caught at compile time, preventing subtle runtime bugs.

## 8. Impact and Adoption

- **ICLR 2025**: Accepted as a conference paper.
- **Industry Adoption**: Used in production by Together AI (inference), Cursor (training for Composer), and Jump Trading.
- **ThunderKittens 2.0**: Released with full Blackwell (B200) GPU support, MXFP8 and NVFP4 precision.
- **Cross-Platform Extensions**:
  - **HipKittens**: Port for AMD GPUs.
  - **ThunderMittens**: Port for Apple Silicon (Metal Shading Language).
  - **ParallelKittens**: Multi-GPU extension with communication primitives.

## 9. Limitations and Considerations

- Currently focused on NVIDIA GPUs (with AMD/Apple ports as separate projects).
- The LCSF template, while general, may not cover all possible kernel patterns.
- Requires understanding of GPU architecture concepts (warps, shared memory, tensor cores) even if the details are abstracted.
- The 16×16 tile size is fixed and may not be optimal for all workloads.

## 10. Conclusion

ThunderKittens demonstrates that a small, carefully designed set of tile-based abstractions can achieve performance matching or exceeding heavily optimized industrial kernels while remaining dramatically simpler to write and maintain. By building from the hardware up and centering on the 16×16 tile as the fundamental compute unit, TK eliminates common performance pitfalls (bank conflicts, layout mismatches, poor async overlap) and makes high-performance GPU programming accessible to ML researchers. The framework represents a significant step toward democratizing GPU kernel development for AI.

---

## References

- Paper: [https://arxiv.org/abs/2410.20399](https://arxiv.org/abs/2410.20399)
- GitHub: [https://github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- Blog (GPUs Go Brrr): [https://hazyresearch.stanford.edu/blog/2024-05-12-tk](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
- Blog (TK on Blackwell): [https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell](https://hazyresearch.stanford.edu/blog/2025-03-15-tk-blackwell)
- OpenReview (ICLR 2025): [https://openreview.net/forum?id=0fJfVOSUra](https://openreview.net/forum?id=0fJfVOSUra)
