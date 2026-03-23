# ThunderKittens 使用案例深度解析

本文档基于 ThunderKittens 仓库中的 `kernels/gemm/educational_h100/` 教程代码，展示从朴素 CUDA 到 TK 高性能内核的渐进式演化过程。

---

## 教程结构总览

仓库提供了 8 级渐进式 GEMM 教程（`level_01` 到 `level_08`），逐步引入优化技术：

| 级别 | 技术 | 核心变化 |
|------|------|----------|
| Level 1 | 朴素 CUDA (FP32) | 每线程计算一个输出元素 |
| Level 2 | 朴素 CUDA (BF16) | 切换到 BF16 数据类型 |
| Level 3 | 共享内存分块 | 手写 shared memory tiling |
| Level 4 | **TK 入门** | 引入 TK 瓦片抽象，单 warp |
| Level 5 | **TK warpgroup** | 使用 warpgroup 级异步 MMA |
| Level 6 | **TK + TMA** | 引入 TMA 异步加载 + 双缓冲 |
| Level 7 | **TK 生产者-消费者** | warp 特化：加载与计算分离 |
| Level 8 | **TK 多瓦片** | 多行多列瓦片 + 完整生产者-消费者 |

---

## Level 3：传统 CUDA 共享内存分块（对照基线）

```cpp
// 传统方式：手动管理共享内存、同步、类型转换
constexpr int BLOCK_SIZE = 32;
__global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    __shared__ __nv_bfloat16 As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    float sum = 0.0f;
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        As[ty][tx] = A[row * N + tile * BLOCK_SIZE + tx];  // 手动加载
        Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
            sum += __bfloat162float(As[ty][k] * Bs[k][tx]); // 逐元素标量乘
        __syncthreads();
    }
    C[row * N + col] = __float2bfloat16(sum);
}
```

**问题**：
- 无法利用 Tensor Core（标量乘法）
- 手动管理 bank conflict
- 代码不可扩展

---

## Level 4：TK 入门 — 第一个 ThunderKittens 内核

这是最关键的转折点，展示了 TK 如何把 GEMM 从手写 CUDA 转变为瓦片化编程。

```cpp
#include "kittens.cuh"
using namespace kittens;

static constexpr int BLOCK_SIZE = 32;     // 瓦片尺寸 32×32
static constexpr int NUM_WORKERS = 1;     // 单 warp
static constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS; // 32 线程

// ① 定义全局内存布局
struct matmul_globals {
    using sub_tile = st_bf<BLOCK_SIZE, BLOCK_SIZE>;   // 共享内存瓦片类型: 32×32 bf16
    using tile_gl = gl<bf16, 1, 1, -1, -1, sub_tile>; // 全局内存布局描述符
    tile_gl A, B, C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    // ② 分配共享内存瓦片
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE>>();

    // ③ 声明寄存器瓦片
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> A_reg;                          // 行优先 A 瓦片
    rt_bf<BLOCK_SIZE, BLOCK_SIZE> B_reg;                          // 行优先 B 瓦片（临时）
    rt_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::rt_layout::col> B_reg_col; // 列优先 B 瓦片（MMA 需要）
    rt_fl<BLOCK_SIZE, BLOCK_SIZE> C_accum;                        // FP32 累加器

    int col = blockIdx.x;
    int row = blockIdx.y;

    // ④ 初始化累加器为零
    kittens::warp::zero(C_accum);

    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {

        // ⑤ 从全局内存加载到共享内存
        kittens::warp::load(As, g.A, {0, 0, row, tile});  // A[row][tile] 瓦片
        kittens::warp::load(Bs, g.B, {0, 0, tile, col});  // B[tile][col] 瓦片
        __syncthreads();

        // ⑥ 从共享内存加载到寄存器
        kittens::warp::load(A_reg, As);   // shared → register
        kittens::warp::load(B_reg, Bs);

        // ⑦ 布局转换：行优先 → 列优先（MMA 要求 B 为列优先）
        kittens::warp::swap_layout(B_reg_col, B_reg);
        __syncthreads();

        // ⑧ Tensor Core 矩阵乘累加: C_accum += A_reg × B_reg_col
        kittens::warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);
        __syncthreads();
    }

    // ⑨ 将结果从寄存器写回全局内存
    kittens::warp::store(g.C, C_accum, {0, 0, row, col});
}

// ⑩ 启动内核
void matmul(bf16* A, bf16* B, bf16* C, size_t N) {
    using tile_gl = matmul_globals::tile_gl;
    tile_gl a_arg{A, nullptr, nullptr, N, N};
    tile_gl b_arg{B, nullptr, nullptr, N, N};
    tile_gl c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, (int)N};

    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 100000;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
}
```

### 核心概念解析

**① 数据类型体系**：
```
st_bf<R,C>  — 共享内存中的 bf16 瓦片 (Shared Tile)
rt_bf<R,C>  — 寄存器中的 bf16 瓦片 (Register Tile)
rt_fl<R,C>  — 寄存器中的 float 瓦片 (用于累加)
gl<T,...>   — 全局内存布局描述符 (Global Layout)
```

**② 内存层次对应**：
```
全局内存 (HBM)  →  gl<bf16, ...>    →  g.A, g.B, g.C
  ↕ warp::load(shared, global, coord)
共享内存 (SRAM)  →  st_bf<R,C>       →  As, Bs
  ↕ warp::load(register, shared)
寄存器           →  rt_bf<R,C>       →  A_reg, B_reg
                    rt_fl<R,C>       →  C_accum
```

**③ 布局转换**：
- `mma_AB` 要求 A 为行优先、B 为列优先
- `swap_layout(B_col, B_row)` 调用 PTX 指令 `movmatrix.sync` 在 warp 内重分布数据
- 如果布局不匹配，**编译时报错**（不会运行后产生错误结果）

---

## Level 5：Warpgroup 级异步 MMA

从单 warp 扩展到 warpgroup（4 个 warp = 128 线程），利用 H100 的 WGMMA 指令：

```cpp
static constexpr int BLOCK_SIZE = 64;
static constexpr int NUM_WORKERS = 4;  // 1 warpgroup = 4 warps
static constexpr int NUM_THREADS = NUM_WORKERS * kittens::WARP_THREADS; // 128 线程

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // 共享内存分配（同上）
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &As = al.allocate<...>();
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &Bs = al.allocate<...>();

    // 每个 warp 持有 16×64 的累加器切片
    rt_fl<16, BLOCK_SIZE> C_accum;      // 注意：16 行，不是 64 行！
    rt_fl<16, BLOCK_SIZE> C_accum_cpy;

    kittens::warp::zero(C_accum_cpy);
    for (int tile = 0; tile < num_tiles; ++tile) {
        // warpgroup 协作加载 64×64 瓦片到共享内存
        warpgroup::load(As, g.A, {0, 0, row, tile});
        warpgroup::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();

        // 关键：warpgroup 级 MMA（异步，直接从共享内存操作）
        warpgroup::mma_AB(C_accum, As, Bs);  // 无需先加载到寄存器！
        warpgroup::mma_async_wait();          // 等待异步 MMA 完成

        kittens::warp::add(C_accum_cpy, C_accum_cpy, C_accum);
        kittens::warp::zero(C_accum);
    }
    warpgroup::store(g.C, C_accum_cpy, {0, 0, row, col});
}
```

**关键区别**：
- `warpgroup::mma_AB` 直接从共享内存瓦片 (`st`) 操作，无需显式 `load` 到寄存器
- 底层调用 H100 的异步 WGMMA PTX 指令
- 每个 warp 只持有 16 行结果，4 个 warp 合起来覆盖完整的 64 行

---

## Level 6：TMA 异步加载 + 双缓冲

引入 Tensor Memory Accelerator (TMA) 和双缓冲流水线：

```cpp
__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // 双缓冲：两组共享内存瓦片
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>();
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>();
    int tic = 0, toc = 1;

    // 信号量：协调 TMA 异步加载
    __shared__ semaphore bar;
    if (threadIdx.x == 0) {
        init_semaphore(bar, 0, 1);
        // 告诉硬件预期接收多少字节
        tma::expect_bytes(bar, size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>);
        // 异步 TMA 加载第一组瓦片
        tma::load_async(As[tic], g.A, {0, 0, row, 0}, bar);
        tma::load_async(Bs[tic], g.B, {0, 0, 0, col}, bar);
    }
    __syncthreads();

    for (int tile = 0; tile < num_tiles; ++tile, tic ^= 1, toc ^= 1) {
        // 等待当前瓦片加载完成
        wait(bar, tic);
        __syncthreads();

        // 同时启动下一组瓦片的 TMA 加载（与计算重叠！）
        if (threadIdx.x == 0 && tile + 1 < num_tiles) {
            tma::expect_bytes(bar, ...);
            tma::load_async(As[toc], g.A, {0, 0, row, tile+1}, bar);
            tma::load_async(Bs[toc], g.B, {0, 0, tile+1, col}, bar);
        }

        // 计算当前瓦片
        warpgroup::mma_AB(C_accum, As[tic], Bs[tic]);
        warpgroup::mma_async_wait();
        ...
    }
}
```

**TMA 的优势**：
- 只需 1 个线程发起加载，硬件自动搬运整个瓦片
- 双缓冲 (`tic/toc`) 实现加载与计算的流水线重叠
- 信号量 (`semaphore`) 替代传统 `__syncthreads()`，更精细的同步控制

---

## Level 7：生产者-消费者 Warp 特化

最核心的架构模式——不同 warpgroup 承担不同角色：

```cpp
static constexpr int NUM_WORKERS = 8; // 2 个 warpgroup

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    const int warpgroupid = kittens::warpid() / 4;

    // 信号量：full = "数据已就绪", empty = "缓冲区已释放"
    __shared__ semaphore full[QSIZE], empty[QSIZE];

    if (warpgroupid == 0) {
        // ===== 生产者 warpgroup：专注于数据搬运 =====
        warpgroup::decrease_registers<32>();  // 释放寄存器给消费者
        if (warpgroup::laneid() == 0) {
            for (int tile = 0; tile < num_tiles; ++tile) {
                wait(empty[qidx], p);        // 等待消费者释放缓冲区
                tma::expect_bytes(full[qidx], ...);
                tma::load_async(As[qidx], g.A, ..., full[qidx]);
                tma::load_async(Bs[qidx], g.B, ..., full[qidx]);
            }
        }
    } else {
        // ===== 消费者 warpgroup：专注于计算 =====
        warpgroup::increase_registers<256>(); // 获取更多寄存器
        rt_fl<16, BLOCK_SIZE> C_accum;
        kittens::warp::zero(C_accum);

        for (int tile = 0; tile < num_tiles; ++tile) {
            wait(full[qidx], p);             // 等待数据就绪
            warpgroup::mma_AB(C_accum, As[qidx], Bs[qidx]);
            warpgroup::mma_async_wait();
            arrive(empty[qidx], 1);          // 通知生产者缓冲区可复用
        }
        warpgroup::store(g.C, C_accum, {0, 0, row, col});
    }
}
```

**关键设计**：
- **寄存器重分配**：生产者减少寄存器（`decrease_registers<32>`），消费者增加（`increase_registers<256>`）
- **双重信号量**：`full[]` 通知"数据已就绪"，`empty[]` 通知"缓冲区可复用"
- **完全重叠**：生产者加载 tile N+1 的同时，消费者计算 tile N

---

## Level 8：完整的高性能生产级 GEMM（LCF 模板）

最终版本使用 TK 的 LCF (Load-Compute-Finish) 原型模板：

```cpp
#include "kittens.cuh"
#include "prototype.cuh"
using namespace kittens::prototype::lcf;

template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using base_tile     = st_bf<64, 64>;
    using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
    struct globals        { global_layout A, B, C; };
    struct input_block    { base_tile a[M_BLOCK], b[N_BLOCK]; };  // 输入缓冲
    struct finish_block   { base_tile c[M_BLOCK][N_BLOCK]; };     // 输出缓冲
    struct consumer_state { rt_fl<16, N_BLOCK*64> accum; };       // 每消费者累加器
};

template<int M_BLOCK=2, int N_BLOCK=4, int SUPER_M=12>
struct matmul_template {
    using layout = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4;

    // 生产者：加载数据
    struct producer {
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x+i, args.iter}, args.inputs_arrived);
                for (int i = 0; i < N_BLOCK; i++)
                    tma::load_async(args.input.b[i], args.globals.B,
                                    {args.iter, args.common.coord.y+i}, args.inputs_arrived);
            }
        }
    };

    // 消费者：计算 + 输出
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                reinterpret_cast<wide_tile&>(args.input.b)
            );
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(
                args.finish.c[warpgroup::groupid()]), args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if (warpgroup::laneid() == 0)
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C,
                        args.finish.c[warpgroup::groupid()][i],
                        {args.common.coord.x, args.common.coord.y+i});
                    tma::store_async_read_wait();
                }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// 启动：仅需 2 行
dim3 grid(matmul_template::grid(M, N, K));
prototype::lcf::kernel<matmul_template><<<grid, block, shared_mem>>>(globals);
```

**LCF 模板的优势**：
- 用户只需定义 `layout`、`producer::load`、`consumer::compute`、`consumer::finish`
- 流水线管理、信号量、双缓冲全部由框架自动处理
- 支持持久化网格（persistent grid）：132 个 block 循环处理所有瓦片
- 支持超级瓦片 (super tiling) 优化 L2 cache 命中率

---

## LayerNorm 案例：展示向量操作

LayerNorm 内核展示了 TK 在非 MMA 场景下的使用（向量级操作）：

```cpp
template<int D>
__global__ void layernorm_tk(const __grid_constant__ norm_globals<D> g, int n_per_tile) {
    auto warpid = kittens::warpid();

    // 共享内存中的向量
    sv_bf<d_model> (&x_s)[2][NUM_WORKERS] = al.allocate<sv_bf<d_model>, 2, NUM_WORKERS>();
    sv_bf<d_model> (&residual_s)[2][NUM_WORKERS] = al.allocate<...>();
    sv_bf<d_model> &norm_weight_s = al.allocate<sv_bf<d_model>>();
    sv_bf<d_model> &norm_bias_s   = al.allocate<sv_bf<d_model>>();

    // 加载 norm 参数
    warp::load(norm_weight_s, g.norm_weight, {0,0,0,0});
    warp::load(norm_bias_s,   g.norm_bias,   {0,0,0,0});

    for (int block = 0; block < n_blocks; block++) {
        // 异步预取下一批数据
        warp::load_async(x_s[warpid][toc], g.x, ...);

        // residual = residual + x
        warp::add(residual_s[...], residual_s[...], x_s[...]);

        // mean = sum(residual) / d_model
        warp::sum(mean, residual_s[...]);
        mean = mean / __float2bfloat16(d_model);

        // residual = residual - mean
        warp::sub(residual_s[...], residual_s[...], mean);

        // var = sum(residual^2) / d_model
        warp::mul(x_s[...], residual_s[...], residual_s[...]);
        warp::sum(var, x_s[...]);

        // normalize: residual = residual / sqrt(var + eps)
        warp::div(residual_s[...], residual_s[...], var);

        // affine: residual = residual * weight + bias
        warp::mul(residual_s[...], residual_s[...], norm_weight_s);
        warp::add(residual_s[...], residual_s[...], norm_bias_s);

        // 写回
        warp::store(g.o, residual_s[...], ...);
    }
}
```

**向量操作展示了**：
- `sv_bf<D>`：共享内存向量（Shared Vector）
- `warp::add/sub/mul/div`：在共享向量上的逐元素操作
- `warp::sum`：warp 级规约求和
- 双缓冲流水线同样适用于非 MMA 场景

---

## API 速查表

### 数据类型
```cpp
// 寄存器瓦片 (Register Tile)
rt_bf<rows, cols>                     // bf16，默认行优先
rt_fl<rows, cols>                     // float，用于累加
rt_bf<R, C, ducks::rt_layout::col>   // 列优先（MMA 的 B 操作数）

// 共享内存瓦片 (Shared Tile)
st_bf<rows, cols>                     // bf16 共享瓦片

// 共享内存向量 (Shared Vector)
sv_bf<length>                         // bf16 共享向量

// 全局内存布局 (Global Layout)
gl<bf16, batch, head, rows, cols, sub_tile>  // -1 表示运行时动态尺寸
```

### Warp 级操作
```cpp
kittens::warp::zero(tile);                          // 清零
kittens::warp::load(shared_tile, global, {coords}); // HBM → SRAM
kittens::warp::load(reg_tile, shared_tile);         // SRAM → 寄存器
kittens::warp::store(global, reg_tile, {coords});   // 寄存器 → HBM
kittens::warp::swap_layout(dst_col, src_row);       // 行优先 ↔ 列优先
kittens::warp::mma_AB(D, A, B, C);                  // D = A×B + C (Tensor Core)
kittens::warp::add(dst, a, b);                      // 逐元素加
kittens::warp::mul(dst, a, b);                      // 逐元素乘
kittens::warp::exp(dst, src);                       // 逐元素 exp
kittens::warp::sum(scalar, vector);                 // 规约求和
```

### Warpgroup 级操作
```cpp
warpgroup::load(shared_tile, global, {coords});     // 4 warp 协作加载
warpgroup::mma_AB(accum, shared_A, shared_B);       // 异步 WGMMA（直接从 SRAM）
warpgroup::mma_async_wait();                        // 等待异步 MMA 完成
warpgroup::store(global, reg_tile, {coords});       // 协作存储
warpgroup::increase_registers<N>();                 // 增加寄存器分配
warpgroup::decrease_registers<N>();                 // 减少寄存器分配
```

### TMA 操作
```cpp
tma::expect_bytes(semaphore, nbytes);               // 声明期望字节数
tma::load_async(shared_tile, global, {coords}, sem); // TMA 异步加载
tma::store_async(global, shared_tile, {coords});     // TMA 异步存储
init_semaphore(bar, initial_count, expected_count);  // 初始化信号量
wait(bar, phase);                                    // 等待信号量
arrive(bar, count);                                  // 到达信号量
```

---

## 性能对比（4096×4096 GEMM, H100）

| Level | 方法 | 预期性能 |
|-------|------|---------|
| 1 | 朴素 FP32 | ~0.5 TFLOPs |
| 2 | 朴素 BF16 | ~1 TFLOPs |
| 3 | 共享内存分块 | ~3-5 TFLOPs |
| 4 | TK 单 warp | ~30-50 TFLOPs |
| 5 | TK warpgroup WGMMA | ~200+ TFLOPs |
| 6 | TK + TMA 双缓冲 | ~300+ TFLOPs |
| 7 | TK 生产者-消费者 | ~400+ TFLOPs |
| 8 | TK 完整 LCF 模板 | ~800+ TFLOPs (≈CuBLAS) |

引入 Tensor Core（Level 4）后性能跳跃式提升，之后每级优化进一步挖掘硬件潜力。
