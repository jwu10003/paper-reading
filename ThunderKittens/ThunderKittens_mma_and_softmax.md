# ThunderKittens 深度解析：MMA 指令与 Online Softmax

基于 ThunderKittens 开源仓库源码的详细分析。

---

## 一、`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` 指令详解

### 1.1 指令语义拆解

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
│    │     │       │       │   │   │    │     │    └─ C/D 累加器类型: float32
│    │     │       │       │   │   │    │     └────── B 矩阵类型: bfloat16
│    │     │       │       │   │   │    └──────────── A 矩阵类型: bfloat16
│    │     │       │       │   │   └─────────────────  D 输出类型: float32
│    │     │       │       │   └─────────────────────  B 布局: 列优先 (col-major)
│    │     │       │       └─────────────────────────  A 布局: 行优先 (row-major)
│    │     │       └─────────────────────────────────  形状: M=16, N=8, K=16
│    │     └─────────────────────────────────────────  内存对齐要求
│    └───────────────────────────────────────────────  warp 内 32 线程同步执行
└────────────────────────────────────────────────────  矩阵乘累加指令
```

**核心语义**：在一个 warp（32 线程）内协作计算：

```
D[16×8] = A[16×16] × B[16×8] + C[16×8]
```

注意硬件 MMA **一次只产生 16×8 的输出**，而非 16×16。

### 1.2 TK 中的 PTX 包装函数

源码位置：`include/ops/group/mma/warp.cuh:23-50`

```cpp
__device__ static inline void hmma16816(
    float2 &d0, float2 &d1,                     // D 输出 [16×8]
    const bf16_2 &a0, const bf16_2 &a1,          // A 矩阵 [16×16]
    const bf16_2 &a2, const bf16_2 &a3,
    const bf16_2 &b0, const bf16_2 &b1,          // B 矩阵 [16×8]
    const float2 &c0, const float2 &c1           // C 累加器 [16×8]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "     // D: 4 个 float (2 个 float2)
        "{%4, %5, %6, %7}, "     // A: 4 个 uint32 (4 个 bf16_2)
        "{%8, %9}, "             // B: 2 个 uint32 (2 个 bf16_2)
        "{%10, %11, %12, %13};"  // C: 4 个 float (2 个 float2)
        : "+f"(d0.x), "+f"(d0.y), "+f"(d1.x), "+f"(d1.y)
        : "r"(*(uint32_t*)(&a0)), "r"(*(uint32_t*)(&a1)),
          "r"(*(uint32_t*)(&a2)), "r"(*(uint32_t*)(&a3)),
          "r"(*(uint32_t*)(&b0)), "r"(*(uint32_t*)(&b1)),
          "f"(c0.x), "f"(c0.y), "f"(c1.x), "f"(c1.y)
    );
}
```

### 1.3 每线程的寄存器分布

每个线程持有矩阵的一个片段：

```
A 矩阵 [16×16, bf16, 行优先]：每线程 4 个 bf16_2 = 8 个 bf16 元素
  a0 = data[0] → 上左 8×8 象限 (row 0-7,  col 0-7)
  a1 = data[1] → 下左 8×8 象限 (row 8-15, col 0-7)
  a2 = data[2] → 上右 8×8 象限 (row 0-7,  col 8-15)
  a3 = data[3] → 下右 8×8 象限 (row 8-15, col 8-15)

B 矩阵 [16×8, bf16, 列优先]：每线程 2 个 bf16_2 = 4 个 bf16 元素
  b0, b1 → 分布在 16×8 中

C/D 矩阵 [16×8, f32, 行优先]：每线程 2 个 float2 = 4 个 float 元素
  c0/d0 → 上半部分 (row 0-7)
  c1/d1 → 下半部分 (row 8-15)
```

线程到矩阵元素的精确映射（行优先布局）：

```
thread  k=0: row = laneid/4,      col = (laneid%4)*2     → 上左象限
thread  k=1: row = 8 + laneid/4,  col = (laneid%4)*2     → 下左象限
thread  k=2: row = laneid/4,      col = 8 + (laneid%4)*2 → 上右象限
thread  k=3: row = 8 + laneid/4,  col = 8 + (laneid%4)*2 → 下右象限

每个 data[k] 是 packed 类型 (float2/bf16_2)，.x 和 .y 覆盖相邻两列
```

### 1.4 从 16×8 到 16×16：两次 MMA 调用

硬件一次只计算 16×8，TK 的 `mma_AB_base` 调用**两次** `hmma16816` 覆盖完整的 16×16 输出。

源码位置：`warp.cuh:200-216`

```cpp
__device__ static inline void mma_AB_base(
    rt_base<float, row> &d,      // D [16×16, f32]
    const rt_base<bf16, row> &a, // A [16×16, bf16, 行优先]
    const rt_base<bf16, col> &b, // B [16×16, bf16, 列优先]
    const rt_base<float, row> &c // C [16×16, f32]
) {
    // 第一次 MMA: 计算 D 的左半部分 D[:, 0:8]
    hmma16816(
        d.data[0], d.data[1],                         // D[:, 0:8]  输出
        a.data[0], a.data[1], a.data[2], a.data[3],   // 完整 A [16×16]
        b.data[0], b.data[2],                          // B 的左半列 [:, 0:8]
        c.data[0], c.data[1]                           // C[:, 0:8]  累加器
    );

    // 第二次 MMA: 计算 D 的右半部分 D[:, 8:16]
    hmma16816(
        d.data[2], d.data[3],                          // D[:, 8:16] 输出
        a.data[0], a.data[1], a.data[2], a.data[3],   // 相同的 A
        b.data[1], b.data[3],                          // B 的右半列 [:, 8:16]
        c.data[2], c.data[3]                           // C[:, 8:16] 累加器
    );
}
```

图示：

```
         B [16×16, 列优先]
         ┌────────┬────────┐
         │ b[0]   │ b[1]   │
         │ b[2]   │ b[3]   │   16 rows
         └────────┴────────┘
          左8列     右8列

A [16×16, 行优先]         D [16×16, f32]
┌──────────────┐         ┌────────────────────┐
│ a[0]   a[2]  │         │  第1次MMA   │ 第2次MMA  │
│ a[1]   a[3]  │  16行    │  D[:,0:8]  │ D[:,8:16] │
│              │         │  A×B左     │ A×B右     │
└──────────────┘         └────────────────────┘
   16 cols
```

### 1.5 从基础瓦片到任意尺寸：分块循环

源码位置：`warp.cuh:583-632`

```cpp
template<row_layout D, row_layout A, col_layout B, row_layout C>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, const C &c) {
    // 编译时检查（概述见下文 1.6 节）
    static_assert(D::rows == A::rows && D::cols == B::cols);
    static_assert(A::cols == B::rows);  // K 维度必须匹配
    static_assert(D::rows == C::rows && D::cols == C::cols);

    #pragma unroll
    for(int n = 0; n < D::height; n++) {          // 遍历输出行方向的子瓦片
        #pragma unroll
        for(int m = 0; m < D::width; m++) {        // 遍历输出列方向的子瓦片
            // 第一次 K 迭代：用 C 初始化
            mma_AB_base(d.tiles[n][m], a.tiles[n][0], b.tiles[0][m], c.tiles[n][m]);
            #pragma unroll
            for(int k = 1; k < A::width; k++) {    // 沿 K 维度累加
                // 后续迭代：累加到 D 自身
                mma_AB_base(d.tiles[n][m], a.tiles[n][k], b.tiles[k][m], d.tiles[n][m]);
            }
        }
    }
}
```

举例：`rt_bf<64, 128>` × `rt_bf<128, 64>` → `rt_fl<64, 64>`

```
D::height = 64/16 = 4,  D::width = 64/16 = 4,  A::width = 128/16 = 8
→ 4 × 4 × 8 = 128 次 mma_AB_base 调用
→ 每次内部 2 次 hmma16816
→ 共 256 次 PTX MMA 指令
→ 全部在寄存器内完成，#pragma unroll 展开为直线代码
```

### 1.6 编译时类型与布局安全检查

```cpp
// mma_AB 要求: A=行优先, B=列优先, C/D=行优先
template<ducks::rt::row_layout D,    // ← concept 约束
         ducks::rt::row_layout A,
         ducks::rt::col_layout B,    // ← B 必须列优先
         ducks::rt::row_layout C>

// 维度检查
static_assert(D::rows == A::rows && D::cols == B::cols);
static_assert(A::cols == B::rows);

// 类型组合检查（只允许合法的精度组合）
static_assert(
    (D=float, A=bf16, B=bf16, C=float) ||   // bf16 输入, f32 累加
    (D=half,  A=half, B=half, C=half)  ||   // fp16 全精度
    (D=float, A=fp8,  B=fp8,  C=float)      // fp8 输入, f32 累加 (Hopper+)
);
```

如果用户传入行优先的 B 矩阵给 `mma_AB`，**编译时直接报错**，而不是运行时产生错误结果。

### 1.7 四种 MMA 变体

TK 提供了四种转置组合，全部基于相同的 `hmma16816` PTX 指令，通过不同的数据索引实现：

| 函数 | 计算 | A 布局 | B 布局 | 典型场景 |
|------|------|--------|--------|----------|
| `mma_AB(d,a,b,c)` | D=A×B+C | row | **col** | GEMM |
| `mma_ABt(d,a,b,c)` | D=A×B^T+C | row | row | Attention: Q×K^T |
| `mma_AtB(d,a,b,c)` | D=A^T×B+C | col | col | 反向传播 |
| `mma_AtBt(d,a,b,c)` | D=A^T×B^T+C | col | row | 特殊场景 |

关键技巧：`mma_ABt` 让 A 和 B 都是行优先——因为 B^T 的行优先等于 B 的列优先，所以硬件仍然看到 `row × col` 的 MMA。

---

## 二、Attention 前向中的 Online Softmax

源码位置：`kernels/attention/mha_h100/mha_h100.cu`

### 2.1 内核架构

```
fwd_attend_ker<D, is_causal>
├── 4 个 warpgroup（16 个 warp = 512 线程）
│   ├── warpgroup 0-2：消费者（并行处理 3 个 Q 块的 attention）
│   └── warpgroup 3：  生产者（TMA 异步加载 K, V 瓦片）
├── 共享内存布局：
│   ├── q_smem[3]：3 个 Q 瓦片（每消费者 1 个）
│   ├── k_smem[stages]：K 瓦片流水线缓冲区
│   ├── v_smem[stages]：V 瓦片流水线缓冲区
│   └── l_smem[3]：logsumexp 输出
└── 瓦片尺寸（D=64 为例）：
    ├── Q/O: 64×64 (qo_height × tile_width)
    └── K/V: 128×64 (kv_height × tile_width)
```

### 2.2 消费者 warp 的寄存器变量

源码：`mha_h100.cu:130-137`

```cpp
// 每个 warp 持有 16 行（warpgroup 的 4 个 warp 合计覆盖 64 行）
rt_fl<16, K::kv_height>  att_block;      // S = Q×K^T [16×128, float]
rt_bf<16, K::kv_height>  att_block_mma;  // P 转 bf16（用于 P×V 的 MMA）
rt_fl<16, K::tile_width> o_reg;          // 输出 O [16×D, float]

// 列向量（每行一个标量，用于 online softmax）
col_vec<rt_fl<16, K::kv_height>> max_vec;            // m_i: 行最大值
col_vec<rt_fl<16, K::kv_height>> norm_vec;           // l_i: 归一化因子
col_vec<rt_fl<16, K::kv_height>> max_vec_last_scaled; // 上一轮缩放后的 max
col_vec<rt_fl<16, K::kv_height>> max_vec_scaled;      // 当前轮缩放后的 max
```

### 2.3 初始化

```cpp
warp::neg_infty(max_vec);  // m_i = -∞
warp::zero(norm_vec);      // l_i = 0
warp::zero(o_reg);         // O = 0
```

### 2.4 Online Softmax 主循环（逐步解析）

源码：`mha_h100.cu:149-205`

```cpp
for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
```

#### 步骤 1：计算注意力分数 S = Q × K^T

```cpp
wait(k_smem_arrived[...]);  // 等待 K 瓦片加载完成
warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[...]);
// ↑ 从共享内存直接做异步 WGMMA
// 内部最终调用 mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// Q [64×D] × K^T [D×128] → S [64×128]（每 warp 持有 16 行）
```

#### 步骤 2：保存上一轮的 max

```cpp
warp::copy(max_vec_last_scaled, max_vec);
warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
//                                                   log2(e) × 1/√d_k
// 使用 exp2 代替 exp：exp(x) = 2^(x·log2(e))，exp2 是硬件原生指令更快
```

```cpp
warpgroup::mma_async_wait();  // 等待 S 计算完成
```

#### 步骤 3：因果遮罩（可选）

```cpp
if constexpr (is_causal) {
    // 逐 16×16 子瓦片处理
    if (k_idx > q_blk)       warp::neg_infty(attn_subtile);   // 完全在未来 → -∞
    else if (k_idx == q_blk) warp::make_causal(attn_subtile, attn_subtile, neg_infty);
    //                       ↑ 对角线以上设为 -∞，对角线及以下保留
}
```

`make_causal` 内部利用 `apply(tile, lambda)` 根据每个元素的 (row, col) 坐标判断是否置零。

#### 步骤 4：更新行最大值

```cpp
warp::row_max(max_vec, att_block, max_vec);
// max_vec[i] = max(max_vec[i], max_j(att_block[i, j]))
// 内部实现：
//   1. 先在每个线程的 packed 值内取 max
//   2. 用 __shfl_down_sync 在 warp 内跨线程规约
//   3. 最终每行一个标量 max
```

#### 步骤 5：缩放并减去 max

```cpp
// 缩放因子 = log2(e) / sqrt(d_k)
warp::mul(att_block, att_block, 1.44269504089f * 0.125f);      // S_scaled = S × scale
warp::mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);   // m_scaled = m × scale
warp::sub_row(att_block, att_block, max_vec_scaled);            // S_shifted = S_scaled - m_scaled
// sub_row: 每行减去对应的标量，利用 rv 的 align/ortho 布局直接对齐操作
```

#### 步骤 6：计算 exp2（硬件原生指令）

```cpp
warp::exp2(att_block, att_block);  // P = 2^(S_shifted)
// 等价于 P = exp(S - max)，但用 exp2 更快（GPU 有专用硬件单元）
```

#### 步骤 7：修正历史状态（Online Softmax 核心）

```cpp
// correction = 2^(m_old_scaled - m_new_scaled) = exp(m_old - m_new)
warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
warp::exp2(max_vec_last_scaled, max_vec_last_scaled);

// 修正旧的归一化因子
warp::mul(norm_vec, norm_vec, max_vec_last_scaled);   // l_i *= correction

// 累加当前块的行和
warp::row_sum(norm_vec, att_block, norm_vec);          // l_i += Σ_j P[i,j]
```

#### 步骤 8：类型转换 float → bf16

```cpp
warp::add(att_block, att_block, 0.f);    // flush denormal floats
warp::copy(att_block_mma, att_block);     // float32 → bfloat16 类型转换
```

#### 步骤 9：修正并累加输出 O

```cpp
// 修正旧的 O（因为 max 更新了）
warp::mul_row(o_reg, o_reg, max_vec_last_scaled);  // O *= correction

// O += P × V（第二次 Tensor Core MMA）
wait(v_smem_arrived[...]);
warpgroup::mma_AB(o_reg, att_block_mma, v_smem[...]);
// att_block_mma [16×kv_height, bf16] × V [kv_height×D, bf16] → O [16×D, f32]
// 再次调用 mma.sync.aligned.m16n8k16

warpgroup::mma_async_wait();
arrive(compute_done[...]);  // 通知生产者缓冲区可复用
```

#### 步骤 10：最终归一化

```cpp
warp::div_row(o_reg, o_reg, norm_vec);  // O[i,:] /= l_i
```

### 2.5 Online Softmax 数学对应

标准 softmax：`softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)`

Online 版本维护两个运行变量 `m_i`（行最大值）和 `l_i`（归一化因子），每处理一个新 KV 块时：

```
输入: 当前 KV 块得到的 S 分数, 历史状态 (m_old, l_old, O_old)

1. m_new = max(m_old, max(S_j))
   → TK: warp::row_max(max_vec, att_block, max_vec)

2. correction = exp(m_old - m_new)
   → TK: warp::sub + warp::exp2

3. P_j = exp(S_j - m_new)
   → TK: warp::sub_row + warp::exp2

4. l_new = l_old × correction + Σ P_j
   → TK: warp::mul(norm_vec, correction) + warp::row_sum(norm_vec, P)

5. O_new = O_old × correction + P_j × V_j
   → TK: warp::mul_row(o_reg, correction) + warpgroup::mma_AB(o_reg, P, V)

6. 循环结束后: O_final = O / l
   → TK: warp::div_row(o_reg, norm_vec)
```

### 2.6 每次 KV 迭代中 MMA 指令的使用总结

```
┌─────────────────────────────────────────────────────────────────────┐
│ KV 块迭代 kv_idx                                                    │
│                                                                     │
│ ┌── 第 1 次 MMA: S = Q × K^T ──────────────────────────────────┐   │
│ │  warpgroup::mm_ABt(att_block, q_smem, k_smem)                │   │
│ │  └→ mma_ABt_base() × (qo_h/16 × kv_h/16 × D/16) 次         │   │
│ │     └→ hmma16816() × 2 次 per subtile                        │   │
│ │        └→ mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32│   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│ ┌── Online Softmax（纯寄存器操作，无 MMA）─────────────────────┐   │
│ │  row_max → mul → sub_row → exp2 → mul → row_sum              │   │
│ │  → copy(bf16) → mul_row                                       │   │
│ │  全部在 rt_fl / col_vec 上原地执行，延迟极低                    │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│ ┌── 第 2 次 MMA: O += P × V ───────────────────────────────────┐   │
│ │  warpgroup::mma_AB(o_reg, att_block_mma, v_smem)             │   │
│ │  └→ 同上调用链                                                │   │
│ │  att_block_mma [bf16] × V [bf16] → O [f32]                   │   │
│ └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│ 同时: 生产者 warpgroup 通过 TMA 预加载下一个 K, V 块              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.7 TK 对比传统实现的优势

| 方面 | 传统 CUDA | ThunderKittens |
|------|-----------|----------------|
| MMA 调用 | 手写 PTX asm | `warpgroup::mma_ABt(dst, A, B)` 一行 |
| Softmax max | 手动 `__shfl` 规约 | `warp::row_max(vec, tile, vec)` |
| Softmax exp | `expf()` (标量) | `warp::exp2(tile, tile)` (瓦片级) |
| 类型转换 | 手动 `__float2bfloat16` | `warp::copy(bf16_tile, f32_tile)` |
| 布局安全 | 运行时错误结果 | **编译时报错** |
| Bank conflict | 手动计算 swizzle | 自动消除（85% 停顿减少） |
| 代码量 | 数千行 | ~200 行内核 + 框架 |

### 2.8 性能关键：exp2 vs exp

TK 使用 `exp2`（以 2 为底）而非 `exp`（以 e 为底），因为：

1. GPU 有专用 `ex2.approx.ftz.f32` SFU 指令（Special Function Unit），吞吐量远高于通用 `exp`
2. 数学等价：`exp(x) = 2^(x × log2(e))`
3. TK 将缩放因子预乘 `log2(e) = 1.44269504089`，一次 `mul` + 一次 `exp2` 替代一次 `exp`
4. 最终 logsumexp 输出时再乘回 `ln(2) = 0.69314718056` 修正

---

## 附录：源码文件索引

| 文件 | 内容 |
|------|------|
| `include/ops/group/mma/warp.cuh` | PTX MMA 包装函数、mma_AB/ABt/AtB/AtBt |
| `include/types/register/rt_base.cuh` | 16×16 基础瓦片数据结构 |
| `include/types/register/rt.cuh` | 可组合的寄存器瓦片 |
| `include/types/register/rt_layout.cuh` | 行/列布局定义与 concepts |
| `include/ops/group/register/tile/maps.cuh` | 逐元素操作 (exp2, mul, sub_row...) |
| `include/ops/group/register/tile/reductions.cuh` | 规约操作 (row_max, row_sum...) |
| `include/ops/group/register/tile/conversions.cuh` | 布局转换、make_causal |
| `kernels/attention/mha_h100/mha_h100.cu` | 完整 FlashAttention 前向+反向 |
| `kernels/gemm/educational_h100/level_04.cu` | TK GEMM 入门（warp 级 MMA） |
| `kernels/gemm/educational_h100/level_05.cu` | Warpgroup 级异步 WGMMA |
