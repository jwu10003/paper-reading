# ThunderKittens 深度解析：线程块级别（共享内存层）与 Grid 级别（全局内存层）

基于 ThunderKittens 开源仓库源码的详细分析。

---

## 一、线程块级别（共享内存层）

线程块级别对应 GPU 的共享内存（Shared Memory / SRAM），是 TK 三层抽象的中间层。共享内存在同一线程块内的所有线程之间共享，带宽远高于全局内存（HBM），但容量有限（H100 上最高 227KB）。

### 1.1 共享内存瓦片 `st<T, rows, cols>`

源码位置：`include/types/shared/st.cuh`

#### 基本结构

```cpp
template<typename _T, int _rows, int _cols, bool _swizzle=true, int _swizzle_bytes=0>
struct KITTENS_DEFAULT_ALIGN st {
    using identifier = ducks::st::identifier;  // "鸭子类型"标识符
    using dtype = T;

    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
    static constexpr int num_elements = rows * cols;

    dtype data[rows * cols];  // 原始数据存储
};
```

**关键设计**：
- 尺寸约束：`rows` 必须是 `TILE_ROW_DIM`（16）的倍数，`cols` 必须是 `TILE_COL_DIM`（16）的倍数
- 默认启用 swizzle（`_swizzle=true`），自动消除 bank conflict
- 128 字节对齐（`KITTENS_DEFAULT_ALIGN`），满足 TMA 和 WGMMA 要求

#### 类型别名

```cpp
st_bf<R, C>      // bf16 共享瓦片
st_hf<R, C>      // half (fp16) 共享瓦片
st_fl<R, C>      // float 共享瓦片
st_fp8e4m3<R, C> // fp8 共享瓦片 (Hopper+)
```

### 1.2 Swizzle 机制：自动消除 Bank Conflict

这是 TK 共享内存层最核心的创新之一。

#### 什么是 Bank Conflict

GPU 共享内存被分为 32 个 bank，每个 bank 宽 4 字节。当一个 warp 中的多个线程同时访问同一 bank 的不同地址时，访问会被串行化，产生性能损失。

#### TK 的 Swizzle 解决方案

TK 在编译时根据瓦片宽度自动选择 swizzle 模式：

```cpp
// include/types/shared/st.cuh:91-103
static constexpr int swizzle_bytes = _swizzle_bytes > 0 ? _swizzle_bytes : (
    sizeof(dtype) == 2 ? (                    // bf16/fp16 (2 bytes)
        (cols/TILE_COL_DIM<T>)%4 == 0 ? 128 : // ≥4 个子瓦片 → 128B swizzle
        (cols/TILE_COL_DIM<T>)%2 == 0 ?  64 : // ≥2 个子瓦片 → 64B swizzle
        32                                      // 1 个子瓦片 → 32B swizzle
    ) :
    sizeof(dtype) == 4 ? (                    // float (4 bytes)
        (cols/TILE_COL_DIM<T>)%2 == 0 ? 128 :
        64
    ) : -1
);
```

选择规则：列越宽，swizzle 粒度越大，bank conflict 消除越彻底。

#### Swizzle 索引公式

```cpp
// st.cuh:107-119 — idx() 函数
__device__ static inline T* idx(T *ptr, int2 coord) {
    int r = coord.x, c = coord.y;
    if constexpr (swizzle) {
        static constexpr int swizzle_repeat = swizzle_bytes * 8;
        static constexpr int subtile_cols   = swizzle_bytes / sizeof(T);
        const int outer_idx = c / subtile_cols;
        const uint64_t addr = (uint64_t)(&ptr[outer_idx*rows*subtile_cols
                                              + r*subtile_cols
                                              + c%subtile_cols]);
        const int swizzle = ((addr % swizzle_repeat) >> 7) << 4;
        return (T*)(addr ^ swizzle);
    } else {
        return &ptr[r*cols + c];       // 无 swizzle 时退化为行优先
    }
}
```

**核心 XOR 公式**：`addr ^ (((addr % (swizzle_bytes*8)) >> 7) << 4)`

工作原理：
1. 将逻辑上连续的行拆分为 `swizzle_bytes` 大小的子块
2. 对每个子块内的地址，根据其在 swizzle 周期内的位置，对 bit[4:6] 做 XOR 变换
3. 这使得相邻行的数据错开到不同的 bank，避免 warp 内线程冲突

图示（128B swizzle, bf16）：

```
逻辑布局 (行优先)                  物理布局 (swizzled)
┌─────────────────┐               ┌─────────────────┐
│ row0: 0  1  2  3│               │ row0: 0  1  2  3│  ← 不变
│ row1: 0  1  2  3│               │ row1: 1  0  3  2│  ← XOR 变换
│ row2: 0  1  2  3│               │ row2: 2  3  0  1│  ← XOR 变换
│ row3: 0  1  2  3│               │ row3: 3  2  1  0│  ← XOR 变换
└─────────────────┘               └─────────────────┘
(数字代表32B bank组)               不同行的相同列落在不同bank
```

**用户透明**：通过重载 `operator[]`，用户使用 `tile[{row, col}]` 访问时自动 swizzle，无需关心底层细节。

### 1.3 共享内存向量 `sv<T, length>`

源码位置：`include/types/shared/sv.cuh`

```cpp
template<typename _T, size_t _length>
struct KITTENS_DEFAULT_ALIGN sv {
    using identifier = ducks::sv::identifier;
    using dtype = T;
    static constexpr int length = _length;

    // Hopper+ 上对齐到 128 字节边界
    static constexpr int num_alloc_elements =
        ((length * sizeof(dtype) + 127) / 128) * (128 / sizeof(dtype));

    dtype data[num_alloc_elements];

    __device__ inline dtype& operator[](size_t idx) { return data[idx]; }
};
```

**关键特点**：
- 简单的一维数组布局（无 swizzle，因为向量访问模式不会产生 bank conflict）
- `length` 必须是 `TILE_ROW_DIM`（16）的倍数
- 在 Hopper+ 上自动对齐到 128 字节（为 TMA 准备）
- 类型别名：`sv_bf<L>`, `sv_hf<L>`, `sv_fl<L>`

### 1.4 子瓦片 `st_subtile`

TK 支持从一个大瓦片中提取子瓦片视图，避免数据拷贝：

```cpp
// 创建子瓦片视图（零拷贝）
auto sub = my_tile.subtile<subtile_rows, subtile_cols>({row_idx, col_idx});
// sub 引用原始瓦片的一个子区域，支持所有 st 操作
```

也支持按列方向切分为独立的 `st` 对象：

```cpp
// 按列切分：要求 subtile_cols 是 swizzle_elements 的倍数
auto &left_half = my_tile.subtile<32>(0);   // 左 32 列
auto &right_half = my_tile.subtile<32>(1);  // 右 32 列
```

### 1.5 共享内存分配器

源码位置：`include/common/util.cuh:200-265`

```cpp
template<int default_alignment=1024>  // Hopper 默认 1024 字节对齐
struct shared_allocator {
    int *ptr;

    __device__ shared_allocator(int *_ptr): ptr(_ptr) {}

    template<typename A, size_t... dims>
    __device__ inline auto& allocate() {
        align_ptr<default_alignment>();  // 对齐指针
        using at = variadic_array_t<A, dims...>;
        at *p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at) / sizeof(int);
        return *p;
    }
};

// TMA 专用分配器（要求 1024 字节对齐）
using tma_allocator = shared_allocator<1024>;
using tma_swizzle_allocator = tma_allocator;
```

**使用方法**（来自教程 Level 4）：

```cpp
__global__ void kernel(...) {
    extern __shared__ alignment_dummy __shm[];           // 声明动态共享内存
    shared_allocator al((int*)&__shm[0]);                // 创建分配器

    // 分配单个瓦片
    st_bf<64, 64> &As = al.allocate<st_bf<64, 64>>();

    // 分配多维数组
    st_bf<64, 64> (&Bs)[2] = al.allocate<st_bf<64, 64>, 2>();  // 双缓冲

    // 分配多维嵌套数组
    sv_bf<1024> (&x_s)[2][NUM_WORKERS] = al.allocate<sv_bf<1024>, 2, NUM_WORKERS>();
}
```

分配器自动处理对齐，支持任意维度的数组分配。

### 1.6 共享内存操作

#### 瓦片级操作

源码位置：`include/ops/group/shared/tile/maps.cuh`

所有操作都在 warp/warpgroup 级别并行化，每个线程处理瓦片中的一部分元素：

```cpp
// 核心模式：每线程以 GROUP_THREADS 为步长遍历所有元素
template<typename op, ducks::st::all T>
__device__ static inline void unary_map(T &dst, const T &src) {
    for(int i = laneid(); i < T::num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i]);
    }
}
```

可用操作包括：

| 类别 | 函数 | 说明 |
|------|------|------|
| 常量填充 | `zero`, `one`, `neg_infty`, `pos_infty` | 填充整个瓦片 |
| 一元操作 | `exp`, `exp2`, `log`, `log2`, `abs`, `relu` | 逐元素数学函数 |
| 二元操作 | `add`, `sub`, `mul`, `div`, `max`, `min` | 逐元素/标量运算 |
| 行/列操作 | `add_row`, `sub_row`, `mul_row`, `div_row` | 每行与向量对应元素运算 |
| 列操作 | `add_col`, `sub_col`, `mul_col`, `div_col` | 每列与向量对应元素运算 |
| 广播 | `broadcast_row`, `broadcast_col` | 向量广播为瓦片 |

#### 瓦片级规约

源码位置：`include/ops/group/shared/tile/reductions.cuh`

```cpp
// 行规约核心：每个线程处理一行的完整规约
template<typename op, ducks::sv::all V, ducks::st::all T, bool reset>
__device__ static inline void row_reduce(V &row_accum, const T &src, const V &src_accum) {
    for (int row = laneid(); row < T::rows; row += GROUP_THREADS) {
        dtype accum = src[{row, 0}];
        for (int col = 1; col < T::cols; col++) {
            accum = op::template op<dtype>(accum, src[{row, col}]);
        }
        row_accum[row] = reset ? accum : op::template op<dtype>(src_accum[row], accum);
    }
}
```

可用规约：`row_max`, `row_min`, `row_sum`, `row_prod` 以及对应的 `col_*` 版本。所有规约支持两种模式：重置模式（直接计算）和累加模式（与现有值合并）。

#### 向量级操作

源码位置：`include/ops/group/shared/vec/maps.cuh`

```cpp
// 向量操作：同样的 laneid 步长并行模式
template<typename op, ducks::sv::all T>
__device__ static inline void unary_op(T &dst, const T &src) {
    for(auto cur = laneid(); cur < T::length; cur += GROUP_THREADS) {
        dst[cur] = op::template op<typename T::dtype>(src[cur]);
    }
}
```

支持与瓦片相同的操作集：`zero`, `one`, `exp`, `exp2`, `log`, `abs`, `relu`, `add`, `sub`, `mul`, `div`, `max`, `min`, `copy`。

### 1.7 同步原语：信号量（Semaphore）

源码位置：`include/ops/group/util/sync.cuh`

TK 封装了 CUDA 的 `mbarrier`（memory barrier）指令，提供三个核心原语：

```cpp
// 初始化信号量
__device__ static inline void init_semaphore(semaphore& bar,
                                              int thread_count,
                                              int transaction_count=0) {
    if (laneid() == 0) {
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                     :: "r"(bar_ptr), "r"(thread_count + transaction_count));
    }
}

// 到达信号量（减少计数器）
__device__ static inline void arrive(semaphore& sem) {
    if(laneid() == 0) {
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
                     : : "r"(mbar_ptr) : "memory");
    }
}

// 等待信号量（自旋等待直到 phase 匹配）
__device__ static inline void wait(semaphore& sem, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&sem));
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra.uni DONE;\n"
        "bra.uni LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr), "r"(kPhaseBit)
    );
}
```

信号量用于协调 TMA 异步加载和生产者-消费者流水线，详见下文使用案例。

---

## 二、Grid 级别（全局内存层）

Grid 级别对应 GPU 的全局内存（HBM），是 TK 三层抽象的最外层。TK 通过 `gl` 布局描述符和 TMA（Tensor Memory Accelerator）硬件单元管理全局内存访问。

### 2.1 全局布局描述符 `gl<T, b, d, r, c, TMA_Types...>`

源码位置：`include/types/global/gl.cuh`

```cpp
template<typename _T, int b, int d, int r, int c, typename... TMA_Types>
struct gl {
    using identifier = ducks::gl::identifier;
    using dtype = T;

    T* raw_ptr;                              // 全局内存裸指针

    // 四维张量尺寸（正数=编译时固定，-1=运行时动态）
    static constexpr int __b__ = b;          // batch
    static constexpr int __d__ = d;          // depth (e.g., num_heads)
    static constexpr int __r__ = r;          // rows (e.g., seq_len)
    static constexpr int __c__ = c;          // cols (e.g., d_model)

    // TMA 描述符字典（编译时从 TMA_Types... 生成）
    detail::descriptor_dict<TMA_Types...> tma_descs;

    // 构造函数（仅在 host 端调用）
    __host__ inline gl(T *_data,
                       make_arg_t<b> _batch,
                       make_arg_t<d> _depth,
                       make_arg_t<r> _rows,
                       make_arg_t<c> _cols);

    // 元素访问（row-major 线性化）
    __device__ inline T& operator[](const coord<> &idx) const {
        return raw_ptr[(((size_t)idx.b*depth() + idx.d)*rows() + idx.r)*cols() + idx.c];
    }

    // 获取 TMA 描述符（设备端调用）
    template<typename U, int axis>
    __device__ inline const CUtensorMap* get_tma() const;
};
```

#### 维度系统

`gl` 采用 4 维张量模型：`[batch, depth, rows, cols]`

- **正数模板参数** → 编译时固定，不占运行时存储
- **`-1` 模板参数** → 运行时动态，存储在 struct 成员中

```cpp
// 示例：batch 和 depth 动态，rows 和 cols 固定
using my_gl = gl<bf16, -1, -1, 128, 64, st_bf<64, 64>>;
// → batch 和 depth 在构造时传入，128 和 64 编译时确定

// 便利别名
gl3<T, d, r, c>  // 3D (batch=1)
gl2<T, r, c>     // 2D (batch=1, depth=1)
gl1<T, c>        // 1D (batch=1, depth=1, rows=1)
```

#### TMA 描述符嵌入

`gl` 的模板参数列表 `TMA_Types...` 指定需要哪些 TMA 描述符：

```cpp
// 示例：为 st_bf<64,64> 和 sv_bf<64> 生成 TMA 描述符
using my_gl = gl<bf16, -1, -1, -1, -1, st_bf<64, 64>, sv_bf<64>>;

// 构造时自动创建 TMA tensor map
my_gl g{ptr, B, D, R, C};
// → 内部调用 cuTensorMapEncodeTiled() 为每种 TMA_Type 创建描述符

// 设备端获取描述符
const CUtensorMap* desc = g.get_tma<st_bf<64,64>, 2>();  // axis=2 (row-major)
```

`descriptor_dict` 是一个编译时递归链表结构：

```cpp
// gl.cuh:82-101 — 递归描述符字典
template<typename _T, typename... Args>
struct descriptor_dict<_T, Args...> {
    CUtensorMap tma_desc;                       // 当前类型的描述符
    descriptor_dict<Args...> other_descs;        // 剩余类型的描述符

    __host__ descriptor_dict(T::dtype *data, int b, int d, int r, int c)
        : other_descs(data, b, d, r, c) {
        // 为当前类型创建 TMA tensor map
        tma::create_tensor_map<T, axis>(&tma_desc, data, b, d, r, c);
    }

    template<typename U, int axis>
    __device__ inline const CUtensorMap* get() const {
        if constexpr (std::is_same_v<T, U> && axis_match)
            return &tma_desc;
        else
            return other_descs.template get<U, axis>();
    }
};
```

### 2.2 TMA（Tensor Memory Accelerator）

TMA 是 Hopper (H100) 引入的硬件单元，可以在 **单线程** 发起指令后，由硬件自动将整个瓦片在全局内存和共享内存之间搬运。

#### TMA Tensor Map 创建

源码位置：`include/types/global/tma.cuh`

```cpp
template<ducks::st::all ST, int axis>
__host__ static inline void create_tensor_map(
    CUtensorMap *tma_map, const typename ST::dtype *src,
    int batch, int depth, int rows, int cols
) {
    // 根据 swizzle 模式映射到 CUDA TMA swizzle 枚举
    constexpr CUtensorMapSwizzle tma_swizzle = ST::swizzle ? (
        ST::swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  :
        ST::swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        ST::swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        CU_TENSOR_MAP_SWIZZLE_NONE
    ) : CU_TENSOR_MAP_SWIZZLE_NONE;

    // 设置全局内存形状和步长
    // 对于 swizzled 模式，将行优先 2D 张量重新编码为 5D TMA 张量
    constexpr int swizzle_elements = ST::swizzle_bytes / sizeof(dtype);
    gmem_shape[0] = swizzle_elements;                          // 子行宽度
    gmem_shape[1] = rows;                                       // 行数
    gmem_shape[2] = (cols + swizzle_elements-1) / swizzle_elements; // 子行数
    gmem_shape[3] = depth;
    gmem_shape[4] = batch;

    // 设置共享内存瓦片形状
    smem_shape[0] = swizzle_elements;                // 子行宽度
    smem_shape[1] = ST::rows;                        // 瓦片行数
    smem_shape[2] = ST::cols / swizzle_elements;     // 瓦片子行数

    // 调用 CUDA Driver API 创建 tensor map
    cuTensorMapEncodeTiled(tma_map, tma_format, tma_dim,
                           global_addr, gmem_shape, gmem_stride,
                           smem_shape, smem_stride,
                           tma_interleave, tma_swizzle,
                           tma_l2Promotion, tma_oobFill);
}
```

**关键洞察**：TK 将共享内存的 swizzle 模式直接编码到 TMA 描述符中，这样 TMA 硬件在搬运数据时就已经完成 swizzle 变换，数据到达共享内存后直接可用，无需额外重排。

#### TMA 异步加载/存储操作

源码位置：`include/ops/group/memory/tile/tma.cuh`

```cpp
// TMA 异步加载：从全局内存到共享内存
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ static inline void load_async(ST &dst, const GL &src,
                                          const COORD &idx, semaphore& bar) {
    if(laneid() == 0) {  // 只需 1 个线程发起！
        ::kittens::tma::load_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx, bar);
    }
}

// TMA 异步存储：从共享内存到全局内存
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD>
__device__ static inline void store_async(const GL &dst, const ST &src, const COORD &idx) {
    if(laneid() == 0) {
        ::kittens::tma::store_async<dim::ROW, cache_policy::NORMAL>(dst, src, idx);
    }
}
```

**TMA 相比传统加载的优势**：
- 传统方式：warp 中 32 个线程各自发起 1 次内存请求 → 32 次请求
- TMA 方式：1 个线程发起 1 条指令 → 硬件自动搬运整个瓦片
- 支持原子归约存储：`store_add_async`, `store_min_async`, `store_max_async`
- 自动处理 swizzle、对齐、越界填充

#### TMA 的 expect 机制

在使用 TMA 加载前，需要告诉信号量预期接收多少字节：

```cpp
// 告诉信号量预期的字节数
tma::expect_bytes(bar, size_bytes<typeof(tile)>);  // 或 tma::expect(bar, tile)

// 发起异步加载
tma::load_async(tile, global, {coords}, bar);

// 等待加载完成
wait(bar, phase_bit);
```

### 2.3 WGMMA 描述符

源码位置：`include/types/shared/descriptor.cuh`

WGMMA（Warpgroup Matrix Multiply-Accumulate）指令需要通过 64 位描述符来定位共享内存中的矩阵数据：

```cpp
template<ducks::st::all ST, int MN_major>
struct st_descriptor {
    uint64_t data;

    // 从共享内存瓦片构造描述符
    __device__ inline void build(const ST &tile, int chunk_idx) {
        uint32_t base_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tile.data[0]));
        // 编码 swizzle 模式、地址、步长到 64 位描述符中
        // bit[0:13]  = base address >> 4
        // bit[14:15] = leading dimension mode
        // bit[16:29] = stride
        // bit[30:31] = swizzle mode (128B=3, 64B=2, 32B=1, none=0)
        data = encode(base_addr, swizzle_mode, stride, chunk_idx);
    }

    // 沿 K 维度前进到下一个 chunk
    __device__ inline st_descriptor chunk_advance(int chunks) const;
};
```

这些描述符由 `warpgroup::mma_AB` 等函数内部自动生成和管理，用户无需直接操作。

### 2.4 超级瓦片（Super Tiling）与持久化网格

源码位置：`include/common/util.cuh:336-372`

TK 提供了 `get_swizzled_2d_idx` 函数实现蛇形遍历（snake ordering），优化 L2 cache 命中率：

```cpp
template <int SUPERGROUP_SIZE, bool ROW_MAJOR = true>
__device__ static inline int2 get_swizzled_2d_idx(
    const int num_rows, const int num_cols, const int linear_idx
) {
    const int supergroup_numel = num_rows * SUPERGROUP_SIZE;
    const int supergroup_idx = linear_idx / supergroup_numel;
    int row_idx = (linear_idx % supergroup_numel) / SUPERGROUP_SIZE;
    int col_idx = supergroup_idx * SUPERGROUP_SIZE + linear_idx % SUPERGROUP_SIZE;
    // 奇数 supergroup 反转行方向（蛇形遍历）
    return { (supergroup_idx & 1) ? num_rows - row_idx - 1 : row_idx, col_idx };
}
```

在持久化网格模式下，132 个 block（H100 SM 数量）循环处理所有瓦片，配合蛇形遍历，相邻瓦片共享 K 维度的 L2 缓存行。

---

## 三、使用案例

### 3.1 案例一：TMA + 双缓冲 GEMM（Level 6）

展示线程块级别的共享内存管理和 Grid 级别的 TMA 数据搬运：

```cpp
// 源码：kernels/gemm/educational_h100/level_06.cu
__global__ void kernel(const __grid_constant__ matmul_globals g) {
    // ===== 线程块级别：共享内存分配 =====
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    // 双缓冲：2 组共享内存瓦片
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&As)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>();
    st_bf<BLOCK_SIZE,BLOCK_SIZE> (&Bs)[2] = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>, 2>();
    int tic = 0, toc = 1;

    // ===== 线程块级别：信号量初始化 =====
    __shared__ semaphore bar;
    if (threadIdx.x == 0) {
        init_semaphore(bar, 0, 1);
        // ===== Grid 级别：TMA 异步加载 =====
        tma::expect_bytes(bar, size_bytes<typeof(As[0])> + size_bytes<typeof(Bs[0])>);
        tma::load_async(As[tic], g.A, {0, 0, row, 0}, bar);  // 全局→共享
        tma::load_async(Bs[tic], g.B, {0, 0, 0, col}, bar);
    }
    __syncthreads();

    for (int tile = 0; tile < num_tiles; ++tile, tic ^= 1, toc ^= 1) {
        // 等待当前瓦片加载完成
        wait(bar, tic);
        __syncthreads();

        // 预加载下一组瓦片（与计算重叠）
        if (threadIdx.x == 0 && tile + 1 < num_tiles) {
            tma::expect_bytes(bar, ...);
            tma::load_async(As[toc], g.A, {0, 0, row, tile+1}, bar);
            tma::load_async(Bs[toc], g.B, {0, 0, tile+1, col}, bar);
        }

        // Warpgroup MMA：直接从共享内存计算
        warpgroup::mma_AB(C_accum, As[tic], Bs[tic]);
        warpgroup::mma_async_wait();
        ...
    }
}
```

数据流：

```
┌──────────┐     TMA          ┌──────────┐     WGMMA      ┌──────────┐
│ 全局内存  │ ──────────────→  │ 共享内存  │ ──────────────→ │ 寄存器   │
│ (HBM)    │  1线程发起指令    │ (SRAM)   │  直接从SRAM读   │          │
│ g.A, g.B │  硬件自动搬运    │ As, Bs   │  无需显式load   │ C_accum  │
└──────────┘  自动swizzle     └──────────┘                 └──────────┘
```

### 3.2 案例二：生产者-消费者 Warp 特化（Level 7）

展示不同 warpgroup 承担不同角色的架构模式：

```cpp
// 源码：kernels/gemm/educational_h100/level_07.cu
static constexpr int NUM_WORKERS = 8; // 2 个 warpgroup

__global__ void kernel(const __grid_constant__ matmul_globals g) {
    const int warpgroupid = kittens::warpid() / 4;

    // 双重信号量系统
    __shared__ semaphore full[QSIZE], empty[QSIZE];

    if (warpgroupid == 0) {
        // ===== 生产者 warpgroup：专注于 TMA 数据搬运 =====
        warpgroup::decrease_registers<32>();   // 释放寄存器给消费者

        if (warpgroup::laneid() == 0) {        // 只需 1 个线程
            for (int tile = 0; tile < num_tiles; ++tile) {
                wait(empty[qidx], p);           // 等待消费者释放缓冲区
                tma::expect_bytes(full[qidx], ...);
                tma::load_async(As[qidx], g.A, ..., full[qidx]);
                tma::load_async(Bs[qidx], g.B, ..., full[qidx]);
            }
        }
    } else {
        // ===== 消费者 warpgroup：专注于 Tensor Core 计算 =====
        warpgroup::increase_registers<256>();   // 获取更多寄存器

        rt_fl<16, BLOCK_SIZE> C_accum;
        kittens::warp::zero(C_accum);

        for (int tile = 0; tile < num_tiles; ++tile) {
            wait(full[qidx], p);               // 等待数据就绪
            warpgroup::mma_AB(C_accum, As[qidx], Bs[qidx]);
            warpgroup::mma_async_wait();
            arrive(empty[qidx], 1);            // 通知生产者缓冲区可复用
        }
        warpgroup::store(g.C, C_accum, {0, 0, row, col});
    }
}
```

协调图：

```
时间 →

生产者 warpgroup:
  [加载 tile0] → [加载 tile1] → [加载 tile2] → ...
     ↓ full[0]      ↓ full[1]      ↓ full[2]
消费者 warpgroup:
            → [计算 tile0] → [计算 tile1] → [计算 tile2]
              ↓ empty[0]     ↓ empty[1]     ↓ empty[2]
                    ↑              ↑              ↑
               释放缓冲区     释放缓冲区     释放缓冲区
```

### 3.3 案例三：LayerNorm（共享向量操作）

展示共享向量（`sv`）在非 MMA 场景下的使用：

```cpp
// 源码：kernels/layernorm/layernorm.cu
template<int D>
__global__ void layernorm_tk(const __grid_constant__ norm_globals<D> g, int n_per_tile) {
    auto warpid = kittens::warpid();

    // ===== 共享内存分配 =====
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    sv_bf<d_model> (&x_s)[2][NUM_WORKERS] = al.allocate<sv_bf<d_model>, 2, NUM_WORKERS>();
    sv_bf<d_model> (&residual_s)[2][NUM_WORKERS] = al.allocate<sv_bf<d_model>, 2, NUM_WORKERS>();
    sv_bf<d_model> &norm_weight_s = al.allocate<sv_bf<d_model>>();
    sv_bf<d_model> &norm_bias_s   = al.allocate<sv_bf<d_model>>();

    // 加载 norm 参数（从全局内存到共享内存）
    warp::load(norm_weight_s, g.norm_weight, {0,0,0,0});
    warp::load(norm_bias_s,   g.norm_bias,   {0,0,0,0});

    // 预取第一批数据
    warp::load_async(x_s[warpid][tic], g.x, {batch, 0, seq_start+warpid, 0});
    warp::load_async(residual_s[warpid][tic], g.residual, {batch, 0, seq_start+warpid, 0});

    for (int block = 0; block < n_blocks; block++, tic ^= 1, toc ^= 1) {
        // 预取下一批
        if (block < n_blocks - 1) {
            warp::load_async(x_s[warpid][toc], g.x, {batch, 0, seq_start+next_idx, 0});
        }
        load_async_wait();

        // ===== 共享向量操作 =====
        warp::add(residual_s[...], residual_s[...], x_s[...]);      // residual += x
        warp::sum(mean, residual_s[...]);                             // mean = Σ residual
        mean = mean / __float2bfloat16(d_model);
        warp::sub(residual_s[...], residual_s[...], mean);           // residual -= mean
        warp::mul(x_s[...], residual_s[...], residual_s[...]);       // x = residual²
        warp::sum(var, x_s[...]);                                     // var = Σ residual²
        var = var / __float2bfloat16(d_model);
        var = __float2bfloat16(sqrt(__bfloat162float(var + __float2bfloat16(1e-05f))));

        warp::div(residual_s[...], residual_s[...], var);            // normalize
        warp::mul(residual_s[...], residual_s[...], norm_weight_s);  // scale
        warp::add(residual_s[...], residual_s[...], norm_bias_s);    // shift

        // 写回全局内存
        warp::store(g.o, residual_s[...], {batch, 0, seq_start+cur_idx, 0});
    }
}
```

### 3.4 案例四：Attention 中的三层协同

在 FlashAttention 前向内核中，三层同时工作：

```cpp
// 源码：kernels/attention/mha_h100/mha_h100.cu
// 4 个 warpgroup = 512 线程

// ===== Grid 级别 =====
// gl 描述符定义 4D 张量布局
using q_gl = gl<bf16, -1, -1, -1, -1, st_bf<qo_height, tile_width>>;
// TMA 描述符自动嵌入，支持异步加载

// ===== 生产者 warpgroup (warpgroup 3) =====
// TMA 异步加载 K, V 到共享内存
tma::expect(k_smem_arrived[...], k_smem[...]);
tma::load_async(k_smem[...], g.k, {batch, head, kv_block, 0}, k_smem_arrived[...]);
tma::load_async(v_smem[...], g.v, {batch, head, kv_block, 0}, v_smem_arrived[...]);

// ===== 线程块级别 =====
// 共享内存中的 Q, K, V 瓦片
// q_smem[3]: 3 个消费者各自的 Q 瓦片
// k_smem[stages], v_smem[stages]: 流水线缓冲区

// ===== 消费者 warpgroup (warpgroup 0-2) =====
// 直接从共享内存做 WGMMA
wait(k_smem_arrived[...]);                          // 等待 K 加载完
warpgroup::mm_ABt(att_block, q_smem[wg], k_smem[...]); // S = Q × K^T

// Warp 级别（寄存器层）的 online softmax
warp::row_max(max_vec, att_block, max_vec);          // 行最大值
warp::exp2(att_block, att_block);                    // 逐元素 exp2
warp::row_sum(norm_vec, att_block, norm_vec);        // 行和

// 再次 WGMMA：O += P × V
wait(v_smem_arrived[...]);
warpgroup::mma_AB(o_reg, att_block_mma, v_smem[...]); // O += P × V

// 最终从寄存器写回全局内存
warp::div_row(o_reg, o_reg, norm_vec);               // O /= l
warpgroup::store(g.o, o_reg, {batch, head, q_block, 0});
```

---

## 四、三层协同总结

```
┌──────────────────────────────────────────────────────────┐
│                    Grid 级别（全局内存）                   │
│  gl<T, b, d, r, c>: 4D 张量描述符                        │
│  TMA: 硬件异步搬运，1 线程发起，自动 swizzle              │
│  持久化网格 + 超级瓦片: 优化 L2 缓存命中                  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │            线程块级别（共享内存）                    │    │
│  │  st<T,R,C>: swizzled 瓦片，自动消除 bank conflict   │    │
│  │  sv<T,L>: 共享向量                                 │    │
│  │  shared_allocator: 对齐分配                         │    │
│  │  semaphore: TMA/WGMMA 同步                         │    │
│  │  生产者-消费者 warp 特化                             │    │
│  │                                                    │    │
│  │  ┌──────────────────────────────────────────┐     │    │
│  │  │          Warp 级别（寄存器）               │     │    │
│  │  │  rt<T,R,C>: 分布式寄存器瓦片              │     │    │
│  │  │  rv<T,L>: 寄存器向量                      │     │    │
│  │  │  mma_AB/ABt: Tensor Core MMA             │     │    │
│  │  │  row_max, exp2, row_sum: 纯寄存器操作      │     │    │
│  │  └──────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## 附录：源码文件索引

| 文件 | 内容 |
|------|------|
| `include/types/shared/st.cuh` | 共享瓦片结构体、swizzle idx()、子瓦片 |
| `include/types/shared/sv.cuh` | 共享向量结构体 |
| `include/types/global/gl.cuh` | 全局布局描述符、TMA 描述符字典 |
| `include/types/global/tma.cuh` | TMA tensor map 创建（cuTensorMapEncodeTiled） |
| `include/types/shared/descriptor.cuh` | WGMMA 共享内存描述符 |
| `include/common/util.cuh` | shared_allocator、tma_allocator、超级瓦片 |
| `include/ops/group/shared/tile/maps.cuh` | 共享瓦片逐元素/行/列操作 |
| `include/ops/group/shared/tile/reductions.cuh` | 共享瓦片行/列规约 |
| `include/ops/group/shared/vec/maps.cuh` | 共享向量逐元素操作 |
| `include/ops/group/util/sync.cuh` | 信号量 init/arrive/wait |
| `include/ops/group/memory/tile/tma.cuh` | TMA load_async/store_async 包装 |
| `kernels/gemm/educational_h100/level_06.cu` | TMA + 双缓冲 GEMM 示例 |
| `kernels/gemm/educational_h100/level_07.cu` | 生产者-消费者 warp 特化示例 |
| `kernels/layernorm/layernorm.cu` | 共享向量操作 LayerNorm 示例 |
| `kernels/attention/mha_h100/mha_h100.cu` | 三层协同 FlashAttention 示例 |
