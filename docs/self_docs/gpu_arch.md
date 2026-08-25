```mermaid
flowchart TB

    %% =========================================================
    %% GPU CHIP LEVEL
    %% =========================================================

    GPU["GPU 芯片"]

    GPC1["GPC 1<br/>Graphics Processing Cluster"]
    GPC2["GPC 2"]
    GPCN["GPC N"]

    L2["L2 Cache<br/>全 GPU 共享"]
    MC["Memory Controller<br/>HBM / GDDR 接口"]
    CE["Copy Engine<br/>独立 DMA 引擎"]
    NV["NVLink / NVSwitch Interface<br/>GPU 间高速互联"]

    GPU --> GPC1
    GPU --> GPC2
    GPU --> GPCN

    GPU --> L2
    GPU --> MC
    GPU --> CE
    GPU --> NV


    %% =========================================================
    %% GPC -> TPC -> SM
    %% =========================================================

    TPC1["TPC 1<br/>Texture Processing Cluster"]
    TPC2["TPC 2"]
    TPCN["TPC ..."]

    GPC1 --> TPC1
    GPC1 --> TPC2
    GPC1 --> TPCN

    SM1["SM 1<br/>Streaming Multiprocessor"]
    SM2["SM 2"]

    TPC1 --> SM1
    TPC1 --> SM2


    %% =========================================================
    %% SM HARDWARE RESOURCES
    %% =========================================================

    subgraph SMHW["SM 内部：硬件资源"]
        direction LR

        PBS["Processing Blocks / Partitions<br/>若干执行分区"]

        WS["Warp Scheduler<br/>Warp 调度器"]
        DISP["Dispatch Unit<br/>指令派发"]

        CUDA["CUDA Cores<br/>FP32 / INT 等标量计算"]
        TC["Tensor Cores<br/>矩阵乘加"]
        LSU["Load / Store Units<br/>访存单元"]
        SFU["SFU<br/>Special Function Units"]
        TMA["TMA<br/>Tensor Memory Accelerator<br/>Hopper+"]

        RF["Register File<br/>寄存器文件"]
        L1SMEM["L1 Cache / Shared Memory<br/>片上高速存储"]

        PBS --> WS
        WS --> DISP

        DISP --> CUDA
        DISP --> TC
        DISP --> LSU
        DISP --> SFU

        LSU --> TMA

        CUDA --- RF
        TC --- RF
        LSU --- RF

        LSU --- L1SMEM
        TMA --- L1SMEM
    end

    SM1 --> PBS


    %% =========================================================
    %% SOFTWARE / EXECUTION HIERARCHY
    %% =========================================================

    subgraph EXEC["CUDA 执行 / 调度层级"]
        direction TB

        GRID["Grid<br/>一次 Kernel Launch"]

        BLOCK["Thread Block / CTA<br/>一个 Block 只驻留在一个 SM"]

        WARP["Warp<br/>32 Threads<br/>NVIDIA 基本调度单位"]

        THREAD["Thread<br/>单个 CUDA 线程"]

        GRID --> BLOCK
        BLOCK --> WARP
        WARP --> THREAD
    end


    %% =========================================================
    %% MAPPING EXECUTION TO HARDWARE
    %% =========================================================

    BLOCK -. "调度到某个 SM<br/>一个 SM 可同时驻留多个 Block" .-> SM1
    BLOCK -.-> SM2

    WARP -. "由 Warp Scheduler 选择并发射" .-> WS


    %% =========================================================
    %% HOPPER+ WARP GROUP
    %% =========================================================

    WG["Warp Group<br/>4 Warps = 128 Threads<br/>Hopper WGMMA 等场景"]

    WARP -. "特定指令/编程模型下<br/>4 个 Warp 组成" .-> WG
    WG -. "例如 WGMMA / Warp Specialization" .-> TC


    %% =========================================================
    %% MEMORY PATH
    %% =========================================================

    L1SMEM -. "Cache miss / 全局访存" .-> L2
    L2 --> MC


    %% =========================================================
    %% STYLE
    %% =========================================================

    style GPU fill:#efa4e8,stroke:#666,color:#111

    style GPC1 fill:#e7b0ea,stroke:#666,color:#111
    style GPC2 fill:#e7b0ea,stroke:#666,color:#111
    style GPCN fill:#e7b0ea,stroke:#666,color:#111

    style TPC1 fill:#cfcdf5,stroke:#666,color:#111
    style TPC2 fill:#cfcdf5,stroke:#666,color:#111
    style TPCN fill:#cfcdf5,stroke:#666,color:#111

    style SM1 fill:#bfc5f3,stroke:#666,color:#111
    style SM2 fill:#bfc5f3,stroke:#666,color:#111

    style L2 fill:#f4d5a3,stroke:#666,color:#111
    style MC fill:#f4d5a3,stroke:#666,color:#111
    style CE fill:#f4d5a3,stroke:#666,color:#111
    style NV fill:#f4d5a3,stroke:#666,color:#111

    style PBS fill:#d6ecce,stroke:#666,color:#111
    style WS fill:#d6ecce,stroke:#666,color:#111
    style DISP fill:#d6ecce,stroke:#666,color:#111

    style CUDA fill:#f3dbb5,stroke:#666,color:#111
    style TC fill:#f3dbb5,stroke:#666,color:#111
    style LSU fill:#c9e8c7,stroke:#666,color:#111
    style SFU fill:#c9e8c7,stroke:#666,color:#111
    style TMA fill:#c9e8c7,stroke:#666,color:#111

    style RF fill:#cce7d5,stroke:#666,color:#111
    style L1SMEM fill:#cce7d5,stroke:#666,color:#111

    style GRID fill:#ddd9f4,stroke:#666,color:#111
    style BLOCK fill:#ddd9f4,stroke:#666,color:#111
    style WARP fill:#ddd9f4,stroke:#666,color:#111
    style THREAD fill:#ddd9f4,stroke:#666,color:#111

    style WG fill:#bde7b8,stroke:#666,color:#111