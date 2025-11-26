# Cuda Heirarchy

┌─────────────────────────────────────────────────────────────┐
│  CUDA Context (software concept)                            │
│  - Your "session" with the GPU                              │
│  - Holds memory allocations, loaded modules, streams        │
│  - One per process typically                                │
└─────────────────────────┬───────────────────────────────────┘
                          │ launches
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Grid (software concept)                                    │
│  - One kernel launch = one grid                             │
│  - Contains all blocks for this launch                      │
│  ┌─────────┬─────────┬─────────┬─────────┐                  │
│  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ ...              │
│  └─────────┴─────────┴─────────┴─────────┘                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Block (maps to SM - Streaming Multiprocessor)              │
│  - Threads that can share memory & synchronize              │
│  - Max 1024 threads per block                               │
│  ┌───────────────────────────────────────┐                  │
│  │  Warp 0   │  Warp 1   │  Warp 2  │... │                  │
│  │ (32 thds) │ (32 thds) │ (32 thds)│    │                  │
│  └───────────────────────────────────────┘                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Warp (hardware execution unit)                             │
│  - 32 threads that execute in LOCKSTEP                      │
│  - All 32 run same instruction at same time (SIMT)          │
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐          │
│  │T0│T1│T2│T3│T4│T5│T6│T7│...                 │T31│          │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘          │
└─────────────────────────────────────────────────────────────┘