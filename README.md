# Research Interests: How Execution Environment Forks the Scheduling & Design Space

> 🇰🇷 한국어 원문: [README.ko.md](README.ko.md)

> Research interests compiled by a first-year (2nd semester) undergraduate. I'm still learning, so some parts may be wrong.
> **Criticism, questions, corrections, and feedback are all welcome.** Feel free to open an issue or PR.

## 1. Core Problem Statement

Even for the same "LLM inference," the scheduling strategy and hardware design philosophy diverge fundamentally depending on **where you run it.**

| | Server (multi-request) | Edge / SoC (single-request) |
|---|---|---|
| Scheduler goal | Maximize throughput, power efficiency, goodput within SLO | Minimize latency, maximize power efficiency |
| Cache behavior | Hard to reuse as requests intermix (mitigated by prefix caching) | Scratchpad-like — a single request monopolizes the cache → very high hit rate |
| Bottleneck | HBM bandwidth (KV streaming), interconnect (NVLink, CXL) | Bus transactions (memory ↔ compute round-trips), on-chip memory capacity |
| Design direction | Secure arithmetic intensity via large batches | Eliminate bus transactions, maximize on-chip data reuse |

This divergence is quantified by the **Roofline model**. The x-axis is arithmetic intensity (ops/byte), the y-axis is attainable performance.
- The server pushes intensity to the right with large batches, trying to enter the **compute-bound** region (sloped part → flat part of the roofline).
- Edge single-request (especially decode) is inherently low-intensity, staying in the **memory-bound** region → performance is tied directly to memory bandwidth and bus transactions.

To tackle the server-scale scheduling problem, I worked on **PULS (PIM-Unified LLM Serving)**, combined with PIM.

---

## 2. Core Claim: Bus Transactions Are Decisive

In a low-power, high-efficiency chip, **minimizing the number of bus transactions** matters more than compute-unit efficiency.

- Bus-transaction energy >> compute energy
- On an SoC with limited on-chip memory, even a slight increase in data size triggers repeated off-chip memory references
- In a single-request environment the access pattern is predictable → the scheduler has ample room to optimize prefetch and reuse order

#### Evidence: Compute vs. Memory-Access Energy (45nm, 0.9V)

Based on Horowitz's measurements [1] — memory-access energy dominates compute energy.

| Operation / Access | Energy | Ratio vs. DRAM access (1.3–2.6 nJ) |
|---|---|---|
| 8-bit Int Add | 0.03 pJ | ~43,000–87,000× |
| 32-bit FP Add | 0.9 pJ | ~1,400–2,900× |
| 32-bit FP Mult | 3.7 pJ | ~350–700× |
| 32-bit FP MAC (Add+Mult) | ~4.6 pJ | ~280–560× |
| 1MB cache access | 100 pJ | (~111× vs. FP Add) |

- Even by **FP MAC**, the core operation in AI workloads, **a DRAM access is ~300–600× more expensive** (figures based on Fig. 1.1.9 [1])

---

## 3. Connection to PULS

PULS (PIM-Unified LLM Serving): prior work that validated the same problem at server scale.

- PIM attention processes KV data in-place → eliminates the GPU ↔ HBM KV-streaming traffic itself
  - Result: `Aux2: 79.8% reduction in bus traffic, 4.95× speedup`
- Mixed batching lets prefill/decode tokens share FFN weights → reduces per-token bus transactions
- Demonstrates that even in the server environment, reducing bus transactions is decisive for both performance and energy
- The principle applies even more extremely on edge/SoC — because on-chip SRAM can be managed directly as a scratchpad, and the compiler/scheduler can statically decide when data moves

### Supporting Case: FlashAttention — SW Only, No HW Change

If PULS is a full-stack approach that changed both HW and SW, **FlashAttention is the representative case of reducing transactions on the same HW using SW scheduling/tiling alone.**

- Splits attention into tiles processed inside SRAM → eliminates the round-trip of writing the huge N×N attention matrix to HBM and reading it back
- The algorithm (FLOPs) is unchanged, yet reducing only HBM transactions improves both speed and memory
- **Implication**: a GPU-side empirical demonstration of this document's claim — "bus transactions are decisive, and much of the optimization is the SW's job"

---

## 4. Directions to Explore

1. **Static scheduling** — when the access pattern is predictable, how can the compiler decide the data placement and operation order that minimizes bus transactions?
2. **Cache vs. scratchpad design choice** — should the HW manage hit rate automatically, or should SW explicitly manage on-chip memory? Does the regular access pattern of LLM inference favor the latter?
3. **SW-HW scheduling interface** — like PULS's interceptor (occupying JEDEC RFU bits), what is the minimal interface that lets a SW scheduler finely control HW behavior?
4. **Thermal-aware scheduling** — fewer bus transactions → less heat (PULS's anticipated incidental heat reduction). How should a scheduling policy change when it targets both energy savings and heat reduction simultaneously?

---

## 5. The Importance of SW-HW Co-Design

### The Generality Constraint on Hardware

Hardware must retain some degree of generality even if it only supports specific operations.

- For an AI chip, the only realistic answer is ultimately **integrating MAC (Multiply-Accumulate) units** — even an ASIC is built on general MAC operations
- HW-level efficiency tricks all operate within the constraint of generality:
  - **Clock gating / power gating** — cut the clock/power to unused circuit blocks to remove wasted power consumption
  - **Shortening physical distance between modules (Apple M-series)** — integrating CPU·GPU·Neural Engine·DRAM into one package (Unified Memory Architecture). Without a PCIe bus or external VRAM, all compute units share the same memory pool → this lowers not the transaction *count* but the **latency/energy per transaction**. However, server GPU clusters have limits on physical integration, so this is a structural advantage unique to edge SoCs

### Generality Offloads the Schedule onto SW

The core logic is a **trade-off between specialization and who decides the schedule.**

- A **fully specialized ASIC** can **statically bake** the data-movement order (dataflow) into the circuit — the schedule *is* the hardware. But it can only run that one workload.
- A **general-purpose chip** cannot fix dataflow into the circuit because it can't know in advance which operation will come → **the decision of when to process what, in what order, inevitably falls to SW (scheduler/compiler).**
- That is, the moment you preserve generality, the responsibility of minimizing transactions falls on SW, not HW. **This is exactly why the scheduler directly determines efficiency and performance in a general-purpose low-power chip.**

So the HW merely provides MAC compute capability and a memory hierarchy; the bus-transaction count, timing, and data-reuse order are the SW's domain, which HW alone cannot optimize. On an edge SoC, the two ultimately operate together — **the SW scheduler minimizing transaction count** + **HW integration minimizing per-transaction energy.**

---

## 6. Extension: Device–Edge-Server Collaborative Inference & Communication-Aware Scheduling

### Problem Statement

This addresses a **device ↔ edge-server** 2-tier structure.

- To make a model more sophisticated you must grow the parameters, but processing all computation inside the device has clear limits in power, heat, and memory
- The device handles I/O (display·sensors) and lightweight computation
- Heavy computation is handled by a physically nearby small high-performance edge server, which returns the result over the network
- The edge server, with sufficient cooling and a high-performance AI chip, computes overwhelmingly faster than the device
- **Heat relocation is the key benefit**: if the device runs a large model directly, the compute heat concentrates on the poorly-cooled device, causing throttling. Offloading moves the compute heat to the well-cooled edge server, leaving only communication heat on the device

### But the "Tier's Existence" Is Not New

This structure itself is already actively studied in academia and industry:
- **Neurosurgeon** [2] — partitions a DNN across device↔cloud based on latency/energy
- **Split computing / early exiting** survey [3]
- **Apple Private Cloud Compute** [4] — on-device small model + private cloud offload
- **Gemini Nano hybrid inference** [5] — on-device + cloud

Simply splitting it as "device = display, edge server = compute" is nothing more than a thin client + remote inference. To be a research contribution, the **scheduling policy that decides what to process where and when** must be the core.

### Research Contribution: Partition Scheduling That Moves Heat to the Edge Server

The key is **where the heat is generated.**

- Device computes directly → heat concentrates on the poorly-cooled device → throttling
- Offload to edge server → that heat relocates to the well-cooled edge server

| | Device computes directly | Offload to edge server |
|---|---|---|
| **Compute heat location** | Device (poor cooling → throttling) | Edge server (sufficient cooling) |
| **Communication heat location** | None | Device — text-token send/receive, small (receiving the result is even smaller) |
| **Total device heat** | **Large** | **Small** |

- LLM output is text, so the data exchanged is only a few KB → communication heat is small, while what truly dominates heat is computation (each decode reads billions of parameters from memory)
- That is, the core benefit of this structure is **moving only the high-heat computation to the edge server, leaving only the low-heat communication on the device**
- However, wireless transmission has a higher per-bit energy than local computation [6], so you cannot increase communication volume indiscriminately
- Therefore the core contribution is a scheduler that finds the **partition point that offloads heavy computation while minimizing communication volume** — a dynamic partition policy that aims to beat both pure on-device and pure offload extremes
- LLM-specific partition units:
  - Where to place the KV cache, device or edge server
  - Prefill on the high-performance edge server, part of decode on the device
  - The draft of speculative decoding on the device, verify on the edge server
- **Energy · latency · privacy three-axis trade-off**: reducing communication improves energy and privacy but gives up the sophistication of a large model. The scheduler aims to solve this balance dynamically.

### Project Setup (Experimental)

- **Module A — high-performance edge-server chip for AI compute**: assumes sufficient cooling, handles heavy computation
- **Module B — device-side receive·display module**: lightweight computation + result reception·rendering
- **The core deliverable is not the two modules, but the communication-aware partition scheduler that divides them** — the two modules are the experimental environment that validates that policy

**Hypothesis**: In device–edge-server inference, what dominates the bottleneck and energy is not computation but the **communication transactions between device ↔ edge server**, and a scheduler that explicitly models communication volume·energy is expected to outperform both the pure on-device and pure offload extremes.

---

## 7. Conclusion

One principle runs through it all — **transaction energy ≫ compute energy**, and the responsibility for reducing those transactions ultimately lies with the SW scheduler.

- **Inside the chip**: what determines efficiency and performance is not the compute unit's FLOPS but **how much SW-HW co-design reduces bus transactions.** We must move from a structure where HW "hopes" for cache hits to one where the SW scheduler statically predicts the access pattern and "guarantees" hits.
- **Between device and edge server**: the same principle repeats one layer up. Heavy computation (and its heat) is offloaded to the well-cooled edge server, while the SW scheduler must find the partition point that minimizes the per-bit-expensive communication.

In other words, whether inside the chip or between device and edge server, **the ultimate competitiveness of low-power, high-efficiency AI rests on scheduling that reduces transactions.**

---

## References

[1] M. Horowitz, "Computing's Energy Problem (and What We Can Do About It)," *ISSCC 2014*, pp. 10–14. (Fig. 1.1.9: Rough energy costs for various operations in 45nm 0.9V) https://ieeexplore.ieee.org/document/6757323

[2] Y. Kang et al., "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge," *ASPLOS 2017*, pp. 615–629. https://dl.acm.org/doi/10.1145/3037697.3037698

[3] Y. Matsubara, M. Levorato, F. Restuccia, "Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges," *ACM Computing Surveys*, 55(5), 2022. https://arxiv.org/abs/2103.04505

[4] Apple Security Engineering and Architecture, "Private Cloud Compute: A new frontier for AI privacy in the cloud," 2024. https://security.apple.com/blog/private-cloud-compute/

[5] Android Developers Blog, "Experimental hybrid inference and new Gemini models for Android." https://developer.android.com/blog/posts/experimental-hybrid-inference-and-new-gemini-models-for-android

[6] V. Raghunathan, C. Schurgers, S. Park, M. Srivastava, B. Shaw, "Energy-Aware Wireless Microsensor Networks," *IEEE Signal Processing Magazine*, 19(2), pp. 40–50, 2002. https://ieeexplore.ieee.org/document/985679
