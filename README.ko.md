# 연구 관심사: 실행 환경 차이가 만드는 스케줄링·설계 공간의 분기

> 🇬🇧 English: [README.md](README.md)

> 학부 1학년 2학기 학생이 정리한 연구 관심사입니다. 아직 배우는 단계라 틀린 부분이 있을 수 있습니다.
> **비판·질문·가르침·피드백 모두 환영합니다.** 이슈나 PR로 자유롭게 남겨주세요.

## 1. 핵심 문제의식

같은 LLM inference라도 **어디서 돌리느냐**에 따라 스케줄링 전략과 하드웨어 설계 철학이 근본적으로 달라진다.

| | 서버 (멀티-리퀘스트) | 엣지 / SoC (싱글-리퀘스트) |
|---|---|---|
| 스케줄러 목표 | throughput 최대화, 전력 효율, SLO 내 goodput 최대화 | latency 최소화, 전력 효율 최대화 |
| 캐시 동작 | 요청이 뒤섞여 재사용 어려움 (prefix caching으로 보완) | 스크래치패드 유사 — 단일 요청이 캐시 독점 → 히트율 매우 높음 |
| 병목 | HBM 대역폭 (KV streaming), interconnect (NVLink, CXL) | 버스 트랜잭션 (메모리 ↔ 연산기 왕복), 온칩 메모리 용량 |
| 설계 방향 | 대규모 배치로 arithmetic intensity 확보 | 버스 트랜잭션 제거, 온칩 데이터 재사용 극대화 |

이 분기는 **Roofline 모델**로 정량화된다. 가로축은 arithmetic intensity(연산/바이트), 세로축은 달성 성능이다.
- 서버는 큰 배치로 intensity를 오른쪽으로 밀어 **compute-bound** 영역(roofline의 경사진 부분 → 평평한 부분)으로 진입시키려 한다.
- 엣지의 싱글-리퀘스트(특히 decode)는 본질적으로 intensity가 낮아 **memory-bound** 영역에 머문다 → 성능이 메모리 대역폭과 버스 트랜잭션에 직접 묶인다.

서버 스케일 스케줄링 문제를 해결하기 위해 PIM과 결합한 **PULS (PIM-Unified LLM Serving)** 프로젝트를 진행했다.

---

## 2. 핵심 주장: 버스 트랜잭션이 결정적이다

저전력 고효율 칩에서 연산기 효율보다 **버스 트랜잭션 횟수 최소화**가 더 중요하다.

- 버스 트랜잭션 에너지 >> 연산 에너지
- 온칩 메모리가 제한된 SoC에서는 데이터가 조금만 커져도 오프칩 메모리를 반복 참조
- 싱글-리퀘스트 환경은 접근 패턴이 예측 가능 → 스케줄러가 프리페치·재사용 순서를 최적화할 여지가 큼

#### 근거: 연산 vs 메모리 접근 에너지 (45nm, 0.9V)

Horowitz [1]의 측정값 기준 — 메모리 접근 에너지가 연산 에너지를 압도한다.

| 연산 / 접근 | 에너지 | DRAM 접근 (1.3–2.6 nJ) 대비 배수 |
|---|---|---|
| 8-bit Int Add | 0.03 pJ | ~43,000–87,000× |
| 32-bit FP Add | 0.9 pJ | ~1,400–2,900× |
| 32-bit FP Mult | 3.7 pJ | ~350–700× |
| 32-bit FP MAC (Add+Mult) | ~4.6 pJ | ~280–560× |
| 1MB 캐시 접근 | 100 pJ | (FP Add 대비 ~111×) |

- AI 워크로드의 핵심 연산인 **FP MAC 기준으로도 DRAM 접근은 약 300–600배** 더 비싸다 (수치는 Fig. 1.1.9 [1] 기준)

---

## 3. PULS와의 연결점

PULS (PIM-Unified LLM Serving): 같은 문제를 서버 스케일에서 검증한 선행 연구.

- PIM attention이 KV 데이터를 in-place 처리 → GPU ↔ HBM KV streaming 트래픽 자체 제거
  - 결과: `Aux2: 79.8% 버스 트래픽 감소, 4.95× speedup`
- mixed batching으로 FFN 가중치를 prefill/decode 토큰이 공유 → per-token 버스 트랜잭션 감소
- 서버 환경에서도 버스 트랜잭션 감소가 성능·에너지 모두에 결정적임을 실증
- 엣지/SoC에서는 이 원리가 더 극단적으로 적용 가능 — 온칩 SRAM을 스크래치패드로 직접 관리하고 컴파일러/스케줄러가 데이터 이동 시점을 정적으로 결정할 수 있기 때문

### 보강 사례: FlashAttention — HW 변경 없이 SW만으로

PULS가 HW와 SW를 함께 바꾼 full-stack 접근이라면, **FlashAttention은 동일 HW에서 SW 스케줄링/타일링만으로** 트랜잭션을 줄인 대표 사례다.

- attention을 타일 단위로 쪼개 SRAM 안에서 처리 → 거대한 N×N attention 행렬을 HBM에 쓰고 다시 읽는 왕복을 제거
- 알고리즘(FLOPs)은 그대로인데 HBM 트랜잭션만 줄여 속도·메모리가 동시에 개선됨
- **시사점**: "버스 트랜잭션이 결정적이며, 그 최적화는 상당 부분 SW의 몫"이라는 본 문서의 주장을 GPU 상에서 실증한 사례

---

## 4. 탐구 방향

1. **정적 스케줄링** — 접근 패턴 예측이 가능할 때, 컴파일러가 버스 트랜잭션을 최소화하는 데이터 배치·연산 순서를 어떻게 결정할 수 있는가
2. **캐시 vs 스크래치패드 설계 선택** — HW가 캐시 히트율을 자동 관리할 것인가, SW가 온칩 메모리를 명시적으로 관리할 것인가. LLM inference의 규칙적 접근 패턴은 후자에 유리한가
3. **SW-HW 스케줄링 인터페이스** — PULS의 interceptor (JEDEC RFU bit 점유)처럼, SW 스케줄러가 HW를 세밀하게 제어할 수 있는 최소한의 인터페이스는 무엇인가
4. **thermal-aware 스케줄링** — 버스 트랜잭션 감소 → 발열 감소 (PULS의 부수적 발열 감소 예상). 에너지 절감과 발열 감소를 동시에 목표로 하는 스케줄링 정책은 어떻게 달라져야 하는가

---

## 5. SW-HW Co-Design의 중요성

### 하드웨어의 범용성 제약

하드웨어는 특정 연산만 지원하더라도 어느 정도의 범용성을 반드시 유지해야 한다.

- AI 칩의 경우 결국 **MAC(Multiply-Accumulate) 연산의 집적**이 유일한 현실적 답이다 — ASIC조차 범용 MAC 연산을 기반으로 한다
- HW 레벨의 효율 트릭들은 모두 범용성이라는 제약 안에서만 이루어진다:
  - **클럭 게이팅 / 파워 게이팅** — 미사용 회로 블록의 클럭·전원을 차단해 불필요한 전력 소비 제거
  - **모듈 간 물리적 거리 단축 (Apple M 시리즈)** — CPU·GPU·Neural Engine·DRAM을 하나의 패키지에 집적 (Unified Memory Architecture). PCIe 버스·외장 VRAM 없이 동일 메모리 풀 공유 → 트랜잭션 횟수가 아니라 **트랜잭션 1회당 레이턴시·에너지**를 낮춤. 단, 서버 GPU 클러스터는 물리적 집적에 한계가 있어 엣지 SoC만의 구조적 이점

### 범용성이 스케줄을 SW로 떠넘긴다

핵심 논리는 **특화도와 스케줄 결정 주체의 트레이드오프**다.

- **완전 특화 ASIC**은 데이터 이동 순서(dataflow)를 회로에 **정적으로 박아넣을** 수 있다 — 스케줄이 곧 하드웨어다. 대신 그 워크로드만 돌릴 수 있다.
- **범용 칩**은 어떤 연산이 올지 미리 알 수 없으므로 dataflow를 회로에 고정할 수 없다 → **언제 무엇을 어떤 순서로 처리할지의 결정이 필연적으로 SW(스케줄러/컴파일러)로 넘어온다.**
- 즉 범용성을 지키는 순간, 트랜잭션 최소화의 책임은 HW가 아니라 SW가 진다. **범용 저전력 칩에서 스케줄러가 효율·성능에 직결되는 이유가 바로 이것이다.**

따라서 HW는 MAC 연산 능력과 메모리 계층을 제공할 뿐이고, 버스 트랜잭션 횟수·타이밍·데이터 재사용 순서는 HW만으로 최적화 불가능한 SW의 영역이다. 엣지 SoC에서는 결국 둘이 동시에 작동한다 — **SW 스케줄러의 트랜잭션 횟수 최소화** + **HW 집적에 의한 트랜잭션 1회당 에너지 최소화**.

---

## 6. 확장 방향: 디바이스–엣지서버 협력 추론과 통신-aware 스케줄링

### 문제의식

**디바이스 ↔ 엣지서버** 2-티어 구조를 다룬다.

- 모델이 정교해지려면 파라미터를 키워야 하는데, 모든 연산을 디바이스 안에서 처리하는 것은 전력·발열·메모리 측면에서 한계가 명확하다
- 디바이스는 입출력(디스플레이·센서)과 경량 연산을 담당
- 무거운 연산은 물리적으로 가까운 소규모 고성능 엣지서버가 맡아 결과를 통신으로 반환
- 엣지서버는 충분한 쿨링과 고성능 AI 칩을 갖춰 디바이스보다 압도적으로 빠르게 연산
- **발열 이전이 핵심 이점**: 큰 모델을 디바이스가 직접 돌리면 연산 발열이 쿨링 빈약한 디바이스에 집중돼 쓰로틀링을 유발한다. 오프로드하면 연산 발열을 쿨링이 충분한 엣지서버로 옮기고, 디바이스에는 통신 발열만 남는다

### 그러나 "티어의 존재"는 새롭지 않다

이 구조 자체는 이미 학계·산업계에서 활발히 다뤄진다:
- **Neurosurgeon** [2] — DNN을 디바이스↔클라우드로 latency/energy 기준 분할
- **Split computing / early exiting** 서베이 [3]
- **Apple Private Cloud Compute** [4] — 온디바이스 소형 모델 + 프라이빗 클라우드 오프로드
- **Gemini Nano 하이브리드 추론** [5] — 온디바이스 + 클라우드

단순히 "디바이스는 화면, 엣지서버는 연산"으로 나누면 이는 thin client + 원격 추론에 불과하다. 연구 기여가 되려면 **무엇을 어디서, 언제 처리할지를 가르는 스케줄링 정책**이 핵심이어야 한다.

### 연구 기여 지점: 발열을 엣지서버로 옮기는 분할 스케줄링

핵심은 **발열이 어디서 발생하느냐**다.

- 디바이스 직접 연산 → 발열이 쿨링 빈약한 디바이스에 집중 → 쓰로틀링
- 엣지서버로 오프로드 → 그 발열이 쿨링 충분한 엣지서버로 이전

| | 디바이스 직접 연산 | 엣지서버로 오프로드 |
|---|---|---|
| **연산 발열 위치** | 디바이스 (쿨링 빈약 → 쓰로틀링) | 엣지서버 (쿨링 충분) |
| **통신 발열 위치** | 없음 | 디바이스 — 텍스트 토큰 송수신, 작음 (특히 결과 수신은 더 작음) |
| **디바이스 총 발열** | **큼** | **작음** |

- LLM 출력은 텍스트라 주고받는 데이터가 몇 KB 수준 → 통신 발열은 작고, 정작 발열을 지배하는 것은 연산(디코드마다 수십억 파라미터를 메모리에서 읽음)이다
- 즉 **발열이 큰 연산만 엣지서버로 옮기고, 발열이 작은 통신만 디바이스에 남기는** 것이 이 구조의 핵심 이점
- 단, 무선 전송은 비트당 에너지가 로컬 연산보다 크므로 [6] 통신량을 무작정 늘릴 수는 없음
- 따라서 **무거운 연산은 넘기되 통신량은 최소화하는 분할 지점**을 찾는 스케줄러가 기여의 핵심 — 순수 on-device와 순수 offload 양극단을 모두 이기는 것을 목표로 하는 동적 분할 정책
- LLM 특화 분할 단위:
  - KV cache를 디바이스/엣지서버 중 어디에 둘 것인가
  - prefill은 고성능 엣지서버, decode 일부는 디바이스
  - speculative decoding의 draft는 디바이스, verify는 엣지서버
- **에너지 · 지연 · 프라이버시 3축 트레이드오프**: 통신을 줄이면 에너지·프라이버시가 좋아지지만 큰 모델의 정교함을 포기. 이 균형을 스케줄러가 동적으로 푸는 것을 목표로 한다.

### 프로젝트 구성 (실험)

- **모듈 A — AI 연산용 고성능 엣지서버 칩**: 충분한 쿨링 전제, 무거운 연산 담당
- **모듈 B — 디바이스 측 수신·디스플레이 모듈**: 경량 연산 + 결과 수신·렌더링
- **핵심 산출물은 두 모듈이 아니라, 둘을 가르는 통신-aware 분할 스케줄러** — 두 모듈은 그 정책을 검증하는 실험 환경

**가설**: 디바이스–엣지서버 추론에서 병목과 에너지를 지배하는 것은 연산이 아니라 **디바이스 ↔ 엣지서버 간 통신 트랜잭션**이며, 통신량·통신 에너지를 명시적으로 모델링한 스케줄러가 순수 on-device/순수 offload 양극단을 모두 능가할 것으로 예상된다.

---

## 7. 결론

관통하는 원리는 하나다 — **트랜잭션 에너지 ≫ 연산 에너지**, 그리고 그 트랜잭션을 줄이는 책임은 결국 SW 스케줄러에 있다.

- **칩 내부**: 효율과 성능을 가르는 것은 연산기의 FLOPS가 아니라 **SW-HW co-design이 버스 트랜잭션을 얼마나 줄이는가**다. HW가 캐시 히트를 "기대"하는 구조에서, SW 스케줄러가 접근 패턴을 정적으로 예측해 히트를 "보장"하는 구조로 가야 한다.
- **디바이스–엣지서버 간**: 같은 원리가 한 계층 위에서 반복된다. 무거운 연산(과 그 발열)은 쿨링 충분한 엣지서버로 넘기되, 비트당 비싼 통신은 최소화하는 분할 지점을 SW 스케줄러가 찾아야 한다.

즉 칩 안에서든 디바이스–엣지서버 사이에서든, **저전력 고효율 AI의 최종 경쟁력은 트랜잭션을 줄이는 스케줄링에 달려 있다.**

---

## References

[1] M. Horowitz, "Computing's Energy Problem (and What We Can Do About It)," *ISSCC 2014*, pp. 10–14. (Fig. 1.1.9: Rough energy costs for various operations in 45nm 0.9V) https://ieeexplore.ieee.org/document/6757323

[2] Y. Kang et al., "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge," *ASPLOS 2017*, pp. 615–629. https://dl.acm.org/doi/10.1145/3037697.3037698

[3] Y. Matsubara, M. Levorato, F. Restuccia, "Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges," *ACM Computing Surveys*, 55(5), 2022. https://arxiv.org/abs/2103.04505

[4] Apple Security Engineering and Architecture, "Private Cloud Compute: A new frontier for AI privacy in the cloud," 2024. https://security.apple.com/blog/private-cloud-compute/

[5] Android Developers Blog, "Experimental hybrid inference and new Gemini models for Android." https://developer.android.com/blog/posts/experimental-hybrid-inference-and-new-gemini-models-for-android

[6] V. Raghunathan, C. Schurgers, S. Park, M. Srivastava, B. Shaw, "Energy-Aware Wireless Microsensor Networks," *IEEE Signal Processing Magazine*, 19(2), pp. 40–50, 2002. https://ieeexplore.ieee.org/document/985679
