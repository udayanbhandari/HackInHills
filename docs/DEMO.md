# 🎬 SwarnaScope: 3-Minute Hackathon Demo Script

> **Hack in Hills '26** — *Commodity Derivatives Intelligence*  
> *Presenter*: Quant Engineer Team

---

## ⏱️ Demo Overview (3 Minutes Total)

| Time | Section | Focus | Key UI Screen |
|---|---|---|---|
| **0:00 - 0:45** | **1. Opening & Data-Trust Layer** | Data integrity, Bhavcopy validation & quality score | Tab 2: Data Trust & Quality |
| **0:45 - 1:30** | **2. Carry-Adjusted Fair Value** | Decomposition of spreads vs carrying cost & z-scores | Tab 3: Fair-Value Lab |
| **1:30 - 2:15** | **3. Time-Machine Replay** | Proof of zero look-ahead bias day by day | Tab 6: Time-Machine Replay |
| **2:15 - 3:00** | **4. Falsification Verdict & Closing** | "Try to kill our own signal" suite & honest verdict | Tab 1: Overview & Tab 8: Falsification |

---

## 📜 Minute-by-Minute Script

### 🎤 Minute 1: Opening & Data-Trust Layer (0:00 - 0:45)
> **Presenter:**  
> *"Good morning judges. Welcome to **SwarnaScope** — an honest alpha lab built specifically for MCX gold futures derivatives.*  
> *Most trading demos start by showing a perfect upward equity curve. We start with **Data Trust**.*  
> *In commodity futures, raw exchange data is full of pitfalls: silent holiday fallbacks, space-padded contract symbols, and zero-volume settlement prints.*  
> *Here on our **Data-Trust Scorecard**, SwarnaScope automatically validates every Bhavcopy row, catches date mismatches, strips symbol padding, and filters non-tradable zero-volume days.*  
> *Our dataset is live-compatible, raw-file compatible, and auto-falls back to an 18-month realistic simulation clearly labeled as `SYNTHETIC DEMO DATA` when offline."*

---

### 🎤 Minute 2: Carry-Adjusted Fair Value & Integer Hedging (0:45 - 1:30)
> **Presenter:**  
> *"Now let's look at our core USP: **Never call a raw spread a mispricing**.*  
> *If GOLDM expires on the 5th and GOLDGUINEA expires on the 30th, the raw price difference is NOT an arbitrage — it reflects the cost of carrying gold across that 25-day expiry gap.*  
> *In our **Fair-Value Lab**, SwarnaScope decomposes the observed spread into carrying cost, purity adjustments (995 vs 999), and noise using strictly walk-forward rolling regressions.*  
> *The residual is the true tradable mispricing.*  
> *And because lot sizes are integer-constrained (100g, 10g, 8g, 1g), our **Integer Hedge Builder** calculates the exact whole-lot ratio (like 2 lots GOLDM vs 25 lots GOLDGUINEA for 200g) and tracks residual unhedged gold exposure."*

---

### 🎤 Minute 3: Time-Machine Replay & Falsification Verdict (1:30 - 3:00)
> **Presenter:**  
> *"To prove our backtest is completely free of look-ahead bias, we built the **Time-Machine Replay**.*  
> *By sliding across historical trading dates, judges can verify that on any day $t$, our model sees ONLY information knowable on or before $t-1$.*  
> *Finally, our quant philosophy is **Rigor beats flashiness**.*  
> *We run an automated **Falsification Suite** that tries to kill our own signal with placebo tests, return permutation tests, bootstrap confidence intervals, and 2.5x cost sensitivity sweeps.*  
> *If micro-structure costs absorb the edge, SwarnaScope reports an honest verdict: `NO PERSISTENT EDGE AFTER COSTS` and stays quiet.*  
> *Thank you, and we welcome your questions!"*
