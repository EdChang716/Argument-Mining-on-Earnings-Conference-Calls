# Argument Mining on Earnings Conference Calls (NLP Final Project — Task 2)

This repository contains code and notebooks for **argument mining** on earnings conference call transcripts, including:
- **Argument Unit Classification** (Premise vs. Claim)
- **Argument Relation Classification** (No / Support / Attack)

Task definitions and data scope follow the NTU course brief derived from **NTCIR-17 FinArg-1** setup. :contentReference[oaicite:0]{index=0}

---

## 📌 Tasks

### 1) Argument Unit Classification (AU)
Classify a single argumentative sentence into:
- **0 = Premise** (facts/examples/observations)
- **1 = Claim** (opinions/conclusions/suggestions) :contentReference[oaicite:1]{index=1}

### 2) Argument Relation Classification (AR)
Given a sentence pair `(Arg1, Arg2)`, predict the relation:
- **0 = No relation**, **1 = Support**, **2 = Attack**. :contentReference[oaicite:2]{index=2}

**Source:** Earnings conference calls (English). :contentReference[oaicite:3]{index=3}

---

## 🛠️ Methods

We implemented and compared:
- **In-Context learning (with/without examples)** using GPT-3.5/“GPT-4 class” models. Prompts with examples consistently improve performance. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
- **Supervised fine-tuning of GPT-3.5** on the training split (chat-style format).
- **LLaMA-3 fine-tuning** via **PEFT** (parameter-efficient fine-tuning) with a sequence-classification head. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

Typical fine-tuning settings (illustrative):
`learning_rate=1e-4`, `per_device_train_batch_size=8`, `per_device_eval_batch_size=8`, `num_train_epochs=2`. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

---

## 📊 Results (Weighted F1)

| Task | Method | Weighted F1 |
|---|---|---|
| AU (Premise vs. Claim) | GPT-3.5 **prompt w/ examples** (no FT) | 0.67 |
| AU (Premise vs. Claim) | **GPT-3.5 fine-tuned** | **0.76** |
| AU (Premise vs. Claim) | LLaMA-3 fine-tuned (PEFT) | 0.75 |
| AR (No/Support/Attack) | GPT-class **prompt w/ examples** (no FT) | 0.60 |
| AR (No/Support/Attack) | **GPT-class fine-tuned** | **0.77** |
| AR (No/Support/Attack) | LLaMA-3 fine-tuned (PEFT) | 0.76 |

> Prompts with examples improve results over zero-shot; **fine-tuning further lifts F1** (GPT slightly > LLaMA-3 under our setup). :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

---

## 📂 Repository Structure

