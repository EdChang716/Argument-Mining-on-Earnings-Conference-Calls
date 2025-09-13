# Argument Mining on Earnings Conference Calls

This repository contains code and notebooks for **argument mining** on earnings conference call transcripts, including:
- **Argument Unit Classification** (Premise vs. Claim)
- **Argument Relation Classification** (No / Support / Attack)

Task definitions and data scope follow the NTU course brief derived from **NTCIR-17 FinArg-1** setup. 
---

## 📌 Tasks

### 1) Argument Unit Classification (AU)
Classify a single argumentative sentence into:
- **0 = Premise** (facts/examples/observations)
- **1 = Claim** (opinions/conclusions/suggestions)

### 2) Argument Relation Classification (AR)
Given a sentence pair `(Arg1, Arg2)`, predict the relation:
- **0 = No relation**, **1 = Support**, **2 = Attack**.

**Source:** Earnings conference calls (English). 

---

## 🛠️ Methods

We implemented and compared:
- **In-Context learning (with/without examples)** using GPT-3.5/“GPT-4 class” models. Prompts with examples consistently improve performance. 
- **Supervised fine-tuning of GPT-3.5** on the training split (chat-style format).
- **LLaMA-3 fine-tuning** via **PEFT** (parameter-efficient fine-tuning) with a sequence-classification head. :contentReference[oaicite:6]{index=6} 

Typical fine-tuning settings (illustrative):
`learning_rate=1e-4`, `per_device_train_batch_size=8`, `per_device_eval_batch_size=8`, `num_train_epochs=2`. :contentReference[oaicite:8]{index=8} 

---

## 📊 Results (Weighted F1)

<table>
<tr>
<td>

**Argument Unit Classification (AU: Premise vs. Claim)**

| Method | Weighted F1 |
|---|---|
| GPT-3.5 prompt without examples | 0.66 |
| GPT-3.5 In-context learning | 0.67 |
| **GPT-3.5 fine-tuned** | **0.76** |
| LLaMA-3 fine-tuned (PEFT) | 0.75 |

</td>
<td>

**Argument Relation Classification (AR: No / Support / Attack)**

| Method | Weighted F1 |
|---|---|
| GPT-4 prompt without examples | 0.13 |
| GPT-4 In-context learning | 0.60 |
| **GPT-4 fine-tuned** | **0.77** |
| LLaMA-3 fine-tuned (PEFT) | 0.76 |

</td>
</tr>
</table>

> In-context learning improve results over zero-shot; **fine-tuning further lifts F1** (GPT slightly > LLaMA-3 under our setup). :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

---

## 📂 Repository Structure

```plaintext
.
├── README.md
├── Code/
│   ├── Task2.1_LLAMA3_FT.ipynb
│   ├── Task2.2_LLAMA3_FT.ipynb
│   ├── Task_2.1_GPT_FT_code.ipynb
│   └── Task_2.2_GPT_FT_code.ipynb
│   └── Task2_ICL_GPT.ipynb  # contains both task 2.1 and 2.2
├── data/
│   ├── Task2.1/               # Original Data for task 2.1
│   ├── Task2.2/                 # Original Data for task 2.2
│   └── others             # preprocessed data for model fine-tuning and ICL

