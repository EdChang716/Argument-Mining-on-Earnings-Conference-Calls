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
├── notebooks/
│   ├── 2_1_argument_unit_classification_gpt35.ipynb
│   ├── 2_1_argument_unit_classification_llama3_peft.ipynb
│   ├── 2_2_argument_relation_classification_gpt35.ipynb
│   └── 2_2_argument_relation_classification_llama3_peft.ipynb
├── data/
│   ├── train/               # training split (not tracked; see Data Access)
│   ├── dev/                 # dev split (not tracked)
│   └── test/                # test split (not tracked)
├── scripts/
│   ├── preprocess.py        # text cleaning / dataset builders
│   ├── peft_ft_llama.py     # PEFT finetune utilities
│   └── eval_metrics.py      # F1 / confusion matrix / reports
├── results/
│   ├── au/                  # AU predictions / logs
│   └── ar/                  # AR predictions / logs
└── requirements.txt


