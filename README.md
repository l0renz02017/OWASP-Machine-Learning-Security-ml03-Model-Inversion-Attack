# 🕵️ ML03:2023 Model Inversion Attack - Stealing Data from an AI's Memory Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l0renz02017/OWASP-Machine-Learning-Security-ml03-Model-Inversion-Attack/blob/main/demo.py)
![OWASP](https://img.shields.io/badge/OWASP%20ML%20Top%2010-ML03:2023_Model_Inversion-%23bb0a1e?link=https://owasp.org/www-project-machine-learning-security-top-10/)
![Related](https://img.shields.io/badge/See_Also-ML01_&_ML02-blue?link=https://github.com/l0renz02017/)

**Demonstration of a privacy attack that reverse-engineers a machine learning model to reconstruct the confidential data it was trained on.**

This repository demonstrates an **ML03:2023 Model Inversion Attack**, where an attacker with only query access to a model can steal sensitive features of its training data, effectively reading the model's "memory".

## 🚨 Why Is This A Security Nightmare

Unlike previous attacks that fool or poison a model, this attack **steals from it**. It exploits the model as a leakage channel for its own training data.

No Database Breach Required: The attacker never had to hack the database of stored training data. The attacker can steal the training data from the AI's memory by repeatedly asking it questions.

The AI is both the Victim and the Accomplice: The very system designed to protect security is tricked into revealing the secret it was designed to protect.

This attack could work for:

a) Facial/Fingerprint Recognition: Generating a face/fingerprint that unlocks someone's phone.

b) Voice Authentication: Generating a voice print that bypasses voice ID systems.

c) Medical Records: Reconstructing someone's private medical scan (like an MRI) from a diagnostic AI.

## ⚡ Quick Demo

Click the button below to run the complete code on Google Colab. No setup required. In minutes, you will:
1.  **Train** a "victim" model on a confidential dataset (images of the digit '3').
2.  **Perform the Inversion Attack:** Start with random noise and query the model to slowly steal the features of the digit '3'.
3.  **Succeed:** Watch as the model confidently identifies the generated noise as a '3', proving its memory has been stolen.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l0renz02017/OWASP-Machine-Learning-Security-ml03-Model-Inversion-Attack/blob/main/demo.py)

## 🔍 What You Will See

A successful run will produce a compelling visual result:

The visualization will show two images:
1.  **The Inverted Image:** A recognizable digit '3' that was **reverse-engineered from the model's parameters**.
2.  **A Real Training Image:** An actual '3' from the dataset for comparison.

The model will be over **100% confident** that the inverted image is genuine, proving the attack successfully extracted the "idea" of a '3' from the model's memory.

## 🏗️ How The Attack Works (The "Inversion")

The attack does not attack the model's code. It attacks its **knowledge**.

### 1. The Setup
A model is trained on sensitive data (in this case, the digit '3').

### 2. The Reverse-Engineering Process
The attacker starts with random noise and performs the following steps in a loop:
-   **Query:** "How confident are you that this noise is a '3'?"
-   **Learn:** The model responds with a confidence score and (critically) internal information on *how* to change the noise.
-   **Adapt:** The attacker uses this information to adjust the noise, making it look more like what the model "thinks" a '3' should be.
-   **Repeat:** This process repeats hundreds of times.

### 3. The Result
The random noise is gradually transformed into a recognizable copy of the sensitive data features. The model has been **inverted**—used in reverse to spit out the data it was trained on.


✅ All libraries installed and imported!  
🔒 Training a 'victim' model on 'confidential' data (Digit 3s)...  
100%|██████████| 9.91M/9.91M [00:00<00:00, 34.7MB/s]  
100%|██████████| 28.9k/28.9k [00:00<00:00, 1.11MB/s]  
100%|██████████| 1.65M/1.65M [00:00<00:00, 8.18MB/s]  
100%|██████████| 4.54k/4.54k [00:00<00:00, 5.20MB/s]  
✅ Victim model trained on digit 3s!  

🎯 Launching Model Inversion Attack...  
   Goal: Steal the features of digit '3' from the model's memory.  
   Step 0: Confidence for class 3 = 7.22%  
   Step 200: Confidence for class 3 = 100.00% 
   Step 400: Confidence for class 3 = 100.00%  
   Step 600: Confidence for class 3 = 100.00%  
   Step 800: Confidence for class 3 = 100.00%  
   Step 1000: Confidence for class 3 = 100.00%  

✅ Attack Complete! Final confidence for class '3': 100.00%  

👁️  Comparing inverted image with real training data:
<img width="799" height="368" alt="image" src="https://github.com/user-attachments/assets/449ecf67-2bbf-4179-a898-f3d96bd9621d" />


🧠 **WHAT DOES THIS MEAN?**
🎯 ATTACK SUCCESSFUL!
   The model inversion attack successfully reconstructed features of a '3'.
   The inverted image (left) is what the model 'thinks' a perfect '3' looks like.
   This demonstrates that model parameters can leak information about their training data.

🔒 This is ML03:2023 - an attacker can potentially steal sensitive training data just by querying a model.

## 🛡️ OWASP ML03:2023 - Model Inversion Attack

This project demonstrates **[ML03:2023](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML03_2023-Model_Inversion_Attack.html)** from the OWASP ML Security Top 10.

> "Model Inversion Attacks occur when an adversary with access to a machine learning model attempts to reconstruct the model's training data or extract sensitive information about individuals or confidential data from the model's parameters."

## 📁 Repository Contents

-   **`demo.py`**: The complete, self-contained Python script to run the demo.
-   **`README.md`**: This file.

## 🏃‍♂️ How to Run
1. Open Google Colab: Go to Google Colab.
2. Create a New Notebook: Click File > New notebook.
3. Run the Code: Copy the entire code block from demo.py and paste it into a single cell in your Colab notebook.
4. Execute: Click the play (▶️) button or go to Runtime > Run all.

## 🔬 Related Work

This demo is a series OWASP ML Top 10 demonstrations on MNIST:
-   [**ML01:2023 Input Manipulation Attack Demo**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml01-input-manipulation-attack) - Fooling a trained model.
-   [**ML02:2023 Data Poisoning Attack Demo**](https://github.com/l0renz02017/OWASP-Machine-Learning-Security-ml02-Data-Poisoning-Attack) - corrupting a model during training.
-   **ML03:2023 Model Inversion Attack** (This repo) - stealing data from a trained model.

## 📚 Learn More

-   [OWASP ML Top 10: ML03:2023 Model Inversion Attack](https://owasp.org/www-project-machine-learning-security-top-10/docs/ML03_2023-Model_Inversion_Attack.html)
-   [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://arxiv.org/abs/1809.06532) - Seminal paper on the topic.

## ⚠️ Disclaimer

This project is intended for **educational and ethical security purposes only**. The goal is to help developers, security professionals, and students understand ML privacy vulnerabilities to build more secure and privacy-preserving AI systems.

---

**If this project helped you understand the threat of model inversion, please give it a ⭐!**
