# Income Prediction with Bias Check and Fix (PyTorch + Fairlearn)

## What This Project Does
This project trains a neural network to predict whether a person earns more than $50K a year, using the UCI Adult Income dataset. But it doesn't stop at just building a model — it also checks whether the model is being **unfair to any group of people**, and then fixes that unfairness.

In short: build a model → check if it's biased → fix the bias → compare the results.

## Tools Used
* **PyTorch** – to build and train the neural network from scratch (custom model, custom training loop, DataLoader)
* **Fairlearn** – to measure bias and fix it (`MetricFrame`, `ThresholdOptimizer`)
* **Pandas, NumPy, Scikit-learn** – for data cleaning, splitting, and scoring

## Step 1: The Problem We Found
The neural network was trained the normal way, using Binary Cross Entropy Loss. On its own, it looked great:

* **Overall Accuracy: 86.29%**

But when we split the results by **gender**, something unfair showed up:

* **Men predicted as high-income:** 24.53% of the time
* **Women predicted as high-income:** only 7.98% of the time
* **Gap between the two (Demographic Parity Difference): 16.55%**

In simple words: the model was about **3 times more likely** to predict a high income for a man than for a woman, just because that pattern existed in the historical data it was trained on. This is a classic example of a model learning real-world bias instead of just learning the actual skills or qualifications.

## Step 2: How We Fixed It
To fix this, the trained PyTorch model was wrapped so that Fairlearn could work with it, and then Fairlearn's `ThresholdOptimizer` was used. Instead of using one single cutoff (like "50% probability = high income") for everyone, this tool finds a **separate, fairer cutoff for each group** so that both genders get predicted as high-income at a similar rate.

## Step 3: The Tradeoff — Accuracy vs Fairness
Fixing bias usually comes with some cost to accuracy, and that tradeoff is shown clearly below:

| Metric | Before Fix | After Fix | What Changed |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 86.29% | 83.62% | Dropped slightly (-2.67%) |
| **Bias Gap (Demographic Parity)** | 16.55% | 0.41% | Almost completely fixed |
| **Women predicted as high-income** | 7.98% | 17.80% | Much more balanced |
| **Men predicted as high-income** | 24.53% | 17.39% | Much more balanced |

**Bottom line:** By giving up a small amount of accuracy (about 2.6%), the project shrunk the gender bias gap from 16.5% down to less than 0.5% — making the model much fairer while still working well.

## Project Files
* `train.py` — Everything is in this one file: loading the data, preparing it, building the neural network, training it, checking for bias, and fixing the bias.

## How to Run It
1. Clone this repository.
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the environment and install the required packages:
   ```
   pip install torch pandas scikit-learn fairlearn
   ```
4. Run the script:
   ```
   python train.py
   ```
