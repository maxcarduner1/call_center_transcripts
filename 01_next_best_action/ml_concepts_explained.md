# Machine Learning Concepts Explained: A Beginner's Guide

## Introduction

This document explains the machine learning terminology and concepts used in the DBX Outreach Model. We'll build from the ground up, assuming no prior ML knowledge.

---

## Part 1: The Fundamentals

### What is Machine Learning?

Machine learning is teaching computers to learn patterns from historical data, then use those patterns to make predictions about new data.

**Traditional Programming:**
```
Human writes rules → Computer follows rules → Output

Example: "If member answered 3+ calls last year, mark as 'likely to engage'"
```

**Machine Learning:**
```
Human provides data → Computer discovers rules → Output

Example: Computer analyzes 100,000 past calls and figures out what 
         predicts engagement (maybe it's not just call count, but 
         call count + time of day + age + language preference...)
```

The key difference: **we don't tell the computer the rules—it figures them out from examples.**

---

### Features and Labels

These are the two most fundamental ML concepts:

#### Features (Inputs)
**Features** are the information we give the model to make predictions. Think of them as the "clues" or "characteristics" the model uses.

```
For our outreach model, features include:

MEMBER FEATURES:
├── Age: 67
├── Gender: Female
├── Insurance Type: Medicare
├── Chronic Conditions: 3
├── Has Mobile Phone: Yes
└── Preferred Language: Spanish

BEHAVIORAL FEATURES:
├── Past Calls Received: 12
├── Calls Answered: 8
├── Answer Rate: 66.7%
├── Days Since Last Contact: 45
└── Average Call Duration: 4.5 minutes

SDOH FEATURES:
├── Transportation Barrier: Yes
├── Food Insecurity: No
└── Area Deprivation Index: 72
```

#### Labels (Outputs)
**Labels** are the answers we're trying to predict. During training, we show the model examples where we already know the answer.

```
For our engagement model:

Label = "Did the member engage?" 

Possible values:
├── 1 = Yes, they answered and completed the call
└── 0 = No, they didn't engage
```

#### The Learning Process

```
TRAINING PHASE:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Historical Data (we know what happened):                       │
│                                                                 │
│  Member    Features                           Label             │
│  ──────    ────────                           ─────             │
│  Alice     Age=45, Answer_Rate=80%, ...   →   Engaged=1        │
│  Bob       Age=72, Answer_Rate=20%, ...   →   Engaged=0        │
│  Carol     Age=55, Answer_Rate=65%, ...   →   Engaged=1        │
│  David     Age=38, Answer_Rate=15%, ...   →   Engaged=0        │
│  ...       (thousands more examples)                            │
│                                                                 │
│  Model learns: "High answer rate + middle age = likely engage"  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

PREDICTION PHASE:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  New Member (we don't know yet):                                │
│                                                                 │
│  Eve       Age=52, Answer_Rate=70%, ...   →   ???               │
│                                                                 │
│  Model predicts: "Probably Engaged=1 (78% confidence)"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

### Training and Test Sets

We split our data into two groups:

```
ALL HISTORICAL DATA (10,000 members)
            │
            ├──────────────────────────────┐
            │                              │
            ▼                              ▼
    TRAINING SET (80%)              TEST SET (20%)
    8,000 members                   2,000 members
            │                              │
            │                              │
            ▼                              ▼
    Model learns from               Model is evaluated on
    these examples                  these examples
                                    (model never saw these
                                    during training)
```

**Why split?** To check if the model actually learned general patterns, not just memorized the training data.

**Analogy:** It's like studying for a test:
- Training set = Practice problems you study
- Test set = The actual exam (different questions)
- A good student (model) learns concepts, not just memorizes answers

---

### Classification vs. Regression

These are the two main types of prediction:

#### Classification
Predicting **categories** or **classes** (discrete outcomes).

```
BINARY CLASSIFICATION (2 possible outcomes):
├── Will member engage? → Yes / No
├── Will they respond to SMS? → Yes / No
└── Is this a fraudulent claim? → Yes / No

MULTI-CLASS CLASSIFICATION (3+ possible outcomes):
├── What's the best channel? → Phone / SMS / Email / Mail
├── Risk level? → Low / Medium / High
└── Diagnosis category? → Category A / B / C / D
```

#### Regression
Predicting **continuous numbers**.

```
REGRESSION EXAMPLES:
├── How long will the call last? → 4.5 minutes
├── What's the member's risk score? → 2.7
└── How many days until they respond? → 12.3 days
```

**Our engagement model is binary classification:** predicting Yes (1) or No (0).

---

### Probability vs. Hard Predictions

Models can output either:

#### Hard Prediction
```
"This member WILL engage" or "This member WON'T engage"
```

#### Probability (Soft Prediction)
```
"This member has a 73% chance of engaging"
```

**Probabilities are more useful because:**

1. **Ranking:** We can rank members by likelihood
   ```
   Alice: 92% → Call first
   Bob:   73% → Call second
   Carol: 45% → Call third
   David: 12% → Maybe don't call
   ```

2. **Threshold flexibility:** Business can decide the cutoff
   ```
   "Call everyone above 50%" → More calls, some won't answer
   "Call everyone above 80%" → Fewer calls, most will answer
   ```

3. **Confidence:** We know when the model is uncertain
   ```
   92% → Model is confident
   51% → Model is basically guessing
   ```

---

## Part 2: Model Types

### What is a "Model"?

A model is a mathematical function that takes features as input and produces predictions as output.

```
                    ┌─────────────────┐
Features ────────►  │     MODEL       │  ────────► Prediction
(age, answer_rate,  │  (learned from  │           (73% will engage)
 SDOH, etc.)        │   training)     │
                    └─────────────────┘
```

Different model types learn different kinds of patterns.

---

### Decision Trees

The simplest model to understand. It asks a series of yes/no questions:

```
                    ┌─────────────────────────┐
                    │ Answer Rate > 50%?      │
                    └───────────┬─────────────┘
                          Yes   │   No
                    ┌───────────┴───────────┐
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │ Age < 75?       │     │ Phone Verified? │
          └────────┬────────┘     └────────┬────────┘
              Yes  │  No              Yes  │  No
              ┌────┴────┐             ┌────┴────┐
              ▼         ▼             ▼         ▼
           ENGAGE    ENGAGE        ENGAGE    DON'T
           (85%)     (60%)         (40%)     (15%)
```

**Pros:** Easy to understand and explain
**Cons:** A single tree often isn't accurate enough

---

### Random Forest

Instead of one tree, train MANY trees (a "forest") and let them vote:

```
           Features
               │
     ┌─────────┼─────────┬─────────┐
     ▼         ▼         ▼         ▼
  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
  │Tree 1│ │Tree 2│ │Tree 3│ │Tree 4│  ... (hundreds of trees)
  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
     │        │        │        │
     ▼        ▼        ▼        ▼
   Engage   Don't    Engage   Engage
     │        │        │        │
     └────────┴────────┴────────┘
                  │
                  ▼
         FINAL VOTE: Engage
         (3 out of 4 trees said Engage)
```

**Why it works better:** Each tree sees a random subset of data and features, so they make different mistakes. The vote cancels out individual errors.

---

### Gradient Boosted Trees (XGBoost, LightGBM)

The models we use. Instead of training trees independently, train them **sequentially** where each tree learns from the previous trees' mistakes:

```
ROUND 1:
┌──────────────────────────────────────────────────────────────┐
│ Tree 1 makes predictions                                     │
│                                                              │
│ Alice: Predicted 70% → Actual: Engaged    ✓ (close!)        │
│ Bob:   Predicted 60% → Actual: Not Engaged ✗ (wrong!)       │
│ Carol: Predicted 40% → Actual: Engaged    ✗ (wrong!)        │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
ROUND 2:
┌──────────────────────────────────────────────────────────────┐
│ Tree 2 focuses on fixing Tree 1's mistakes                   │
│                                                              │
│ "I'll pay extra attention to Bob and Carol"                  │
│                                                              │
│ Tree 2 learns patterns that Tree 1 missed                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
ROUND 3:
┌──────────────────────────────────────────────────────────────┐
│ Tree 3 focuses on remaining errors from Trees 1+2            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ... (hundreds of rounds)
                              │
                              ▼
FINAL PREDICTION = Tree1 + Tree2 + Tree3 + ... (weighted sum)
```

**This is called "boosting"** because each tree "boosts" the performance by fixing errors.

**Why XGBoost and LightGBM?**
- XGBoost (eXtreme Gradient Boosting): Highly optimized, very accurate
- LightGBM (Light Gradient Boosting Machine): Even faster, nearly as accurate

---

### Why Trees Beat Neural Networks for Our Data

You've probably heard about neural networks and deep learning (used for ChatGPT, image recognition, etc.). Why don't we use them?

```
NEURAL NETWORKS excel at:                TREES excel at:
├── Images (millions of pixels)          ├── Tabular data (rows & columns)
├── Text (sequences of words)            ├── Mixed data types
├── Audio (sound waves)                  ├── Smaller datasets (thousands)
└── Unstructured data                    └── Structured data

Our data:
├── 10,000 members (not millions)
├── 60 features in columns
├── Mix of numbers and categories
├── Traditional structured format
│
└── → Trees are the better choice!
```

**Research consistently shows:** For tabular/structured data, gradient boosted trees outperform neural networks. This has been proven in countless Kaggle competitions and academic studies.

---

## Part 3: Key Concepts in Our Model

### Class Imbalance

When one outcome is much more common than another:

```
OUR DATA:
┌────────────────────────────────────────────┐
│ ████████████████████████████░░░░░░░░░░░░░░ │
│ 75% Did NOT Engage       25% Engaged       │
└────────────────────────────────────────────┘

This is "imbalanced" - many more non-engaged than engaged
```

**Why it's a problem:**

A lazy model could predict "Not Engaged" for everyone and be 75% accurate! But that's useless—we WANT to find the engaged members.

```
LAZY MODEL:
Predicts "Not Engaged" for all 10,000 members
├── Correct on 7,500 (the actual non-engaged) → 75% accuracy!
└── Wrong on 2,500 (missed all the engaged) → USELESS
```

**How we fix it: `scale_pos_weight`**

We tell the model: "A mistake on an engaged member is 3x worse than a mistake on a non-engaged member."

```python
scale_pos_weight = 7500 / 2500 = 3.0

# Translation: "Pay 3x more attention to engaged members"
```

This forces the model to actually learn what predicts engagement, not just predict the majority class.

---

### Overfitting vs. Underfitting

#### Overfitting
The model memorizes the training data instead of learning general patterns.

```
TRAINING DATA:                    NEW DATA:
├── 95% accuracy                  ├── 60% accuracy
└── "I memorized the answers!"    └── "Wait, these are different questions!"

ANALOGY: A student who memorizes practice test answers word-for-word,
         then fails the real test because questions are phrased differently.
```

**Signs of overfitting:**
- Training accuracy much higher than test accuracy
- Model is too complex (too many trees, trees too deep)
- Performance gets worse as you add more trees

#### Underfitting
The model is too simple to capture the patterns.

```
TRAINING DATA:                    NEW DATA:
├── 55% accuracy                  ├── 54% accuracy
└── "I didn't really learn"       └── "Still don't know"

ANALOGY: A student who only skimmed the textbook and learned nothing.
```

**Signs of underfitting:**
- Low accuracy on both training AND test data
- Model is too simple (too few trees, trees too shallow)

#### The Sweet Spot

```
Model Complexity →

Accuracy
   │
   │                        ╭────────╮
   │                    ╭───╯        ╰───╮
   │                ╭───╯   SWEET SPOT   ╰───╮
   │            ╭───╯      Just right!       ╰─── Overfitting
   │        ╭───╯                                 (too complex)
   │    ╭───╯
   │ Underfitting
   │ (too simple)
   └─────────────────────────────────────────────────
```

---

### Regularization

Techniques to prevent overfitting. Think of it as "rules that keep the model from getting too complicated."

```
REGULARIZATION TECHNIQUES IN XGBoost:

1. MAX DEPTH = 6
   "Each tree can only ask 6 questions deep"
   
   Prevents: Trees that are too specific
   ┌─────────────────────────────────────────────────────────┐
   │ Without limit: "If age=67 AND calls=12 AND state=CA    │
   │                 AND zip=94102 AND..."  (memorization!) │
   │                                                         │
   │ With max_depth=6: Simpler, more general rules          │
   └─────────────────────────────────────────────────────────┘

2. MIN_CHILD_WEIGHT = 5
   "Each final group must have at least 5 members"
   
   Prevents: Rules based on 1-2 unusual members
   ┌─────────────────────────────────────────────────────────┐
   │ Without limit: "The 1 member with exactly these        │
   │                 characteristics engaged, so..."         │
   │                                                         │
   │ With min=5: Need at least 5 similar members            │
   └─────────────────────────────────────────────────────────┘

3. SUBSAMPLE = 0.8
   "Each tree only sees 80% of the data (randomly selected)"
   
   Prevents: All trees learning the same patterns
   ┌─────────────────────────────────────────────────────────┐
   │ Tree 1 sees: Alice, Bob, Carol, David (not Eve)        │
   │ Tree 2 sees: Alice, Carol, Eve, Frank (not Bob)        │
   │                                                         │
   │ Different views = more diverse, robust predictions     │
   └─────────────────────────────────────────────────────────┘

4. LEARNING_RATE = 0.05
   "Each tree only contributes 5% to the final prediction"
   
   Prevents: One tree dominating the prediction
   ┌─────────────────────────────────────────────────────────┐
   │ Without: Tree 1 might completely overfit               │
   │                                                         │
   │ With: Need many trees to agree = more stable           │
   └─────────────────────────────────────────────────────────┘
```

---

### Feature Importance

After training, we can ask: "Which features did the model rely on most?"

```
TOP FEATURES FOR ENGAGEMENT PREDICTION:

historical_success_rate     ████████████████████ 15%
days_since_last_contact     ███████████████░░░░░ 12%
answer_rate_pct             ██████████████░░░░░░ 11%
age                         █████████████░░░░░░░  9%
phone_number_verified       ███████████░░░░░░░░░  8%
chronic_condition_count     ██████████░░░░░░░░░░  7%
sdoh_transportation         █████████░░░░░░░░░░░  6%
...
```

**Why this matters:**

1. **Explainability:** "The model thinks Alice will engage because she has a high historical answer rate and hasn't been contacted recently."

2. **Debugging:** If a weird feature is #1, something might be wrong.

3. **Feature engineering:** We know where to focus future improvements.

---

### Probability Calibration

Raw model outputs aren't always well-calibrated probabilities:

```
UNCALIBRATED MODEL:
"I predict 70% for 1,000 members"
Actual engagement: Only 55% engaged

The model is OVERCONFIDENT - its 70% really means 55%


CALIBRATED MODEL:
"I predict 70% for 1,000 members"
Actual engagement: 70% engaged

The probability means what it says!
```

**Why calibration matters:**
- We use probabilities for prioritization scores
- Business rules depend on accurate probabilities
- "Call everyone with >60% likelihood" needs 60% to mean 60%

**How we calibrate:**
```python
# Isotonic regression: Learns a mapping from raw → calibrated
calibrated_model = CalibratedClassifierCV(model, method='isotonic')
```

This is like creating a "correction table" that fixes the model's probability estimates.

---

## Part 4: Evaluation Metrics

How do we know if our model is good?

### Accuracy

```
Accuracy = Correct Predictions / Total Predictions

Example:
├── 8,500 correct out of 10,000
└── Accuracy = 85%
```

**Problem:** Misleading with imbalanced data (remember the "lazy model" that predicts majority class).

---

### Confusion Matrix

A 2x2 table showing all outcomes:

```
                      PREDICTED
                    No      Yes
              ┌─────────┬─────────┐
         No   │  TRUE   │  FALSE  │
    A         │NEGATIVE │POSITIVE │
    C         │  (TN)   │  (FP)   │
    T    ─────┼─────────┼─────────┤
    U         │  FALSE  │  TRUE   │
    A   Yes   │NEGATIVE │POSITIVE │
    L         │  (FN)   │  (TP)   │
              └─────────┴─────────┘

TN (True Negative):  Predicted No, Actually No    ✓ Correct
FP (False Positive): Predicted Yes, Actually No   ✗ False alarm
FN (False Negative): Predicted No, Actually Yes   ✗ Missed!
TP (True Positive):  Predicted Yes, Actually Yes  ✓ Correct
```

**For our model:**
```
                        PREDICTED
                   Won't Engage  Will Engage
              ┌──────────────┬──────────────┐
    Won't     │    6,800     │     700      │
    Engage    │   (correct)  │ (wasted call)│
  A      ─────┼──────────────┼──────────────┤
  C           │     500      │    2,000     │
  T     Will  │   (missed    │  (correct)   │
  U    Engage │  opportunity)│              │
  A           └──────────────┴──────────────┘
  L
```

---

### Precision and Recall

Two ways to measure performance on the "positive" class (engaged members):

#### Precision
"Of everyone we PREDICTED would engage, how many actually did?"

```
Precision = True Positives / (True Positives + False Positives)
          = 2,000 / (2,000 + 700)
          = 74%

"When we predict someone will engage, we're right 74% of the time"

HIGH PRECISION = Few wasted calls (but might miss some engaged members)
```

#### Recall (Sensitivity)
"Of everyone who ACTUALLY engaged, how many did we identify?"

```
Recall = True Positives / (True Positives + False Negatives)
       = 2,000 / (2,000 + 500)
       = 80%

"We identify 80% of members who would engage"

HIGH RECALL = Few missed opportunities (but might make some wasted calls)
```

#### The Trade-off

```
If we lower our threshold (predict "engage" more often):
├── Recall goes UP (we catch more engaged members)
└── Precision goes DOWN (more false alarms)

If we raise our threshold (predict "engage" less often):
├── Precision goes UP (when we predict engage, we're more sure)
└── Recall goes DOWN (we miss more engaged members)

                         ┌─────────────────────────────┐
Threshold: 80%           │ Only call very likely ones  │
├── Precision: 90%       │ Few calls, but very targeted│
└── Recall: 40%          └─────────────────────────────┘

                         ┌─────────────────────────────┐
Threshold: 30%           │ Call almost everyone        │
├── Precision: 40%       │ Many calls, less targeted   │
└── Recall: 95%          └─────────────────────────────┘
```

---

### F1 Score

Balances precision and recall into a single number:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Example:
├── Precision = 74%
├── Recall = 80%
└── F1 = 2 × (0.74 × 0.80) / (0.74 + 0.80) = 77%
```

**F1 is useful when you care about both precision and recall equally.**

---

### AUC-ROC

The metric we use most. AUC = "Area Under the ROC Curve"

**What it measures:** How well the model RANKS members (separates engaged from non-engaged).

```
PERFECT MODEL (AUC = 1.0):
All engaged members ranked above all non-engaged members

Score:   0.9   0.8   0.7   0.6   0.5   0.4   0.3   0.2
Member:  ███   ███   ███   ███   ░░░   ░░░   ░░░   ░░░
         Engaged (all high)      Not Engaged (all low)


USELESS MODEL (AUC = 0.5):
Random ranking, no separation

Score:   0.9   0.8   0.7   0.6   0.5   0.4   0.3   0.2
Member:  ███   ░░░   ███   ░░░   ███   ░░░   ███   ░░░
         Mixed together randomly


GOOD MODEL (AUC = 0.78):
Mostly separated, some overlap

Score:   0.9   0.8   0.7   0.6   0.5   0.4   0.3   0.2
Member:  ███   ███   ███   ░██   ░░█   ░░░   ░░░   ░░░
         Mostly engaged →  ← Overlap →  ← Mostly not engaged
```

**Interpretation:**
- AUC = 0.5: Model is guessing randomly
- AUC = 0.6-0.7: Poor model
- AUC = 0.7-0.8: Fair model
- AUC = 0.8-0.9: Good model
- AUC = 0.9+: Excellent model

**Why we like AUC:**
- Works regardless of threshold
- Not affected by class imbalance
- Measures ranking ability (perfect for prioritization)

---

## Part 5: Our Specific Choices Explained

### Why XGBoost for Engagement (Recap)

```
TASK: Predict binary outcome (engage yes/no) from 60 tabular features

✓ XGBoost is state-of-the-art for tabular classification
✓ Handles mixed feature types (numbers, categories, yes/no)
✓ Built-in handling of imbalanced classes
✓ Provides feature importance for explainability
✓ Strong regularization prevents overfitting
✓ Can be calibrated for accurate probabilities
```

### Why LightGBM for Channel Propensity (Recap)

```
TASK: Predict 4 outcomes simultaneously (phone/SMS/email/mail response)

✓ 5-10x faster than XGBoost (important for 4 models)
✓ Nearly identical accuracy (~1% difference)
✓ Handles categorical features natively
✓ Lower stakes task (ranking channels, not critical decisions)
✓ Speed advantage worth the tiny accuracy trade-off
```

### Why Rule-Based Prioritization (Recap)

```
TASK: Decide who to call first

✗ No "ground truth" labels (there's no objectively "correct" priority)
✗ Priority is a business decision, not a prediction
✗ Need to balance efficiency vs. equity (a values question)

✓ Explicit weights are transparent: "We weight SDOH at 20%"
✓ Business can adjust weights without retraining models
✓ Combines ML predictions with business knowledge
✓ Auditable and explainable to regulators
```

---

## Part 6: Glossary

Quick reference for all terms:

| Term | Definition |
|------|------------|
| **AUC-ROC** | Area Under the ROC Curve; measures how well a model ranks positive vs. negative examples (0.5 = random, 1.0 = perfect) |
| **Binary Classification** | Predicting one of two outcomes (yes/no, 0/1) |
| **Boosting** | Training models sequentially where each fixes the previous one's errors |
| **Calibration** | Adjusting model outputs so probabilities are accurate (70% means 70% actually happen) |
| **Class Imbalance** | When one outcome is much more common than another |
| **Confusion Matrix** | Table showing true positives, false positives, true negatives, false negatives |
| **F1 Score** | Harmonic mean of precision and recall; balances both metrics |
| **Feature** | An input variable used to make predictions (age, answer_rate, etc.) |
| **Feature Importance** | Measure of how much each feature contributes to predictions |
| **Gradient Boosting** | Type of boosting using gradient descent to minimize errors |
| **Label** | The outcome we're trying to predict (also called "target") |
| **LightGBM** | Fast gradient boosting library from Microsoft |
| **Multi-Output** | Predicting multiple outcomes simultaneously |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **Precision** | Of positive predictions, how many were correct? |
| **Probability** | Model's confidence in a prediction (0% to 100%) |
| **Recall** | Of actual positives, how many did we identify? |
| **Regularization** | Techniques to prevent overfitting by limiting model complexity |
| **scale_pos_weight** | XGBoost parameter to handle class imbalance |
| **Tabular Data** | Data in rows and columns (like a spreadsheet) |
| **Test Set** | Data held out to evaluate model (never seen during training) |
| **Training Set** | Data used to train the model |
| **Underfitting** | Model is too simple to capture patterns |
| **XGBoost** | Popular, highly optimized gradient boosting library |

---

## Summary

Our ML approach uses:

1. **XGBoost** for engagement prediction because it's the most accurate algorithm for tabular binary classification, with built-in handling of our challenges (class imbalance, mixed features, need for explainability).

2. **LightGBM** for channel propensity because we need to train 4 models and speed matters more than a tiny accuracy gain for this secondary task.

3. **Business rules** for prioritization because there's no "correct" answer to who should be called first—it's a values-based decision that should be transparent and adjustable.

The result is a system that's accurate, explainable, and aligned with business needs.
