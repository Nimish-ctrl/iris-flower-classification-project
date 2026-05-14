\# 🌸 Iris Flower Classification



A multi-model classification project on the classic Iris dataset, comparing a Neural Network, Decision Tree, and Random Forest — with a proper train/validation/test split to avoid data leakage during hyperparameter tuning.



\---



\## 📌 Overview



Most Iris tutorials just throw everything into a train-test split and call it done. This project adds a \*\*validation set\*\* to tune hyperparameters (tree depth, number of estimators) honestly before final evaluation on a held-out test set.



Three models are trained and compared:



\- \*\*Neural Network\*\* (TensorFlow/Keras) — two hidden layers with ReLU, softmax output, AdamW optimizer

\- \*\*Decision Tree\*\* — max depth tuned via validation set

\- \*\*Random Forest\*\* — number of estimators tuned via validation set



\---



\## 📁 Project Structure



```

iris-classify.ipynb   # Main notebook (EDA → preprocessing → training → evaluation)

random\_forest\_model.pkl

decision\_tree\_model.pkl

iris\_nn\_model.keras

scaler.pkl

encoder.pkl

```



\---



\## 🔍 Exploratory Data Analysis



\- Pairplot colored by class label to visualize feature separability

\- Grouped bar charts comparing mean feature values across Setosa, Versicolour, and Virginica

\- Feature statistics via `describe()`



\---



\## ⚙️ Preprocessing



| Step | Detail |

|---|---|

| Label encoding | `LabelEncoder` on class labels |

| Train/Val/Test split | 60% / 20% / 20% with stratification |

| Feature scaling | `StandardScaler` fit on train only, applied to val and test |

| One-hot encoding | Applied to labels for the neural network (`to\_categorical`) |



\---



\## 🧠 Models



\### Neural Network

\- Architecture: `Input(4) → Dense(16, ReLU) → Dense(16, ReLU) → Dense(3, Softmax)`

\- Optimizer: AdamW | Loss: Categorical Crossentropy

\- Trained for 50 epochs with batch size 4

\- Validation data used during training to monitor overfitting



\### Decision Tree

\- `max\_depth` swept from 1–9 on the validation set

\- Best depth selected, final model retrained and evaluated on test set



\### Random Forest

\- `n\_estimators` swept from 1–9 on the validation set

\- Best estimator count selected, final model evaluated on test set



\---



\## 📊 Results



| Model | Test Accuracy |

|---|---|

| Neural Network | evaluated via `model.evaluate()` |

| Decision Tree | evaluated via `model.score()` |

| Random Forest | evaluated via `model.score()` |



\*(Run the notebook to see exact values — they vary slightly with random seeds.)\*



\---



\## 💾 Saved Artifacts



All models and preprocessors are saved for inference:



```python

import joblib

from tensorflow.keras.models import load\_model



rf = joblib.load("random\_forest\_model.pkl")

dt = joblib.load("decision\_tree\_model.pkl")

nn = load\_model("iris\_nn\_model.keras")

scaler = joblib.load("scaler.pkl")

encoder = joblib.load("encoder.pkl")

```



\---



\## 🚀 Running the Notebook



```bash

pip install numpy pandas seaborn matplotlib scikit-learn tensorflow

jupyter notebook iris-classify.ipynb

```



> Originally built on Kaggle — if running locally, update the dataset path in the `read\_csv` call.



\---



\## 🛠 Tech Stack



`Python` · `TensorFlow/Keras` · `scikit-learn` · `Pandas` · `NumPy` · `Seaborn` · `Matplotlib` · `joblib`

