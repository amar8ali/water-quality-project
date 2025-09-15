# Water Quality AI Project

## Description
This project analyzes water potability data using machine learning to predict water safety. It includes data preprocessing, exploratory analysis, model development, and performance evaluation.

## Team Information
| AC.NO | Name           | Role           | Contributions                       |
|-------|----------------|----------------|--------------------------------------|
| 1     |Amar Ali Muathar| Lead Developer | Data preprocessing, model development|
| 2     | Student Name 2 | Data Analyst   | EDA, visualization                   |
| 3     | Student Name 3 | ML Engineer    | Model optimization, deployment       |

## Installation and Setup
### Prerequisites
- Python 3.12 or higher
- UV package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <https://github.com/amar8ali/water-quality-project.git>
   cd water-quality
   ```
2. Install dependencies using UV:
   ```bash
   uv sync
   ```
3. Run the project:
   ```bash
   uv run python main.py
   ```

## Project Structure
```
water-quality/
├── README.md
├── pyproject.toml
├── .python-version
├── main.py
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── notebooks/
├── data/
└── docs/
```

## Usage
### Basic Usage
```python
from src.models import YourModel
model = YourModel()
model.train(data)
```
### Running Experiments
```bash
uv run python experiments/train_model.py
```

## Results

- Model Accuracy:
    - Random Forest: 0.67
    - Logistic Regression: 0.63
    - SVM: 0.69
- Key Findings:
    - SVM and Random Forest performed best for this dataset.
    - Logistic Regression struggled with class imbalance and did not predict potable water well.
    - Class imbalance affected recall for class 1 (potable water).
    - Further improvements can be made by balancing classes and tuning model hyperparameters.

### Detailed Classification Reports

#### Random Forest
```
              precision    recall  f1-score   support

           0       0.70      0.85      0.77       412
           1       0.60      0.38      0.46       244

    accuracy                           0.67       656
   macro avg       0.65      0.61      0.61       656
weighted avg       0.66      0.67      0.65       656
```

#### Logistic Regression
```
              precision    recall  f1-score   support

           0       0.63      1.00      0.77       412
           1       0.00      0.00      0.00       244

    accuracy                           0.63       656
   macro avg       0.31      0.50      0.39       656
weighted avg       0.39      0.63      0.48       656
```

#### SVM
```
              precision    recall  f1-score   support

           0       0.69      0.91      0.79       412
           1       0.68      0.32      0.44       244

    accuracy                           0.69       656
   macro avg       0.69      0.62      0.61       656
weighted avg       0.69      0.69      0.66       656
```

## Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request
