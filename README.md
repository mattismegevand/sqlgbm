## sqlgbm

⚠️ Warning: This library is in a very early development stage. The API and functionality will improve significantly over time. Not ready for production use yet.

sqlgbm is a Python library that converts tree-based machine learning models into SQL queries.
This allows you to deploy your ML models directly in your database without any additional infrastructure.

![sqlgbm in action](assets/image.png)

### installation
```bash
pip install sqlgbm
```

### overview

sqlgbm takes your trained tree-based models and generates SQL code that reproduces the model's predictions. This enables you to:

- Run predictions directly in your database
- Eliminate latency from API calls between your database and ML serving infrastructure
- Simplify your production architecture by removing additional serving components

### supported models

Currently supported models:
- LightGBM
- XGBoost

### usage

#### basic example

```python
from sqlgbm import SQLGBM
import lightgbm as lgb
import pandas as pd

# Load titanic dataset
titanic = pd.read_csv('titanic.csv')
features = ['pclass', 'sex', 'age', 'fare']
X = titanic[features]
X['sex'] = X['sex'].astype('category')
y = titanic['survived']

# Train model
clf = lgb.LGBMClassifier(n_estimators=3, max_depth=3)
clf.fit(X, y, categorical_feature=['sex'])

# Convert to SQL
sqlgbm = SQLGBM(clf, cat_features=['sex'])
sql = sqlgbm.generate_query('titanic', 'probability')

print(sql)
```

#### xgboost example

```python
import xgboost as xgb
from sqlgbm import SQLGBM

# Prepare data and train model
# ...

# Convert XGBoost model to SQL
model = xgb.XGBClassifier(n_estimators=3, max_depth=3, base_score=0.5)
model.fit(X, y)

sqlgbm = SQLGBM(model, X=X)  # X used to infer categorical features
sql = sqlgbm.generate_query('my_table', 'all')
```

#### output types

sqlgbm supports different output formats through the `output_type` parameter:

- `raw`: Returns the raw model output
- `probability`: Returns the probability (after sigmoid transformation)
- `prediction`: Returns the binary prediction (0 or 1) based on a 0.5 threshold
- `all`: Returns all three outputs

Additional options:
- `fast_sigmoid`: Use a faster approximation of the sigmoid function

## roadmap

- [ ] Add support for CatBoost
- [ ] Optimize SQL generation for large models
- [ ] Add support for multiclass classification
- [ ] Provide specialized optimizations for different database engines

## license

MIT
