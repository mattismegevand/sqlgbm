Examples
========

Basic Usage
----------

.. code-block:: python

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