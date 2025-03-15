#!/usr/bin/env python3

import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import polars as pl
import unittest
import xgboost as xgb

from sqlgbm import SQLGBM

titanic = pd.read_csv(os.path.join(os.path.dirname(__file__), "../assets/titanic.csv"))
titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["embarked"] = titanic["embarked"].fillna(titanic["embarked"].mode()[0])
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = titanic[features].copy()
y = titanic["survived"]
X["sex"] = X["sex"].astype("category")
X["embarked"] = X["embarked"].astype("category")
cat_features = ["sex", "embarked"]
test_size = 0.2
n_samples = len(X)
n_test = int(test_size * n_samples)
np.random.seed(42)
test_indices = np.random.choice(n_samples, n_test, replace=False)
train_indices = np.array([i for i in range(n_samples) if i not in test_indices])
X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

clf = lgb.LGBMClassifier(n_estimators=10, max_depth=10, verbose=-1)
clf.fit(X_train, y_train, categorical_feature=cat_features)

xgb_clf = xgb.XGBClassifier(n_estimators=10, max_depth=10, enable_categorical=True, base_score=0.5)
xgb_clf.fit(X_train, y_train)

sqlgbm = SQLGBM(clf, cat_features)
sqlgbm_xgb = SQLGBM(xgb_clf, X_train)
table_name = "self"
sql_query = sqlgbm.generate_query(table_name)
sql_query_proba = sqlgbm.generate_query(table_name, output_type="probability")
sql_query_xgb = sqlgbm_xgb.generate_query(table_name)
sql_query_proba_xgb = sqlgbm_xgb.generate_query(table_name, output_type="probability")


class TestTreeSQL(unittest.TestCase):
  def test_initialization(self):
    self.assertEqual(sqlgbm.booster, clf.booster_)
    self.assertEqual(sqlgbm.cat_mappings, sqlgbm_xgb.cat_mappings)
    self.assertIsNotNone(sqlgbm.cat_mappings)

  def test_cat_mapping(self):
    cat_mappings = sqlgbm.cat_mappings
    self.assertEqual(set(cat_mappings.keys()), set(cat_features))
    for cat in cat_features:
      unique_values = set(X_train[cat].astype(str).unique())
      mapped_values = set(str(v) for v in cat_mappings[cat].values())
      self.assertLessEqual(len(unique_values), len(mapped_values))

  def test_generate_query(self):
    query = sqlgbm.generate_query(table_name, output_type="raw")
    self.assertIn("SELECT raw_pred FROM raw_prediction", query)

    query = sqlgbm.generate_query(table_name, output_type="probability")
    self.assertIn("SELECT probability FROM probabilities", query)

    query = sqlgbm.generate_query(table_name, output_type="prediction")
    self.assertIn("SELECT CAST(probability > 0.5 AS INTEGER) as prediction FROM probabilities", query)

    query = sqlgbm.generate_query(table_name, output_type="all")
    self.assertIn("SELECT raw_pred, probability, CAST(probability > 0.5 AS INTEGER) as prediction FROM probabilities", query)

  def test_prediction_correctness_lgb(self):
    y_pred_model = clf.predict(X_test)
    df_pl = pl.from_pandas(X_test)
    y_pred_sql = df_pl.sql(sql_query)["prediction"].to_list()
    accuracy = sum(a == b for a, b in zip(y_pred_model, y_pred_sql)) / len(y_pred_model)
    self.assertGreater(accuracy, 0.99)

    y_prob_model = clf.predict_proba(X_test)[:, 1]
    y_prob_sql = df_pl.sql(sql_query_proba)["probability"].to_list()
    mean_diff = np.mean(np.abs(y_prob_model - y_prob_sql))
    self.assertLess(mean_diff, 0.01)

  def test_prediction_correctness_xgb(self):
    y_pred_model = xgb_clf.predict(X_test)
    df_pl = pl.from_pandas(X_test)
    y_pred_sql = df_pl.sql(sql_query_xgb)["prediction"].to_list()
    accuracy = sum(a == b for a, b in zip(y_pred_model, y_pred_sql)) / len(y_pred_model)
    self.assertGreater(accuracy, 0.99)

    y_prob_model = xgb_clf.predict_proba(X_test)[:, 1]
    y_prob_sql = df_pl.sql(sql_query_proba_xgb)["probability"].to_list()
    mean_diff = np.mean(np.abs(y_prob_model - y_prob_sql))
    self.assertLess(mean_diff, 0.01)


if __name__ == "__main__":
  unittest.main()
