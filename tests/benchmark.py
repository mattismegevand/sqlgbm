#!/usr/bin/env python3

import argparse
import lightgbm as lgb
import os
import pandas as pd
import polars as pl
import random
import time
import xgboost as xgb

from sqlgbm import SQLGBM


def run_benchmark(args):
  """Run a simple benchmark comparing TreeSQL to native LightGBM and XGBoost."""
  print("Loading data...")
  titanic = pd.read_csv(os.path.join(os.path.dirname(__file__), "../assets/titanic.csv"))
  titanic["age"] = titanic["age"].fillna(titanic["age"].median())
  titanic["embarked"] = titanic["embarked"].fillna(titanic["embarked"].mode()[0])
  features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
  X = titanic[features].copy()
  y = titanic["survived"]
  X["sex"] = X["sex"].astype("category")
  X["embarked"] = X["embarked"].astype("category")

  n = len(X)
  test_size = int(0.2 * n)
  indices = list(range(n))
  random.seed(42)
  random.shuffle(indices)
  test_indices = indices[:test_size]
  train_indices = indices[test_size:]

  X_train = X.iloc[train_indices]
  X_test = X.iloc[test_indices]
  y_train = y.iloc[train_indices]
  y_test = y.iloc[test_indices]

  print("\nTraining LightGBM model...")
  clf_lgb = lgb.LGBMClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, verbose=-1)
  clf_lgb.fit(X_train, y_train, categorical_feature=["sex", "embarked"])

  print("Training XGBoost model...")
  clf_xgb = xgb.XGBClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, enable_categorical=True, base_score=0.5)
  clf_xgb.fit(X_train, y_train)

  print("\nConverting to SQL...")
  sqlgbm_lgb = SQLGBM(clf_lgb, X_train)
  sqlgbm_xgb = SQLGBM(clf_xgb, X_train)
  sql_query_lgb = sqlgbm_lgb.generate_query("self", "probability")
  sql_query_xgb = sqlgbm_xgb.generate_query("self", "probability")

  if args.show_query:
    print(f"\nGenerated LightGBM SQL Query: {sql_query_lgb}")
    print(f"\nGenerated XGBoost SQL Query: {sql_query_xgb}")

  print("\nRunning benchmark...\n")
  df_pl = pl.from_pandas(X_test)

  start_time = time.time()
  y_prob_lgb = clf_lgb.predict_proba(X_test)
  lgb_time = time.time() - start_time

  start_time = time.time()
  y_prob_xgb = clf_xgb.predict_proba(X_test)
  xgb_time = time.time() - start_time

  start_time = time.time()
  y_prob_sql_lgb = df_pl.sql(sql_query_lgb)
  sql_time_lgb = time.time() - start_time

  start_time = time.time()
  y_prob_sql_xgb = df_pl.sql(sql_query_xgb)
  sql_time_xgb = time.time() - start_time

  assert (abs(y_prob_lgb[:, 1] - y_prob_sql_lgb.to_numpy().reshape(-1)) < 1e-4).all()
  # assert (abs(y_prob_xgb[:,1] - y_prob_sql_xgb.to_numpy().reshape(-1)) < 1e-4).all()

  y_pred_lgb = (y_prob_lgb[:, 1] > 0.5).astype(int)
  y_pred_xgb = (y_prob_xgb[:, 1] > 0.5).astype(int)
  accuracy_lgb = (y_test == y_pred_lgb).mean()
  accuracy_xgb = (y_test == y_pred_xgb).mean()
  assert accuracy_lgb > 0.8
  assert accuracy_xgb > 0.8

  print(f"{'=' * 50}")
  print("BENCHMARK RESULTS")
  print(f"{'=' * 50}")
  print("Dataset: Titanic")
  print(f"Trees per model: {args.n_estimators}")
  print(f"Max depth: {args.max_depth or 'unlimited'}")
  print(f"Test samples: {len(X_test)}")
  print(f"{'=' * 50}")
  print("LightGBM:")
  print(f"Native prediction time: {lgb_time:.6f} seconds")
  print(f"SQL query prediction time: {sql_time_lgb:.6f} seconds")
  print(f"SQL/Native ratio: {sql_time_lgb / lgb_time:.2f}x")
  print(f"Accuracy: {accuracy_lgb:.2f}")
  print(f"{'=' * 50}")
  print("XGBoost:")
  print(f"Native prediction time: {xgb_time:.6f} seconds")
  print(f"SQL query prediction time: {sql_time_xgb:.6f} seconds")
  print(f"SQL/Native ratio: {sql_time_xgb / xgb_time:.2f}x")
  print(f"Accuracy: {accuracy_xgb:.2f}")
  print(f"{'=' * 50}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run TreeSQL benchmark")
  parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of trees (default: None - unlimited)")
  parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees (default: 100)")
  parser.add_argument("--show-query", action="store_true", default=False, help="Show the generated SQL query")
  args = parser.parse_args()
  run_benchmark(args)
