#!/usr/bin/env python3

import argparse
import lightgbm as lgb
import os
import pandas as pd
import polars as pl
import random
import time

from sqlgbm import SQLGBM

def run_benchmark(args):
  """Run a simple benchmark comparing TreeSQL to native LightGBM."""
  print("Loading data...")
  titanic = pd.read_csv(os.path.join(os.path.dirname(__file__), '../assets/titanic.csv'))
  titanic['age'].fillna(titanic['age'].median(), inplace=True)
  titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)
  features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
  X = titanic[features]
  y = titanic['survived']
  X['sex'] = X['sex'].astype('category')
  X['embarked'] = X['embarked'].astype('category')
  cat_features = ['sex', 'embarked']

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

  print("Training model...")
  clf = lgb.LGBMClassifier(n_estimators=args.n_estimators,
  max_depth=args.max_depth,
  verbose=-1)
  clf.fit(X_train, y_train, categorical_feature=cat_features)

  print("Converting to SQL...")
  sqlgbm = SQLGBM(clf, cat_features)
  sql_query = sqlgbm.generate_query('self', 'probability')
  if args.show_query: print(f"\nGenerated SQL Query: {sql_query}")

  print("\nRunning benchmark...\n")
  df_pl = pl.from_pandas(X_test)

  start_time = time.time()
  y_prob = clf.predict_proba(X_test)
  lgb_time = time.time() - start_time

  start_time = time.time()
  y_prob_sql = df_pl.sql(sql_query)
  sql_time = time.time() - start_time

  assert (abs(y_prob[:,1] - y_prob_sql.to_numpy().reshape(-1)) < 1e-4).all()
  y_pred = (y_prob[:,1] > 0.5).astype(int)
  accuracy = (y_test == y_pred).mean()
  assert accuracy > 0.8

  print(f"{'=' * 50}")
  print("BENCHMARK RESULTS")
  print(f"{'=' * 50}")
  print("Dataset: Titanic")
  print(f"Model: LightGBM with {clf.n_estimators} trees")
  print(f"Max depth: {args.max_depth or 'unlimited'}")
  print(f"Test samples: {len(X_test)}")
  print(f"{'=' * 50}")
  print(f"LightGBM prediction time: {lgb_time:.6f} seconds")
  print(f"SQL query prediction time: {sql_time:.6f} seconds")
  print(f"SQL/LightGBM ratio: {sql_time/lgb_time:.2f}x")
  print(f"{'=' * 50}")
  print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run TreeSQL benchmark')
  parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of trees (default: None - unlimited)')
  parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees (default: 100)')
  parser.add_argument('--show-query', action='store_true', default=False, help='Show the generated SQL query')
  args = parser.parse_args()
  run_benchmark(args)
