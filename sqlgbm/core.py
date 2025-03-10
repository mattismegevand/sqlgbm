import pandas as pd
from functools import lru_cache
from typing import Dict, List, Optional


class SQLGBM:
  """Convert tree-based machine learning models to SQL queries.

  This class takes a trained tree-based model (currently supports LightGBM)
  and converts it to a SQL query that can be run directly in a database.

  Attributes:
    booster: The trained tree-based model.
    cat_features: List of categorical feature names.
    cat_mappings: Dictionary mapping categorical features to their values.
    tree_df: DataFrame containing the tree structure from the model.
  """

  def __init__(self, model, cat_features: Optional[List[str]] = None):
    """Initialize TreeSQL with a model and categorical features.

    Args:
      model: A trained tree-based model (currently supports LightGBM).
      cat_features: A list of categorical feature names.
    """
    self.booster = model.booster_ if hasattr(model, 'booster_') else model
    self.cat_features = cat_features or []
    self.cat_mappings = self._get_cat_mapping()
    self.tree_df = self.booster.trees_to_dataframe()
    self.tree_df['decision_type'] = self.tree_df['decision_type'].replace({'==': '=', '!=': '<>'})

  def _get_cat_mapping(self) -> Dict[str, Dict[int, str]]:
    """Get mapping from categorical feature indices to actual values.

    Returns:
      A dictionary mapping categorical feature names to their value mappings.
    """
    return {f: dict(enumerate(self.booster.pandas_categorical[i])) for i, f in enumerate(self.cat_features)}

  @lru_cache(maxsize=None)
  def _generate_tree_sql(self, tree_idx: int, node_idx: Optional[str] = None) -> str:
    """Generate SQL for a single tree.

    Args:
      tree_idx: The index of the tree.
      node_idx: The index of the node to start from. If None, starts from the root.

    Returns:
      A SQL CASE expression representing the tree's decision logic.
    """
    tree_df = self.tree_df[self.tree_df['tree_index'] == tree_idx]
    nodes = tree_df.set_index('node_index').to_dict('index')
    node_key = node_idx or f'{tree_idx}-S0'
    if node_key not in nodes: node_key = f'{tree_idx}-L0'
    node = nodes[node_key]

    if pd.isna(node['left_child']): return str(node['value'])

    feature = node['split_feature']
    escaped_feature = f'`{feature}`'

    if feature in self.cat_mappings:
      threshold = f"'{self.cat_mappings[feature][int(node['threshold'])]}'"
    else:
      threshold = node['threshold']

    operator = node['decision_type']
    condition = f"{escaped_feature} {operator} {threshold}"

    left_subtree = self._generate_tree_sql(tree_idx, node['left_child'])
    right_subtree = self._generate_tree_sql(tree_idx, node['right_child'])

    return f"CASE WHEN {condition} THEN {left_subtree} ELSE {right_subtree} END"

  def generate_query(self, table_name: str, output_type: str = 'prediction', fast_sigmoid: bool = False) -> str:
    """Generate a SQL query for the model.

    Args:
      table_name: The name of the table containing feature data.
      output_type: The type of output to generate. One of:
        - 'raw': Raw model output
        - 'probability': Probability (after sigmoid)
        - 'prediction': Binary prediction (0 or 1)
        - 'all': All three outputs
      fast_sigmoid: Whether to use a fast approximation of sigmoid.

    Returns:
      A SQL query string that implements the model's prediction logic.
    """
    tree_indices = range(self.booster.num_trees())
    tree_parts = [self._generate_tree_sql(tree_idx) for tree_idx in tree_indices]

    used_features = set(self.tree_df['split_feature'].dropna().unique())
    feature_cols = [f'`{feature}`' for feature in used_features]

    sigmoid_expr = "raw_pred / (1 + abs(raw_pred))" if fast_sigmoid else "1 / (1 + exp(-raw_pred))"

    sql = f"""
    WITH features_subset AS (
      SELECT {', '.join(feature_cols)}
      FROM {table_name}
    ), raw_prediction AS (
      SELECT ({" + ".join(tree_parts)}) AS raw_pred
      FROM features_subset
    ),
    probabilities AS (
      SELECT raw_pred, {sigmoid_expr} AS probability
      FROM raw_prediction
    )
    """

    output_map = {
      'raw': "SELECT raw_pred FROM raw_prediction",
      'probability': "SELECT probability FROM probabilities",
      'prediction': "SELECT CAST(probability > 0.5 AS INTEGER) as prediction FROM probabilities",
      'all': """SELECT raw_pred, probability, CAST(probability > 0.5 AS INTEGER) as prediction FROM probabilities"""
    }

    return sql + output_map.get(output_type, output_map['all'])