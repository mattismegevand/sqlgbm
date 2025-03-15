import pandas as pd
from functools import lru_cache
from typing import Optional, Union


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

  def __init__(
    self, model, X: Optional[pd.DataFrame] = None, cat_features: Optional[list[str]] = None, cat_mappings: Optional[dict[str, dict[str, int]]] = None
  ):
    """Initialize SQLGBM with a tree-based model.

    Args:
      model: A trained tree-based model (supports LightGBM and XGBoost).
      X: Optional DataFrame used for inferring categorical features.
      cat_features: Optional list of categorical feature names.
      cat_mappings: Optional dictionary mapping categorical features to their values.
    """
    self._initialize_model(model)
    self.cat_mappings = cat_mappings or self._construct_cat_mappings(cat_features or X)
    self.tree_df = self.booster.trees_to_dataframe()
    self._process_tree_dataframe()

  def _initialize_model(self, model):
    if "lightgbm" in str(model.__class__):
      self.booster = model.booster_ if hasattr(model, "booster_") else model
      self.model_type = "lightgbm"
    elif "xgboost" in str(model.__class__):
      self.booster = model.get_booster() if hasattr(model, "get_booster") else model
      self.model_type = "xgboost"
    else:
      raise ValueError(f"Unsupported model type: {model.__class__}")

  def _process_tree_dataframe(self):
    if self.model_type == "lightgbm":
      self._process_lightgbm_dataframe()
    else:
      self._process_xgboost_dataframe()
    self.tree_df["threshold"] = self.tree_df.apply(
      lambda x: f"'{self.cat_mappings[x['split_feature']][int(x['threshold'])]}'" if x["split_feature"] in self.cat_mappings else x["threshold"],
      axis=1,
    )

  def _process_lightgbm_dataframe(self):
    self.tree_df["decision_type"] = self.tree_df["decision_type"].replace({"==": "=", "!=": "<>"})

  def _process_xgboost_dataframe(self):
    self.tree_df.rename(
      {
        "Tree": "tree_index",
        "ID": "node_index",
        "Feature": "split_feature",
        "Yes": "left_child",
        "No": "right_child",
        "Split": "threshold",
        "Gain": "value",
      },
      axis=1,
      inplace=True,
    )
    self.tree_df.loc[self.tree_df["split_feature"] == "Leaf", "split_feature"] = pd.NA
    self.tree_df["decision_type"] = self.tree_df.apply(lambda x: "=" if x["split_feature"] in self.cat_mappings else "<", axis=1)
    self.tree_df["threshold"] = self.tree_df.apply(lambda x: x["Category"][0] if x["split_feature"] in self.cat_mappings else x["threshold"], axis=1)

  def _construct_cat_mappings(self, inp: Union[pd.DataFrame, list[str]]) -> dict[str, dict[str, int]]:
    if isinstance(inp, pd.DataFrame):
      return {f: dict(enumerate(inp[f].cat.categories)) for f in inp.columns if inp[f].dtype == "category"}
    return {f: dict(enumerate(self.booster.pandas_categorical[i])) for i, f in enumerate(inp)}

  @lru_cache(maxsize=None)
  def _generate_tree_sql(self, tree_idx: int, node_idx: Optional[str] = None) -> str:
    tree_df = self.tree_df[self.tree_df["tree_index"] == tree_idx]
    nodes = tree_df.set_index("node_index").to_dict("index")
    node_key = node_idx
    if node_key not in nodes:
      match self.model_type:
        case "lightgbm":
          node_key = tree_df[tree_df["node_depth"] == 1]["node_index"].values[0]
        case "xgboost":
          node_key = f"{tree_idx}-0"
    node = nodes[node_key]

    if pd.isna(node["left_child"]):
      return str(node["value"])

    feature = node["split_feature"]
    escaped_feature = f"`{feature}`"
    threshold = node["threshold"]
    operator = node["decision_type"]
    condition = f"{escaped_feature} {operator} {threshold}"

    left_subtree = self._generate_tree_sql(tree_idx, node["left_child"])
    right_subtree = self._generate_tree_sql(tree_idx, node["right_child"])

    return f"CASE WHEN {condition} THEN {left_subtree} ELSE {right_subtree} END"

  def generate_query(self, table_name: str, output_type: str = "prediction", fast_sigmoid: bool = False) -> str:
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
    tree_indices = self.tree_df["tree_index"].unique()
    tree_parts = [self._generate_tree_sql(tree_idx) for tree_idx in tree_indices]

    used_features = set(self.tree_df["split_feature"].dropna().unique())
    feature_cols = [f"`{feature}`" for feature in used_features]

    sigmoid_expr = "raw_pred / (1 + abs(raw_pred))" if fast_sigmoid else "1 / (1 + exp(-raw_pred))"

    sql = f"""
    WITH features_subset AS (
      SELECT {", ".join(feature_cols)}
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
      "raw": "SELECT raw_pred FROM raw_prediction",
      "probability": "SELECT probability FROM probabilities",
      "prediction": "SELECT CAST(probability > 0.5 AS INTEGER) as prediction FROM probabilities",
      "all": """SELECT raw_pred, probability, CAST(probability > 0.5 AS INTEGER) as prediction FROM probabilities""",
    }

    return sql + output_map.get(output_type, output_map["all"])
