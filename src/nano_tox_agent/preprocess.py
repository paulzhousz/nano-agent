"""训练前特征预处理流水线。

数值列做缺失值填补和标准化，类别列做缺失值填补和独热编码，
确保训练与推理走同一套 sklearn 预处理逻辑。
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nano_tox_agent.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor() -> ColumnTransformer:
    """构建可复用的列转换器，供分类器和回归器共享。"""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
