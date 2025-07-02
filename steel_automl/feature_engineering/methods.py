import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Dict, Any


def apply_knowledge_based_formula(df: pd.DataFrame, formula_template: str, new_feature_name: str,
                                  column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    应用基于领域知识的公式，使用动态映射的列名。

    参数:
    - df: 输入的Pandas DataFrame。
    - formula_template: 包含占位符的公式字符串, e.g., "{C} + {Mn}/6"。
    - new_feature_name: 新特征的列名。
    - column_mapping: 一个字典，将公式占位符映射到DataFrame中的实际列名。
                      e.g., {"C": "ELM_C", "Mn": "ELM_MN"}

    返回:
    - 包含新领域特征的DataFrame。
    """
    df_out = df.copy()

    # 检查所有映射后的列是否存在于DataFrame中
    required_cols = list(column_mapping.values())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告: 计算 '{new_feature_name}' 失败，因为缺少列: {missing_cols}。")
        return df

    # 使用 Series.map() 和字典来替换公式模板中的占位符为实际列名
    # 这是比 str.format() 更安全、更灵活的方式，可以处理复杂的列名
    eval_expression = formula_template
    for placeholder, actual_col in column_mapping.items():
        # 在eval表达式中，列名不能包含特殊字符，如果包含需要用反引号``包围
        # 但这里我们假设预处理后的列名是合法的Python标识符
        eval_expression = eval_expression.replace(f"{{{placeholder}}}", actual_col)

    print(f"正在计算 '{new_feature_name}'，使用表达式: '{eval_expression}'")

    try:
        # 使用 pandas.eval() 来安全、高效地执行表达式
        # 我们需要提供一个本地字典，其中包含列名作为变量
        # fillna(0) 是一个简化处理，实际应在预处理阶段完成
        local_dict = {col: df[col].fillna(0) for col in required_cols}
        df_out[new_feature_name] = pd.eval(eval_expression, local_dict=local_dict, engine='python')
    except Exception as e:
        print(f"警告: 计算特征 '{new_feature_name}' 时发生错误: {e}")
        return df  # 返回原始DataFrame

    print(f"=> 成功创建领域知识特征 '{new_feature_name}'。")
    return df_out


def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2, interaction_only: bool = False) -> \
Tuple[pd.DataFrame, PolynomialFeatures]:
    """为指定列创建多项式特征和交互特征。"""
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        raise ValueError(f"列 {missing_cols} 不在DataFrame中，无法创建多项式特征。")

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    data_for_poly = df[columns].fillna(0)  # 简单填充，应在预处理中完成
    poly_features = poly.fit_transform(data_for_poly)
    poly_feature_names = poly.get_feature_names_out(input_features=columns)

    df_poly_features = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    df_out = pd.concat([df, df_poly_features], axis=1)
    # 删除原始列，因为它们已经作为一阶特征包含在多项式特征中
    df_out = df_out.drop(columns=columns)

    print(f"=> 为列 {columns} 创建了 {len(poly_feature_names)} 个多项式/交互特征。")
    return df_out, poly


def create_ratio_features(df: pd.DataFrame, numerator_col: str, denominator_col: str, new_col_name: str,
                          epsilon: float = 1e-6) -> pd.DataFrame:
    """创建两个数值列的比率特征。"""
    if numerator_col not in df.columns or denominator_col not in df.columns:
        raise ValueError(f"列 '{numerator_col}' 或 '{denominator_col}' 不在DataFrame中。")

    df_out = df.copy()
    df_out[new_col_name] = df[numerator_col] / (df[denominator_col] + epsilon)
    df_out.loc[df[numerator_col].isnull() | df[denominator_col].isnull(), new_col_name] = np.nan

    print(f"=> 创建了比率特征 '{new_col_name}'。")
    return df_out


# 将所有方法映射到一个字典，方便FeatureGenerator调用
FEATURE_ENGINEERING_METHODS_MAP = {
    "apply_knowledge_based_formula": apply_knowledge_based_formula,
    "create_polynomial_features": create_polynomial_features,
    "create_ratio_features": create_ratio_features,
}
