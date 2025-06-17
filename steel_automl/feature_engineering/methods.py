import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Dict, Any


def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2, interaction_only: bool = False) -> \
        Tuple[pd.DataFrame, PolynomialFeatures]:
    """
    为指定列创建多项式特征和交互特征。

    参数:
    - df: 输入的Pandas DataFrame。
    - columns: 需要创建多项式特征的列名列表。
    - degree: 多项式的次数。
    - interaction_only: 是否只生成交互特征。

    返回:
    - df_poly: 包含新多项式特征的DataFrame。
    - poly_transformer: 拟合的PolynomialFeatures对象。
    """
    if not all(col in df.columns for col in columns):
        missing_cols = [col for col in columns if col not in df.columns]
        print(f"警告: 列 {missing_cols} 不在DataFrame中，无法创建多项式特征。")
        # 返回原始DataFrame和一个未拟合的转换器
        return df, PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    # 确保所有列都是数值类型
    numeric_columns = df[columns].select_dtypes(include=np.number).columns.tolist()
    if len(numeric_columns) != len(columns):
        non_numeric_cols = list(set(columns) - set(numeric_columns))
        print(f"警告: 列 {non_numeric_cols} 不是数值类型，将从多项式特征生成中排除。")
        if not numeric_columns:  # 如果没有数值列了
            print("没有有效的数值列可供生成多项式特征。")
            return df, PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        columns_to_process = numeric_columns
    else:
        columns_to_process = columns

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

    # 处理缺失值：多项式特征对NaN敏感，这里简单用0填充，实际应用中应由预处理步骤处理
    data_for_poly = df[columns_to_process].fillna(0)

    poly_features = poly.fit_transform(data_for_poly)

    # 获取特征名称
    # PolynomialFeatures.get_feature_names_out() 在新版本sklearn中可用
    # 如果用旧版本，可能需要手动构造或使用 poly.get_feature_names(input_features=columns_to_process)
    try:
        poly_feature_names = poly.get_feature_names_out(input_features=columns_to_process)
    except AttributeError:  # 兼容旧版本
        # 这是一个简化的名称生成，可能不完全准确，特别是对于交互项
        # 实际项目中应确保sklearn版本支持 get_feature_names_out
        base_names = [f"poly_{name}" for name in columns_to_process]
        poly_feature_names = [f"{name}_deg{d}" for name in base_names for d in range(1, degree + 1)]  # 简化版
        print("警告: 使用简化的多项式特征命名，请考虑升级scikit-learn以获得更准确的名称。")

    df_poly_features = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    # 合并回原始DataFrame，避免重复列 (如果原始列也被包含在输出中)
    # PolynomialFeatures默认会包含原始特征 (x^1)
    # 先删除原始列，再合并新的多项式特征DataFrame
    # df_without_original_poly_cols = df.drop(columns=columns_to_process, errors='ignore')
    # df_combined = pd.concat([df_without_original_poly_cols, df_poly_features], axis=1)

    # 或者，更简单地，直接将新特征添加到原df，如果poly_features包含原始特征，则它们会覆盖
    # 但更好的做法是让poly_features只包含新增的项。
    # PolynomialFeatures(include_bias=False) 默认不包含偏置项 (常数1)
    # 如果degree=1且interaction_only=False, 它会返回原始特征。
    # 如果degree>1, 它会包含原始特征、平方项、交互项等。

    # 为避免与原列名冲突，并将新特征加入
    df_out = df.copy()
    for col_name in df_poly_features.columns:
        # 如果新特征名与原始列名相同 (例如 x1)，并且我们不想覆盖，可以重命名
        # 但通常PolynomialFeatures生成的名称是唯一的，如 "col1", "col1^2", "col1 col2"
        df_out[col_name] = df_poly_features[col_name]

    print(f"为列 {columns_to_process} 创建了 {len(poly_feature_names)} 个多项式/交互特征 (degree={degree})。")
    return df_out, poly


def create_ratio_features(df: pd.DataFrame, numerator_col: str, denominator_col: str, new_col_name: str,
                          epsilon: float = 1e-6) -> pd.DataFrame:
    """
    创建两个数值列的比率特征。

    参数:
    - df: 输入的Pandas DataFrame。
    - numerator_col: 分子列名。
    - denominator_col: 分母列名。
    - new_col_name: 新生成的比率特征列名。
    - epsilon: 防止除以零的小常数。

    返回:
    - 包含新比率特征的DataFrame。
    """
    if numerator_col not in df.columns or denominator_col not in df.columns:
        print(f"警告: 列 '{numerator_col}' 或 '{denominator_col}' 不在DataFrame中，无法创建比率特征 '{new_col_name}'。")
        return df
    if not pd.api.types.is_numeric_dtype(df[numerator_col]) or not pd.api.types.is_numeric_dtype(df[denominator_col]):
        print(f"警告: 列 '{numerator_col}' 或 '{denominator_col}' 不是数值类型，无法创建比率特征 '{new_col_name}'。")
        return df

    # 处理分母为0或接近0的情况
    df[new_col_name] = df[numerator_col] / (df[denominator_col] + epsilon)
    # 处理原始数据中的NaN，比率结果也应为NaN
    df.loc[df[numerator_col].isnull() | df[denominator_col].isnull(), new_col_name] = np.nan

    print(f"创建了比率特征 '{new_col_name}' ({numerator_col} / {denominator_col})。")
    return df


def apply_custom_domain_formula(df: pd.DataFrame, formula_name: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    应用特定的领域知识公式。
    这是一个示例，实际应用中需要根据具体公式实现。

    参数:
    - df: 输入的Pandas DataFrame。
    - formula_name: 公式名称，例如 "carbon_equivalent_ceq"。
    - params: 公式所需的参数，例如 {"C_col": "feature_C", "Mn_col": "feature_Mn", ...}

    返回:
    - 包含新领域特征的DataFrame。
    """
    if formula_name == "carbon_equivalent_ceq":
        # CEQ = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15 (简化版，不同标准公式不同)
        try:
            C = df[params["C_col"]].fillna(0)  # 假设缺失的成分含量为0，或由预处理决定
            Mn = df[params["Mn_col"]].fillna(0)
            # 其他元素类似处理，如果公式需要的话
            # Cr = df[params.get("Cr_col", "non_existent_col")].fillna(0) # 使用.get处理可选参数

            df["CEQ_calculated"] = C + Mn / 6
            # # 更完整的示例:
            # Cr_Mo_V_sum = df.get(params.get("Cr_col"), pd.Series(0, index=df.index)).fillna(0) + \
            #                 df.get(params.get("Mo_col"), pd.Series(0, index=df.index)).fillna(0) + \
            #                 df.get(params.get("V_col"), pd.Series(0, index=df.index)).fillna(0)
            # Ni_Cu_sum = df.get(params.get("Ni_col"), pd.Series(0, index=df.index)).fillna(0) + \
            #             df.get(params.get("Cu_col"), pd.Series(0, index=df.index)).fillna(0)
            # df["CEQ_calculated"] = C + Mn/6 + Cr_Mo_V_sum/5 + Ni_Cu_sum/15
            print(f"应用了领域公式 '{formula_name}'，生成了特征 'CEQ_calculated'。")
        except KeyError as e:
            print(f"警告: 计算 '{formula_name}' 失败，缺少列: {e}。参数: {params}")
        except Exception as e:
            print(f"警告: 计算 '{formula_name}' 时发生错误: {e}。参数: {params}")

    elif formula_name == "example_interaction":  # 另一个示例
        try:
            col1 = df[params["col1"]].fillna(0)
            col2 = df[params["col2"]].fillna(0)
            df["interaction_custom"] = col1 * col2
            print(f"应用了领域公式 '{formula_name}'，生成了特征 'interaction_custom'。")
        except KeyError as e:
            print(f"警告: 计算 '{formula_name}' 失败，缺少列: {e}。参数: {params}")
    else:
        print(f"警告: 未知的领域公式名称 '{formula_name}'。")

    return df


FEATURE_ENGINEERING_METHODS_MAP = {
    "create_polynomial_features": create_polynomial_features,
    "create_ratio_features": create_ratio_features,
    "apply_custom_domain_formula": apply_custom_domain_formula,
    # 可以添加更多特征工程方法
}

if __name__ == '__main__':
    data = {
        'A': np.random.rand(10) * 5,
        'B': np.random.rand(10) * 10 + 1,  # 确保B不为0
        'C': np.random.rand(10) * 2,
        'target': np.random.rand(10)
    }
    sample_df = pd.DataFrame(data)
    sample_df.loc[1, 'A'] = np.nan  # 引入一个NaN

    print("原始数据:")
    print(sample_df)

    # 测试多项式特征
    df_poly, _ = create_polynomial_features(sample_df.copy(), columns=['A', 'B'], degree=2)
    print("\n创建 A, B 的2次多项式特征后:")
    print(df_poly.head())

    # 测试比率特征
    df_ratio = create_ratio_features(sample_df.copy(), numerator_col='A', denominator_col='B', new_col_name='A_div_B')
    print("\n创建 A/B 比率特征后:")
    print(df_ratio[['A', 'B', 'A_div_B']].head())

    # 测试领域公式 (模拟CEQ)
    # 为了测试，我们重命名列以匹配预期的参数名
    df_domain_test = sample_df.rename(columns={'A': 'feature_C', 'B': 'feature_Mn'})
    df_domain = apply_custom_domain_formula(
        df_domain_test.copy(),
        formula_name="carbon_equivalent_ceq",
        params={"C_col": "feature_C", "Mn_col": "feature_Mn"}
    )
    print("\n应用模拟CEQ公式后:")
    if "CEQ_calculated" in df_domain.columns:
        print(df_domain[['feature_C', 'feature_Mn', 'CEQ_calculated']].head())
    else:
        print("CEQ_calculated 未能成功生成。")
