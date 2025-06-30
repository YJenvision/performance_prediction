import pandas as pd
import ibm_db
from typing import Dict, Any, Optional, List, Union
import sqlalchemy
from llm_utils import call_llm


def generate_sql_query(
        sg_sign: Optional[List[str]],
        start_time: str,
        end_time: str,
        product_unit_no: Optional[List[str]],
        st_no: Optional[List[str]],
        steel_grade: Optional[List[str]]
) -> str:
    """
    动态生成SQL查询语句，能处理单个或多个值的条件。

    参数:
    - sg_sign: 牌号列表。
    - start_time: 开始时间。
    - end_time: 结束时间。
    - product_unit_no: 机组号列表。
    - st_no: 出钢记号列表。
    - c: 钢种列表。

    返回:
    - 生成的SQL查询语句。
    """
    system_prompt = """你是一个专业的SQL生成助手，你的任务是根据提供的参数生成安全的SQL查询语句。请遵循以下规则：
1. 只生成SELECT类型的查询语句，在任何情况下都坚决不允许生成任何修改数据库的语句（如INSERT、UPDATE、DELETE等）
2. 不使用任何高级SQL特性，如存储过程、触发器等
3. 不允许执行任何系统命令或访问系统表
4. 返回的SQL语句必须是格式良好的，可以直接执行的
5. 只返回生成的目标SQL语句，不要包含任何解释或注释"""

    user_prompt = f"""请生成一个SQL查询语句，从表 BGTAMAQA.T_ADS_FACT_PCDPF_INTEGRATION_INFO 中查询数据，要求如下：
1. 查询所有列 (SELECT *)
2. 时间范围条件：REC_REVISE_TIME BETWEEN '{start_time}' AND '{end_time}'"""

    condition_num = 3

    # 辅助函数，用于格式化IN子句的值
    def format_in_clause(values: List[str]) -> str:
        return "({})".format(", ".join(f"'{v}'" for v in values))

    # 根据参数情况动态添加条件
    if sg_sign:
        if len(sg_sign) == 1:
            user_prompt += f"\n{condition_num}. 牌号条件：SG_SIGN = '{sg_sign[0]}'"
        else:
            user_prompt += f"\n{condition_num}. 牌号条件：SG_SIGN IN {format_in_clause(sg_sign)}"
        condition_num += 1

    if product_unit_no:
        if len(product_unit_no) == 1:
            user_prompt += f"\n{condition_num}. 机组号条件：PRODUCT_UNIT_NO = '{product_unit_no[0]}'"
        else:
            user_prompt += f"\n{condition_num}. 机组号条件：PRODUCT_UNIT_NO IN {format_in_clause(product_unit_no)}"
        condition_num += 1

    if st_no:
        if len(st_no) == 1:
            user_prompt += f"\n{condition_num}. 出钢记号条件：ST_NO = '{st_no[0]}'"
        else:
            user_prompt += f"\n{condition_num}. 出钢记号条件：ST_NO IN {format_in_clause(st_no)}"
        condition_num += 1

    if steel_grade:
        if len(steel_grade) == 1:
            user_prompt += f"\n{condition_num}. 钢种条件：钢种代码为'{steel_grade[0]}'，需要通过SUBSTR(SIGN_CODE, 5, 2) = '{steel_grade[0]}'进行匹配"
        else:
            user_prompt += f"\n{condition_num}. 钢种条件：钢种代码为'{', '.join(steel_grade)}'，需要通过SUBSTR(SIGN_CODE, 5, 2) IN {format_in_clause(steel_grade)}进行匹配"

    sql_query = call_llm(system_prompt, user_prompt, model="ds_v3")

    # 删除SELECT之前的所有内容（使用R1可能会响应其他内容，比如在语句之前添加 'sql'）
    select_pos = sql_query.upper().find("SELECT")
    if select_pos > 0:
        sql_query = sql_query[select_pos:]

    # 去除末尾可能的分号
    sql_query = sql_query.strip().rstrip(';')

    # 规则检查生成的SQL是否安全
    if any(keyword in sql_query.upper() for keyword in
           ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']):
        raise ValueError("取数阶段生成的SQL语句包含不安全的关键字，停止执行。")

    print(f"动态生成的SQL查询: {sql_query}")
    return sql_query


class DataLoader:
    def __init__(self, db_config: Dict[str, Any]):
        """
        初始化数据加载器。

        参数:
        - db_config: 数据库连接配置字典。
        """
        self.db_config = db_config
        self.db_handle = None
        self.engine = None

    def _connect(self) -> bool:
        """连接到数据库。"""
        print(f"尝试连接到数据库: {self.db_config.get('database')} on {self.db_config.get('host')}")
        try:
            db_username, db_password = self.db_config.get('user'), self.db_config.get('password')
            db_host, db_port = self.db_config.get('host'), self.db_config.get('port')
            database = self.db_config.get('database')

            dsn = (
                f"DATABASE={database};"
                f"HOSTNAME={db_host};"
                f"PORT={db_port};"
                f"PROTOCOL=TCPIP;"
                f"UID={db_username};"
                f"PWD={db_password};"
            )
            self.db_handle = ibm_db.connect(dsn, "", "")

            connection_string = f"db2+ibm_db://{db_username}:{db_password}@{db_host}:{db_port}/{database}"
            self.engine = sqlalchemy.create_engine(connection_string)

            with self.engine.connect() as conn:
                pass  # 测试连接

            print("数据库连接成功。")
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            self._disconnect()
            return False

    def _disconnect(self):
        """断开数据库连接。"""
        try:
            if self.engine:
                self.engine.dispose()
                print("SQLAlchemy引擎已释放。")
            if self.db_handle and ibm_db.active(self.db_handle):
                ibm_db.close(self.db_handle)
                print("IBM DB2连接已断开。")
        except Exception as e:
            print(f"断开连接时发生错误: {e}")
        finally:
            self.db_handle = None
            self.engine = None

    def fetch_data(self, sg_sign: Optional[List[str]], target_metric: str, time_range: str,
                   product_unit_no: Optional[List[str]], st_no: Optional[List[str]],
                   steel_grade: Optional[List[str]]) -> Optional[pd.DataFrame]:
        """
        根据牌号、目标指标、时间范围、机组号、出钢记号和钢种条件动态地从数据库获取数据。

        参数:
        - sg_sign: 牌号列表。
        - target_metric: 目标性能指标列名。
        - time_range: 数据时间范围。
        - product_unit_no: 机组号列表。
        - st_no: 出钢记号列表。
        - steel_grade: 钢种列表（对应SIGN_CODE字段的第5-6位字符）。

        返回:
        - 包含特征和目标指标的Pandas DataFrame或None。
        """
        if not self._connect():
            print("无法连接到数据库，数据获取失败。")
            return None

        # 动态构建调试信息
        debug_info = "开始获取数据:"
        if sg_sign:
            debug_info += f" 牌号为'{', '.join(sg_sign)}',"
        if target_metric:
            debug_info += f" 目标指标为'{target_metric}',"
        if time_range:
            debug_info += f" 时间范围为'{time_range}',"
        if product_unit_no:
            debug_info += f" 机组号为'{', '.join(product_unit_no)}',"
        if st_no:
            debug_info += f" 出钢记号为'{', '.join(st_no)}',"
        if steel_grade:
            debug_info += f" 钢种为'{', '.join(steel_grade)}'"

        print(debug_info.rstrip(','))

        try:
            start_time, end_time = time_range.split('-')
        except ValueError:
            from datetime import datetime, timedelta
            now = datetime.now()
            end_time = now.strftime("%Y%m%d")
            start_time = (now - timedelta(days=365)).strftime("%Y%m%d")
            print(f"时间范围格式无效，使用默认时间范围: {start_time}-{end_time}")

        try:
            # 动态生成SQL查询
            query = generate_sql_query(sg_sign, start_time, end_time, product_unit_no, st_no, steel_grade)
            df = pd.read_sql(query, self.engine)
            print(f"成功获取 {len(df)} 条数据。")
            # 列名大写
            df.columns = df.columns.str.upper()

            # 如果指定了钢种条件，添加钢种列用于验证和分析
            if steel_grade and 'SIGN_CODE' in df.columns:
                df['STEEL_GRADE'] = df['SIGN_CODE'].str[4:6]
                print(f"已提取钢种信息到STEEL_GRADE列，共{df['STEEL_GRADE'].nunique()}种不同钢种")

            return df
        except Exception as e:
            print(f"从数据库获取数据失败: {e}")
            return None
        finally:
            self._disconnect()

    def fetch_data_from_excel(self, sg_sign, target_metric, time_range, product_unit_no, st_no, steel_grade) -> \
            Optional[pd.DataFrame]:
        """
        从Excel文件获取数据，支持钢种条件过滤。

        参数:
        - sg_sign: 牌号列表。
        - target_metric: 目标性能指标列名。
        - time_range: 数据时间范围。
        - product_unit_no: 机组号列表。
        - st_no: 出钢记号列表。
        - steel_grade: 钢种列表（对应SIGN_CODE字段的第5-6位字符）。

        返回:
        - 过滤后的Pandas DataFrame或None。
        """
        print("开始从Excel文件获取数据。")
        try:
            df = pd.read_excel(r"D:\Desktop\性能预报智能体测试_Q235B_20250101-20250501.xlsx")
            print(f"成功从Excel文件获取 {len(df)} 条数据。")

            # 列名大写
            df.columns = df.columns.str.upper()
            return df

        except Exception as e:
            print(f"从Excel文件获取数据失败: {e}")
            return None
