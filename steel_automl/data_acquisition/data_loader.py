import pandas as pd
import ibm_db
from typing import Dict, Any, Optional
import sqlalchemy
from llm_utils import call_llm


def generate_sql_query(sg_sign, start_time, end_time, product_unit_no, st_no) -> str:
    """
    动态生成SQL查询语句。

    参数:
    - sg_sign: 牌号，如果为空字符串则不添加该条件。
    - start_time: 开始时间。
    - end_time: 结束时间。
    - product_unit_no: 机组号，如果为空字符串则不添加该条件。
    - st_no: 出钢记号，如果为空字符串则不添加该条件。

    返回:
    - 生成的SQL查询语句。
    """
    # 构建系统提示词
    system_prompt = """你是一个专业的SQL生成助手，你的任务是根据提供的参数生成安全的SQL查询语句。请遵循以下规则：
1. 只生成SELECT类型的查询语句，在任何情况下都坚决不允许生成任何修改数据库的语句（如INSERT、UPDATE、DELETE等）
2. 不使用任何高级SQL特性，如存储过程、触发器等
3. 不允许执行任何系统命令或访问系统表
4. 返回的SQL语句必须是格式良好的，可以直接执行的
5. 只返回生成的目标SQL语句，不要包含任何解释或注释"""

    # 构建用户提示词
    user_prompt = f"""请生成一个SQL查询语句，从表 BGTAMAQA.T_ADS_FACT_PCDPF_INTEGRATION_INFO 中查询数据，要求如下：
1. 查询所有列 (SELECT *)
2. 时间范围条件：REC_REVISE_TIME BETWEEN '{start_time}' AND '{end_time}'"""

    # 根据参数情况动态添加条件
    condition_num = 3
    if sg_sign is not None:
        user_prompt += f"\n{condition_num}. 牌号条件：SG_SIGN = '{sg_sign}'"
        condition_num += 1

    if product_unit_no is not None:
        user_prompt += f"\n{condition_num}. 机组号条件：PRODUCT_UNIT_NO = '{product_unit_no}'"
        condition_num += 1

    if st_no is not None:
        user_prompt += f"\n{condition_num}. 出钢记号条件：ST_NO = '{st_no}'"

    # 生成SQL语句
    sql_query = call_llm(system_prompt, user_prompt, model="ds_v3")

    # 删除SELECT之前的所有内容（经测试使用R1模型会出现其他内容输出）
    select_pos = sql_query.upper().find("SELECT")
    if select_pos > 0:
        sql_query = sql_query[select_pos:]

    # 简单检查生成的SQL是否安全
    if any(keyword in sql_query.upper() for keyword in
           ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']):
        print("生成的SQL语句包含不安全的关键字，停止执行。")
        sql_query = "生成的SQL语句包含不安全的关键字，停止执行。"

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
        self.db_handle = None  # IBM DB2连接句柄（用于保持连接活跃）
        self.engine = None  # SQLAlchemy引擎

    def _connect(self) -> bool:
        """
        连接到数据库。
        使用SQLAlchemy连接到数据库。
        """
        print(f"尝试连接到数据库: {self.db_config.get('database')} on {self.db_config.get('host')}")
        try:
            # 获取数据库连接信息
            db_username = self.db_config.get('user')
            db_password = self.db_config.get('password')
            db_host = self.db_config.get('host')
            db_port = self.db_config.get('port')
            database = self.db_config.get('database')

            # 创建IBM DB2连接（用于保持连接活跃）
            db = ibm_db.connect(
                "DATABASE={0};".format(database) +
                "HOSTNAME={0};".format(db_host) +
                "PORT={0};".format(db_port) +
                "PROTOCOL=TCPIP;" +
                "UID={0};".format(db_username) +
                "PWD={0};".format(db_password),
                "",
                ""
            )
            self.db_handle = db

            # 创建SQLAlchemy引擎
            connection_string = f"db2+ibm_db://{db_username}:{db_password}@{db_host}:{db_port}/{database}"
            self.engine = sqlalchemy.create_engine(connection_string)

            # 测试
            with self.engine.connect() as conn:
                pass

            print("SQLAlchemy引擎创建成功。")
            print("数据库连接成功。")
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            if self.db_handle and ibm_db.active(self.db_handle):
                ibm_db.close(self.db_handle)
            self.db_handle = None
            self.engine = None
            return False

    def _disconnect(self):
        """断开数据库连接。"""
        try:
            if self.db_handle and ibm_db.active(self.db_handle):  # 检查连接是否活跃
                ibm_db.close(self.db_handle)
                print("IBM DB2连接已断开。")
            if self.engine:
                self.engine.dispose()
                print("SQLAlchemy引擎已释放。")
        except Exception as e:
            print(f"断开连接时发生错误: {e}")
        finally:
            self.db_handle = None
            self.engine = None

    def fetch_data(self, sg_sign, target_metric, time_range, product_unit_no, st_no) -> Optional[pd.DataFrame]:
        """
        根据牌号、目标指标、时间范围和机组号动态地从数据库获取数据。

        参数:
        - sg_sign: 牌号，如果为空字符串则不添加该条件。
        - target_metric: 目标性能指标列名。
        - time_range: 数据时间范围 (如 "20250401-20230501")。
        - product_unit_no: 机组号，如果为空字符串则不添加该条件。
        - st_no: 出钢记号，如果为空字符串则不添加该条件。

        返回:
        - 包含特征和目标指标的Pandas DataFrame
        - 失败则返回None
        """
        if not self.engine and not self._connect():
            print("无法连接到数据库，数据获取失败。")
            return None

        # 动态构建调试信息
        debug_info = "开始获取数据:"
        if sg_sign is not None:
            debug_info += f" 牌号为'{sg_sign}',"
        if target_metric is not None:
            debug_info += f" 目标指标为'{target_metric}',"
        if time_range is not None:
            debug_info += f" 时间范围为'{time_range}',"
        if product_unit_no is not None:
            debug_info += f" 机组号为'{product_unit_no}',"
        if st_no is not None:
            debug_info += f" 出钢记号为'{st_no}'"

        # 去除末尾可能多余的逗号
        debug_info = debug_info.rstrip(',')
        print(debug_info)

        try:

            time_parts = time_range.split('-')
            if len(time_parts) == 2:
                start_time = time_parts[0]  # 开始时间
                end_time = time_parts[1]  # 结束时间
            else:
                # 如果格式不符合预期，使用当前时间作为结束时间，当前时间减一年作为开始时间
                from datetime import datetime, timedelta
                now = datetime.now()
                end_time = now.strftime("%Y%m%d")
                start_time = (now - timedelta(days=365)).strftime("%Y%m%d")
                # print(f"时间范围格式无效，使用默认时间范围: {start_time}-{end_time}")

        except Exception as e:
            # 如果解析失败，同样使用当前时间作为结束时间，当前时间减一年作为开始时间
            from datetime import datetime, timedelta
            now = datetime.now()
            end_time = now.strftime("%Y%m%d")
            start_time = (now - timedelta(days=365)).strftime("%Y%m%d")
            # print(f"解析时间范围失败: {e}, 使用默认时间范围: {start_time}-{end_time}")

        # 动态生成SQL查询
        query = generate_sql_query(sg_sign, start_time, end_time, product_unit_no, st_no)

        try:
            # 使用SQLAlchemy引擎执行查询并获取数据
            df = pd.read_sql(query, self.engine)
            print("使用SQLAlchemy引擎获取数据成功。")
            print(f"成功获取 {len(df)} 条数据。")

            # 预处理：将列名全部转换为大写
            df.columns = df.columns.str.upper()
            return df

        except Exception as e:
            print(f"从数据库获取数据失败: {e}")
            return None

        finally:
            self._disconnect()

    def fetch_data_from_excel(self, sg_sign, target_metric, time_range, product_unit_no, st_no) -> Optional[pd.DataFrame]:
        """
        从Excel文件获取数据。

        返回:
        - 包含特征和目标指标的Pandas DataFrame
        - 失败返回None
        """
        print(f"开始从Excel文件获取数据。")

        try:
            df = pd.read_excel(r"D:\Desktop\性能预报智能体测试_Q235B_20250101-20250501.xlsx")
            print(f"成功从Excel文件获取 {len(df)} 条数据。")

            # 预处理：将列名全部转换为大写
            df.columns = df.columns.str.upper()
            return df

        except Exception as e:
            print(f"从Excel文件获取数据失败: {e}")
            return None
