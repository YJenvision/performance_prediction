import pandas as pd
import ibm_db
import sqlalchemy
from typing import Dict, Any, Optional, List, Tuple, Generator
from datetime import datetime, timedelta
from llm_utils import call_llm
from prompts.prompt_manager import get_prompt

"""
数据获取模块，主要负责从数据库或Excel文件加载数据（列名进行大写转换）（需要手动从performance_model_builder.py中进行切换），包含SQL动态生成、数据库连接管理和数据读取等功能。使用生成器模式实现流式数据处理，便于前端实时展示进度。
"""


def generate_sql_query(
        sg_sign: Optional[List[str]],
        start_time: str,
        end_time: str,
        product_unit_no: Optional[List[str]],
        st_no: Optional[List[str]],
        steel_grade: Optional[List[str]]
) -> Generator[Dict[str, Any], None, str]:
    """
    动态生成SQL查询。
    返回：生成器对象，流式返回生成过程和最终SQL查询。
    """
    system_prompt = get_prompt('data_loader.generate_sql.system')
    user_prompt = f"""现在，严格按照系统提示中的输出要求，从表 BGTAMAQA.T_ADS_FACT_PCDPF_INTEGRATION_INFO 中查询数据，要求如下：
1. 查询所有列 (SELECT *)
2. 时间范围条件：REC_REVISE_TIME BETWEEN '{start_time}' AND '{end_time}'"""

    condition_num = 3

    def format_in_clause(values: List[str]) -> str:
        """
        将列表转换为SQL的IN子句格式
        """
        return "({})".format(", ".join(f"'{v}'" for v in values))

    if sg_sign:
        user_prompt += f"\n{condition_num}. 牌号条件：SG_SIGN IN {format_in_clause(sg_sign)}"
        condition_num += 1
    if product_unit_no:
        user_prompt += f"\n{condition_num}. 机组号条件：PRODUCT_UNIT_NO IN {format_in_clause(product_unit_no)}"
        condition_num += 1
    if st_no:
        user_prompt += f"\n{condition_num}. 出钢记号条件：ST_NO IN {format_in_clause(st_no)}"
        condition_num += 1
    if steel_grade:
        user_prompt += f"\n{condition_num}. 钢种条件：钢种代码通过 SUBSTR(SIGN_CODE, 5, 2) IN {format_in_clause(steel_grade)} 进行匹配"

    sql_query_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
    sql_query = ""
    while True:
        try:
            chunk = next(sql_query_gen)
            if chunk.get("type") == "error":
                yield chunk
                # 返回一个表示失败的信号
                return "AGENT_FAILED"
            yield chunk
        except StopIteration as e:
            sql_query = e.value
            break

    if "AGENT_FAILED" in sql_query:
        return sql_query  # 传递失败信号，call_llm中返回的值为"AGENT_FAILED"被捕获

    select_pos = sql_query.upper().find("SELECT")
    if select_pos > 0:
        sql_query = sql_query[select_pos:]
    sql_query = sql_query.strip().rstrip(';')

    if any(keyword in sql_query.upper() for keyword in
           ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE']):
        raise ValueError("生成的SQL语句包含不安全的关键字，已拒绝执行。")

    return sql_query


class DataLoader:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.db_handle = None
        self.engine = None

    def _connect(self) -> bool:
        """连接到数据库。"""
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
                pass

            return True
        except Exception as e:
            self._disconnect()
            return False

    def _disconnect(self):
        """断开数据库连接。"""
        try:
            if self.engine:
                self.engine.dispose()
            if self.db_handle and ibm_db.active(self.db_handle):
                ibm_db.close(self.db_handle)
        finally:
            self.db_handle = None
            self.engine = None

    def fetch_data(self, sg_sign: Optional[List[str]], target_metric: str, time_range: str,
                   product_unit_no: Optional[List[str]], st_no: Optional[List[str]],
                   steel_grade: Optional[List[str]]
                   ) -> Generator[Dict[str, Any], None, Tuple[Optional[pd.DataFrame], Optional[str]]]:
        """
        从数据库获取数据，生成器。
        """

        current_stage = "数据收集"
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "初始化数据加载器..."}}

        if not self._connect():
            yield {"type": "error", "payload": {"stage": current_stage,
                                                "detail": "数据库连接失败，请检查网络连接或数据库配置。"}}
            return None, None

        query = None
        try:
            try:
                start_time, end_time = time_range.split('-')
            except (ValueError, AttributeError):
                now = datetime.now()
                end_time = now.strftime("%Y%m%d")
                start_time = (now - timedelta(days=365)).strftime("%Y%m%d")
                detail = f"在意图识别阶段提供的时间范围格式无效，使用默认值: {start_time}-{end_time}。"
                yield {"type": "status_update",
                       "payload": {"stage": current_stage, "status": "running", "detail": detail}}

            # 从子生成器流式传输并获取返回的查询语句
            query_gen = generate_sql_query(sg_sign, start_time, end_time, product_unit_no, st_no, steel_grade)
            query = ""
            while True:
                try:
                    chunk = next(query_gen)
                    if chunk.get("type") == "error":
                        yield chunk
                        return None, None  # 终止
                    yield chunk
                except StopIteration as e:
                    query = e.value
                    break

            if "AGENT_FAILED" in query:
                yield {"type": "error", "payload": {"stage": current_stage,
                                                    "detail": "智能体未能成功生成SQL查询。"}}
                return None, None

            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "生成的SQL查询",
                "data": query
            }}

            yield {"type": "status_update",
                   "payload": {"stage": current_stage, "status": "running", "detail": "正在执行数据查询获取数据..."}}

            df = pd.read_sql(query, self.engine)
            df.columns = df.columns.str.upper()

            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "获取的数据概览",
                "data": f"成功获取 {len(df)} 行, {len(df.columns)} 列数据。"
            }}


            return df, query

        except Exception as e:
            error_details = f"从数据库获取数据时发生错误: {e}"
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": error_details}}
            print(error_details)
            return None, query
        finally:
            self._disconnect()

    def fetch_data_from_excel(self, path: str
                              ) -> Generator[Dict[str, Any], None, Tuple[Optional[pd.DataFrame], Optional[str]]]:
        """
        从excel中获取数据，生成器。
        """
        current_stage = "数据收集"
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "初始化数据加载器..."}}

        query = "excel文件中获取数据"
        try:
            yield {"type": "status_update",
                   "payload": {"stage": current_stage, "status": "running",
                               "detail": "正在从excel文件中获取数据..."}}

            df = pd.read_excel(path)
            df.columns = df.columns.str.upper()

            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "数据概览",
                "data": f"成功从excel文件获取 {len(df)} 行, {len(df.columns)} 列数据。"
            }}

            return df, query

        except Exception as e:
            error_details = f"从excel文件中获取数据时发生错误: {e}"
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": error_details}}
            return None, query
