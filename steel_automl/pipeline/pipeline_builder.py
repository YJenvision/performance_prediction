from typing import List, Dict, Any, Optional
import json
import datetime
import os


class PipelineBuilder:
    def __init__(self, user_request_details: Dict[str, Any]):
        """
        初始化Pipeline构建器。

        参数:
        - user_request_details: 包含用户请求信息的字典 (sg_sign, target_metric, etc.)
        """
        self.user_request_details = user_request_details
        self.pipeline_stages: List[Dict[str, Any]] = []
        self.start_time = datetime.datetime.now()
        self.end_time: Optional[datetime.datetime] = None
        self.final_status: str = "pending"
        self.model_name: Optional[str] = None

    def add_stage(self, stage_name: str, status: str, details: Optional[Dict[str, Any]] = None,
                  artifacts: Optional[Dict[str, Any]] = None):
        """
        向Pipeline中添加一个阶段的记录。

        参数:
        - stage_name: 阶段名称 (例如 "data_preprocessing", "model_training")。
        - status: 该阶段的执行状态 ("success", "failed", "skipped")。
        - details: 关于该阶段执行的详细信息或日志。
        - artifacts: 该阶段产生的关键产物描述或引用 (例如文件路径或SQL查询)。
        """
        stage_record = {
            "stage_name": stage_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": status,
            "details": details if details is not None else {},
            "artifacts": artifacts if artifacts is not None else {}
        }
        self.pipeline_stages.append(stage_record)
        print(f"Pipeline Stage Added: '{stage_name}', Status: '{status}'")

    def set_final_status(self, status: str):
        """设置整个Pipeline的最终状态。"""
        self.final_status = status
        self.end_time = datetime.datetime.now()

    def set_model_name(self, model_name: str):
        """
        设置模型名称，用于生成标准化的pipeline_id。
        """
        self.model_name = model_name

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        获取整个Pipeline的摘要信息，并使用标准化命名生成pipeline_id。
        """
        if self.end_time is None:
            self.end_time = datetime.datetime.now()

        # 标准化 Pipeline ID 生成逻辑
        # 命名标准: 目标性能_数据时间范围_牌号_机组_出钢记号_钢种_时间_模型算法
        def format_param(param_value: Any) -> str:
            """Helper to format list or single value to string."""
            if param_value is None:
                return ""
            if isinstance(param_value, list):
                return "-".join(map(str, param_value))
            return str(param_value)

        details = self.user_request_details
        id_parts = [
            format_param(details.get("target_metric")),
            format_param(details.get("time_range")),
            format_param(details.get("sg_sign")),
            format_param(details.get("product_unit_no")),
            format_param(details.get("st_no")),
            format_param(details.get("steel_grade")),
            self.start_time.strftime('%Y%m%d%H%M%S'),
            self.model_name,
        ]
        # 过滤掉None和空字符串部分，用下划线连接
        pipeline_id = "_".join(filter(None, id_parts))
        # 替换掉可能存在于参数中的非法文件名字符
        pipeline_id = pipeline_id.replace(" ", "").replace("/", "-")

        summary = {
            "pipeline_id": pipeline_id,
            "user_intent": self.user_request_details,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "final_status": self.final_status,
            "stages": self.pipeline_stages
        }
        return summary
