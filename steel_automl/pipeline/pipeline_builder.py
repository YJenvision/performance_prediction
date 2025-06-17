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

    def add_stage(self, stage_name: str, status: str, details: Optional[Dict[str, Any]] = None,
                  artifacts: Optional[Dict[str, Any]] = None):
        """
        向Pipeline中添加一个阶段的记录。

        参数:
        - stage_name: 阶段名称 (例如 "data_preprocessing", "model_training")。
        - status: 该阶段的执行状态 ("success", "failed", "skipped")。
        - details: 关于该阶段执行的详细信息或日志。
        - artifacts: 该阶段产生的关键产物描述或引用 (例如 "fitted_preprocessor_object_id")。
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

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        获取整个Pipeline的摘要信息。
        """
        if self.end_time is None:
            self.end_time = datetime.datetime.now()  # 如果未明确设置，则以当前时间为准

        summary = {
            "pipeline_id": f"automl_{self.start_time.strftime('%Y%m%d%H%M%S')}_{self.user_request_details.get('sg_sign', 'unknown_sg')}",
            "user_request": self.user_request_details,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "final_status": self.final_status,
            "stages": self.pipeline_stages
        }
        return summary

    def save_pipeline_summary(self, filepath: Optional[str] = None) -> str:
        """
        将Pipeline摘要保存到JSON文件。

        参数:
        - filepath: 保存文件的路径。如果为None，则生成默认文件名。

        返回:
        - 实际保存的文件路径。
        """
        summary = self.get_pipeline_summary()
        if filepath is None:
            # 确保results目录存在 (假设此文件在 pipeline/ 目录下)
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results_output")
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir, f"pipeline_summary_{summary['pipeline_id']}.json")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4, ensure_ascii=False, default=str)  # default=str for datetime etc.
            print(f"Pipeline摘要已保存到: {filepath}")
        except Exception as e:
            print(f"保存Pipeline摘要失败: {e}")
        return filepath