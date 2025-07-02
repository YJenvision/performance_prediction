import faiss
import pickle
import numpy as np
import os
from typing import List, Dict, Any, Optional
from config import KB_DIR
from llm_utils import get_embedding


class KnowledgeBaseService:
    def __init__(self, knowledge_base_name: str):
        """
        初始化知识库服务。

        参数:
        - knowledge_base_name: 知识库的名称 (不含扩展名), 如 "data_preprocessing_kb"。
        """
        self.kb_name = knowledge_base_name
        self.index_path = os.path.join(KB_DIR, f"{knowledge_base_name}.faiss")
        self.metadata_path = os.path.join(KB_DIR, f"{knowledge_base_name}.pkl")
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_kb()

    def _load_kb(self):
        """加载Faiss索引和元数据。"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(
                    f"知识库 '{self.kb_name}' 加载成功. 索引中文档数: {self.index.ntotal if self.index else 0}, 元数据条目数: {len(self.metadata)}")
            except Exception as e:
                print(f"加载知识库 '{self.kb_name}' 失败: {e}")
                self.index = None  # 确保失败时index为None
                self.metadata = []
        else:
            print(f"知识库文件未找到: {self.index_path} 或 {self.metadata_path}. 将创建一个空的知识库结构。")
            # 即使文件不存在，也初始化为空列表，避免后续操作出错
            self.metadata = []
            # Faiss索引需要维度信息，这里暂时不创建，等有数据添加时再创建
            # 或者可以创建一个空的IndexFlatL2，但需要知道维度

    def add_documents(self, documents: List[Dict[str, Any]], texts_for_embedding: List[str]):
        """
        向知识库中添加新文档。
        注意：这是一个简化的添加过程，实际应用中可能需要更复杂的更新策略。
        为简化，这里假设每次添加都是重建索引。

        参数:
        - documents: 文档列表，每个文档是一个包含元数据的字典。
        - texts_for_embedding: 与documents对应的用于生成嵌入的文本列表。
        """
        if len(documents) != len(texts_for_embedding):
            print("错误: 文档数量和待嵌入文本数量不匹配。")
            return

        new_embeddings_list = []
        valid_documents = []
        for doc, text in zip(documents, texts_for_embedding):
            embedding = get_embedding(text)
            if embedding is not None:
                new_embeddings_list.append(embedding)
                valid_documents.append(doc)
            else:
                print(f"警告:未能为文本 '{text[:50]}...' 生成嵌入，该文档将不被添加。")

        if not new_embeddings_list:
            print("没有有效的嵌入生成，知识库未更新。")
            return

        new_embeddings = np.array(new_embeddings_list).astype('float32')

        if self.index is None or self.index.ntotal == 0:  # 如果索引不存在或为空
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)  # 采用L2距离
            # self.index = faiss.IndexFlatIP(dimension) # 内积，如果嵌入是归一化的，则等价于余弦相似度

        if new_embeddings.shape[1] != self.index.d:
            print(f"错误: 新嵌入的维度 ({new_embeddings.shape[1]}) 与现有索引的维度 ({self.index.d}) 不匹配。")
            return

        self.index.add(new_embeddings)
        self.metadata.extend(valid_documents)

        try:
            os.makedirs(KB_DIR, exist_ok=True)  # 确保目录存在
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)
            print(f"知识库 '{self.kb_name}' 更新并保存成功。当前文档数: {self.index.ntotal}")
        except Exception as e:
            print(f"保存知识库 '{self.kb_name}' 失败: {e}")

    def search(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        在知识库中搜索相关文档。

        参数:
        - query_text: 查询文本。
        - k: 返回最相似的文档数量。

        返回:
        - 相关文档的元数据列表。
        """
        if self.index is None or self.index.ntotal == 0:
            print(f"知识库 '{self.kb_name}' 为空或未加载，无法执行搜索。")
            return []

        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            print(f"未能为查询文本 '{query_text[:20]}...' 生成嵌入，搜索中止。")
            return []

        query_vector = np.array([query_embedding]).astype('float32')

        if query_vector.shape[1] != self.index.d:
            print(f"错误: 查询嵌入的维度 ({query_vector.shape[1]}) 与索引的维度 ({self.index.d}) 不匹配。")
            return []

        try:
            distances, indices = self.index.search(query_vector, k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(self.metadata):  # 确保索引有效
                    results.append({
                        "metadata": self.metadata[idx],
                        "similarity_score": 1 / (1 + distances[0][i]) if distances[0][i] >= 0 else -1
                        # 简单转换为相似度，L2距离越小越相似
                    })
                else:
                    # 目前经常返回，待检查什么情况。
                    print(f"警告: 搜索返回了无效的索引 {idx}。")
            return results
        except Exception as e:
            print(f"在知识库 '{self.kb_name}' 中搜索失败: {e}")
            return []


# 创建和使用知识库实例 (通常这个初始化过程在一个单独的脚本中完成)
def _initialize_sample_kb(kb_name, sample_data):
    """
    一个用于创建示例知识库的辅助函数，未来需要根据知识库构建的结构重写，目前是JSON形式的文本测试数据。
    """
    print(f"\n正在初始化示例知识库: {kb_name}...")
    # 检查KB_DIR是否存在，如果不存在则创建
    if not os.path.exists(KB_DIR):
        os.makedirs(KB_DIR)
        print(f"创建目录: {KB_DIR}")

    # 删除已存在的旧文件，以便重新生成
    # 目前没有增量更新机制，只能全部重新构建索引
    index_file = os.path.join(KB_DIR, f"{kb_name}.faiss")
    meta_file = os.path.join(KB_DIR, f"{kb_name}.pkl")
    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(meta_file):
        os.remove(meta_file)

    kb_service = KnowledgeBaseService(kb_name)  # 这会尝试加载，如果不存在则为空

    docs_to_add = []
    texts_for_embedding = []
    for item in sample_data:
        docs_to_add.append(item["metadata"])
        texts_for_embedding.append(item["text_for_embedding"])

    kb_service.add_documents(docs_to_add, texts_for_embedding)


if __name__ == "__main__":
    # --- 初始化示例知识库 (知识库后续通过独立脚本维护) ---

    # 0. 业务数据知识库
    professional_knowledge_samples = [
        {
            "text_for_embedding": "目标性能指标字段映射对照列表。",
            "metadata": [
                {
                    "standard_name": "屈服延伸率",
                    "field_code": "YS_EL_N",
                    "aliases": ["屈服延伸", "屈服伸长率"],
                    "description": "材料在屈服阶段的延伸率，反映材料的塑性变形能力。"
                },
                {
                    "standard_name": "屈服强度",
                    "field_code": "RP02_N",
                    "aliases": ["0.2%强度", "RP0.2"],
                    "description": "规定非比例延伸率为0.2%时的应力值，是评价钢材强度的重要指标，广泛用于没有明显屈服现象的材料。"
                },
                {
                    "standard_name": "上屈服强度",
                    "field_code": "TOP_YS_N",
                    "aliases": ["上屈服"],
                    "description": "材料在拉伸试验中屈服现象过程中出现的最大应力值，表征材料开始塑性变形时能承受的最高应力。"
                },
                {
                    "standard_name": "下屈服强度",
                    "field_code": "BOT_YS_N",
                    "aliases": ["下屈服"],
                    "description": "材料在拉伸试验中屈服现象过程中出现的最小应力值，是结构设计中常用的强度指标，确保安全性。"
                },
                {
                    "standard_name": "抗拉强度",
                    "field_code": "TS_N",
                    "aliases": ["抗拉", "抗拉能力"],
                    "description": "材料在拉伸试验中能承受的最大拉应力，表征材料的极限承载能力，是钢材分级和选用的重要依据。"
                },
                {
                    "standard_name": "断裂延伸率",
                    "field_code": "BREAK_EL_N",
                    "aliases": ["断后延伸", "断后伸长率"],
                    "description": "试样拉断时的延伸率，反映材料的塑性和韧性，是评价钢材冷加工性能和焊接性能的重要指标。"
                },
                {
                    "standard_name": "均匀延伸率",
                    "field_code": "AGT_EL_N",
                    "aliases": ["均匀延伸", "均匀伸长率"],
                    "description": "试样达到最大拉力时的延伸率，表征材料在均匀塑性变形阶段的延展能力，是评价冷成形性能的关键指标。"
                }
            ]
        },
        {
            "text_for_embedding": "目标性能指标字段列表，用于数据预处理时使用，除了识别除了当前建模目标外数据中包含的其他潜在目标标签列。",
            "metadata": {
                "type": "目标性能指标字段列表",
                "description": "钢铁产品常见的性能指标，在建模时通常作为目标变量(Y)，因此除了当前任务选定的目标外，在用户没有强制要求保留的情况下，其余的都应从训练字段(X)中剔除。",
                "fields": [
                    {"field_code": "YS_EL_N", "standard_name": "屈服延伸率", "aliases": ["屈服延伸", "屈服伸长率"]},
                    {"field_code": "RP02_N", "standard_name": "屈服强度", "aliases": ["0.2%强度", "RP0.2"]},
                    {"field_code": "TOP_YS_N", "standard_name": "上屈服强度", "aliases": ["上屈服"]},
                    {"field_code": "BOT_YS_N", "standard_name": "下屈服强度", "aliases": ["下屈服"]},
                    {"field_code": "TS_N", "standard_name": "抗拉强度", "aliases": ["抗拉", "抗拉能力"]},
                    {"field_code": "BREAK_EL_N", "standard_name": "断裂延伸率", "aliases": ["断后延伸", "断后伸长率"]},
                    {"field_code": "AGT_EL_N", "standard_name": "均匀延伸率", "aliases": ["均匀延伸", "均匀伸长率"]}
                ]
            }
        },
        {
            "text_for_embedding": "经验意义上不适用作训练字段(X)的特征列，例如唯一标识符、高基数类别变量等，这些特征在用户没有强制要求保留的情况下，通常需要被直接删除。",
            "metadata": {
                "type": "经验意义上不适用作训练字段(X)的特征列列表",
                "description": "这类特征通常不包含对预测目标有用的通用信息，或者会给模型带来噪声和过拟合风险，这些特征在用户没有强制要求保留的情况下，建议在预处理早期阶段删除。",
                "fields": [
                    {
                        "field_code": "MAT_NO",
                        "standard_name": "材料号"
                    },
                    {
                        "field_code": "SIGN_CODE",
                        "standard_name": "钢级代码"
                    },
                    {
                        "field_code": "SIGN_LINE_NO",
                        "standard_name": "产线族号"
                    },
                    {
                        "field_code": "ST_NO",
                        "standard_name": "出钢记号"
                    },
                    {
                        "field_code": "PONO",
                        "standard_name": "炉号"
                    },
                    {
                        "field_code": "TEST_COUNT",
                        "standard_name": "试验次数"
                    },
                    {
                        "field_code": "PRODUCT_UNIT_NO",
                        "standard_name": "取样机组号"
                    },
                    {
                        "field_code": "THICK_ZN",
                        "standard_name": "试样厚度"
                    },
                    {
                        "field_code": "YP_JUDGE",
                        "standard_name": "判定屈服强度"
                    },
                    {
                        "field_code": "R_VALUE_0_N",
                        "standard_name": "0度r值"
                    },
                    {
                        "field_code": "R_VALUE_45_N",
                        "standard_name": "45度r值"
                    },
                    {
                        "field_code": "R_VALUE_90_N",
                        "standard_name": "90度r值"
                    },
                    {
                        "field_code": "R_VALUE_AVG_N",
                        "standard_name": "平均r值"
                    },
                    {
                        "field_code": "DELTA_R_N",
                        "standard_name": "各向异性Δr"
                    },
                    {
                        "field_code": "N_VALUE_0_N",
                        "standard_name": "0度n值"
                    },
                    {
                        "field_code": "N_VALUE_45_N",
                        "standard_name": "45度n值"
                    },
                    {
                        "field_code": "N_VALUE_90_N",
                        "standard_name": "90度n值"
                    },
                    {
                        "field_code": "N_VALUE_AVG_N",
                        "standard_name": "平均n值"
                    },
                    {
                        "field_code": "BOT_YS_N_2",
                        "standard_name": "下屈服强度2"
                    },
                    {
                        "field_code": "TOP_YS_N_2",
                        "standard_name": "上屈服强度2"
                    },
                    {
                        "field_code": "RP02_N_2",
                        "standard_name": "屈服强度2"
                    },
                    {
                        "field_code": "YS_EL_N_2",
                        "standard_name": "屈服延伸率2"
                    },
                    {
                        "field_code": "TS_N_2",
                        "standard_name": "抗拉强度2"
                    },
                    {
                        "field_code": "BREAK_EL_N_2",
                        "standard_name": "断裂延伸率2"
                    },
                    {
                        "field_code": "AGT_EL_N_2",
                        "standard_name": "均匀延伸率2"
                    },
                    {
                        "field_code": "BOT_YS_N_3",
                        "standard_name": "下屈服强度3"
                    },
                    {
                        "field_code": "TOP_YS_N_3",
                        "standard_name": "上屈服强度3"
                    },
                    {
                        "field_code": "RP02_N_3",
                        "standard_name": "屈服强度3"
                    },
                    {
                        "field_code": "YS_EL_N_3",
                        "standard_name": "屈服延伸率3"
                    },
                    {
                        "field_code": "TS_N_3",
                        "standard_name": "抗拉强度3"
                    },
                    {
                        "field_code": "BREAK_EL_N_3",
                        "standard_name": "断裂延伸率3"
                    },
                    {
                        "field_code": "AGT_EL_N_3",
                        "standard_name": "均匀延伸率3"
                    },
                    {
                        "field_code": "N_VALUE_0_N_2",
                        "standard_name": "0度n值2"
                    },
                    {
                        "field_code": "N_VALUE_45_N_2",
                        "standard_name": "45度n值2"
                    },
                    {
                        "field_code": "N_VALUE_90_N_2",
                        "standard_name": "90度n值2"
                    },
                    {
                        "field_code": "N_VALUE_AVG_N_2",
                        "standard_name": "平均n值2"
                    },
                    {
                        "field_code": "TOAST_HARDNESS",
                        "standard_name": "烘烤硬化值"
                    },
                    {
                        "field_code": "HARD_TEST_TYPE",
                        "standard_name": "硬度类型"
                    },
                    {
                        "field_code": "HARDNESS_D_N",
                        "standard_name": "硬度D"
                    },
                    {
                        "field_code": "HARDNESS_C_N",
                        "standard_name": "硬度C"
                    },
                    {
                        "field_code": "HARDNESS_W_N",
                        "standard_name": "硬度W"
                    },
                    {
                        "field_code": "HARDNESS_A_N",
                        "standard_name": "硬度A"
                    },
                    {
                        "field_code": "EXPAND_HOLE_D",
                        "standard_name": "扩孔率D"
                    },
                    {
                        "field_code": "EXPAND_HOLE_C",
                        "standard_name": "扩孔率C"
                    },
                    {
                        "field_code": "EXPAND_HOLE_W",
                        "standard_name": "扩孔率W"
                    },
                    {
                        "field_code": "EXPAND_HOLE_A",
                        "standard_name": "扩孔率A"
                    },
                    {
                        "field_code": "AGING_INDEX",
                        "standard_name": "时效指数"
                    },
                    {
                        "field_code": "GHOSTY",
                        "standard_name": "鬼带级别"
                    },
                    {
                        "field_code": "BEND_L",
                        "standard_name": "纵向弯曲"
                    },
                    {
                        "field_code": "BEND_C",
                        "standard_name": "横向弯曲"
                    },
                    {
                        "field_code": "SPHR_RATIO_C",
                        "standard_name": "横截面球化率"
                    },
                    {
                        "field_code": "SPHR_RATIO_GRADE_C",
                        "standard_name": "横截面球化率等级"
                    },
                    {
                        "field_code": "CARB_SIZE_C",
                        "standard_name": "横截面碳化物尺寸"
                    },
                    {
                        "field_code": "SPHR_RATIO_L",
                        "standard_name": "纵截面球化率"
                    },
                    {
                        "field_code": "SPHR_RATIO_GRADE_L",
                        "standard_name": "纵截面球化率等级"
                    },
                    {
                        "field_code": "CARB_SIZE_L",
                        "standard_name": "纵截面碳化物尺寸"
                    },
                    {
                        "field_code": "TOP_DECARB_THICK",
                        "standard_name": "上表面脱碳层厚度"
                    },
                    {
                        "field_code": "BOT_DECARB_THICK",
                        "standard_name": "下表面脱碳层厚度"
                    },
                    {
                        "field_code": "REC_REVISE_TIME",
                        "standard_name": "实绩上传时间"
                    },
                    {
                        "field_code": "REC_CREATE_TIME",
                        "standard_name": "实绩创建时间"
                    },
                    {
                        "field_code": "MAT_TRACK_NO",
                        "standard_name": "材料跟踪号"
                    },
                    {
                        "field_code": "SAMPLE_ORDER_NO",
                        "standard_name": "取样合同号"
                    },
                    {
                        "field_code": "SLAB_NO",
                        "standard_name": "板坯号"
                    },
                    {
                        "field_code": "SM_UNIT_NO",
                        "standard_name": "炼钢机组号"
                    },
                    {
                        "field_code": "SM_PRODTIME",
                        "standard_name": "炼钢生产时间"
                    },
                    {
                        "field_code": "HR_COIL_NO",
                        "standard_name": "热卷号"
                    },
                    {
                        "field_code": "HR_UNIT_NO",
                        "standard_name": "热轧机组号"
                    },
                    {
                        "field_code": "HR_PRODTIME",
                        "standard_name": "热轧生产时间"
                    },
                    {
                        "field_code": "HR_ORDER_NO",
                        "standard_name": "热轧合同号"
                    },
                    {
                        "field_code": "CDCM_COIL_NO",
                        "standard_name": "酸轧出口卷号"
                    },
                    {
                        "field_code": "CDCM_UNIT_NO",
                        "standard_name": "酸轧机组号"
                    },
                    {
                        "field_code": "CDCM_PRODTIME",
                        "standard_name": "酸轧生产时间"
                    },
                    {
                        "field_code": "ANN_IN_COIL_NO",
                        "standard_name": "退火入口卷号"
                    },
                    {
                        "field_code": "ANN_OUT_COIL_NO",
                        "standard_name": "退火出口卷号"
                    },
                    {
                        "field_code": "ANN_UNIT_NO",
                        "standard_name": "退火机组号"
                    },
                    {
                        "field_code": "ANN_PRODTIME",
                        "standard_name": "退火生产时间"
                    },
                    {
                        "field_code": "ETL_COIL_NO",
                        "standard_name": "电镀出口卷号"
                    },
                    {
                        "field_code": "ETL_UNIT_NO",
                        "standard_name": "电镀机组号"
                    },
                    {
                        "field_code": "ETL_PRODTIME",
                        "standard_name": "电镀生产时间"
                    },
                    {
                        "field_code": "SG_STD",
                        "standard_name": "标准号"
                    },
                    {
                        "field_code": "STD_VERSION",
                        "standard_name": "标准版本号"
                    },
                    {
                        "field_code": "SG_SIGN",
                        "standard_name": "牌号"
                    },
                    {
                        "field_code": "ORDER_THICK",
                        "standard_name": "订货厚度"
                    },
                    {
                        "field_code": "ORDER_WIDTH",
                        "standard_name": "订货宽度"
                    },
                    {
                        "field_code": "PROD_WIDTH",
                        "standard_name": "成品宽度"
                    },
                    {
                        "field_code": "ORDER_LEN_MIN",
                        "standard_name": "订货长度"
                    },
                    {
                        "field_code": "PROD_LEN",
                        "standard_name": "成品长度"
                    },
                    {
                        "field_code": "ORDER_WT",
                        "standard_name": "订货重量"
                    },
                    {
                        "field_code": "PSR",
                        "standard_name": "产品规范码"
                    },
                    {
                        "field_code": "MIC",
                        "standard_name": "冶金规范码"
                    },
                    {
                        "field_code": "FIN_CUST_CODE",
                        "standard_name": "最终用户代码"
                    },
                    {
                        "field_code": "FIN_USER_NAME",
                        "standard_name": "最终用户名称"
                    },
                    {
                        "field_code": "APN",
                        "standard_name": "最终用途码"
                    },
                    {
                        "field_code": "APN_DESC",
                        "standard_name": "最终用途描述"
                    },
                    {
                        "field_code": "SORT_GRADE_CODE_F",
                        "standard_name": "成品分选度"
                    },
                    {
                        "field_code": "DELIVERY_DATE_IN",
                        "standard_name": "厂内交货期"
                    },
                    {
                        "field_code": "YS_MIN",
                        "standard_name": "屈服强度下限"
                    },
                    {
                        "field_code": "YS_MAX",
                        "standard_name": "屈服强度上限"
                    },
                    {
                        "field_code": "YS_EL_MAX",
                        "standard_name": "屈服延伸率上限"
                    },
                    {
                        "field_code": "TS_MIN",
                        "standard_name": "抗拉强度下限"
                    },
                    {
                        "field_code": "TS_MAX",
                        "standard_name": "抗拉强度上限"
                    },
                    {
                        "field_code": "BREAK_EL_MIN",
                        "standard_name": "断裂延伸率下限"
                    },
                    {
                        "field_code": "BREAK_EL_MAX",
                        "standard_name": "断裂延伸率上限"
                    },
                    {
                        "field_code": "MEASURE_METHOD_R",
                        "standard_name": "r值测量方法"
                    },
                    {
                        "field_code": "R_VALUE_0_MIN",
                        "standard_name": "0度r值下限"
                    },
                    {
                        "field_code": "R_VALUE_0_MAX",
                        "standard_name": "0度r值上限"
                    },
                    {
                        "field_code": "R_VALUE_45_MIN",
                        "standard_name": "45度r值下限"
                    },
                    {
                        "field_code": "R_VALUE_45_MAX",
                        "standard_name": "45度r值上限"
                    },
                    {
                        "field_code": "R_VALUE_90_MIN",
                        "standard_name": "90度r值下限"
                    },
                    {
                        "field_code": "R_VALUE_90_MAX",
                        "standard_name": "90度r值上限"
                    },
                    {
                        "field_code": "R_VALUE_AVER_MIN",
                        "standard_name": "平均r值下限"
                    },
                    {
                        "field_code": "R_VALUE_AVER_MAX",
                        "standard_name": "平均r值上限"
                    },
                    {
                        "field_code": "DELTA_R_MAX",
                        "standard_name": "各向异性Δr上限"
                    },
                    {
                        "field_code": "MEASURE_RANGE_N",
                        "standard_name": "n值测量方法"
                    },
                    {
                        "field_code": "N_VALUE_0_MIN",
                        "standard_name": "0度n值下限"
                    },
                    {
                        "field_code": "N_VALUE_0_MAX",
                        "standard_name": "0度n值上限"
                    },
                    {
                        "field_code": "N_VALUE_45_MIN",
                        "standard_name": "45度n值下限"
                    },
                    {
                        "field_code": "N_VALUE_45_MAX",
                        "standard_name": "45度n值上限"
                    },
                    {
                        "field_code": "N_VALUE_90_MIN",
                        "standard_name": "90度n值下限"
                    },
                    {
                        "field_code": "N_VALUE_90_MAX",
                        "standard_name": "90度n值上限"
                    },
                    {
                        "field_code": "N_VALUE_AVER_MIN",
                        "standard_name": "平均n值下限"
                    },
                    {
                        "field_code": "N_VALUE_AVER_MAX",
                        "standard_name": "平均n值上限"
                    },
                    {
                        "field_code": "TOAST_HARDNESS_TEST_METHOD",
                        "standard_name": "烘烤硬化值测量方法"
                    },
                    {
                        "field_code": "HARDNESS_OVEN_MIN",
                        "standard_name": "烘烤硬化值下限"
                    },
                    {
                        "field_code": "HARDNESS_OVEN_MAX",
                        "standard_name": "烘烤硬化值上限"
                    },
                    {
                        "field_code": "AGING_AI_MAX",
                        "standard_name": "时效指数上限"
                    },
                    {
                        "field_code": "TEST_DIRECT_CODE_2",
                        "standard_name": "拉伸方向2"
                    },
                    {
                        "field_code": "YS_MIN2",
                        "standard_name": "屈服强度下限2"
                    },
                    {
                        "field_code": "YS_MAX2",
                        "standard_name": "屈服强度上限2"
                    },
                    {
                        "field_code": "YS_EL_MAX2",
                        "standard_name": "屈服延伸率上限2"
                    },
                    {
                        "field_code": "TS_MIN2",
                        "standard_name": "抗拉强度下限2"
                    },
                    {
                        "field_code": "TS_MAX2",
                        "standard_name": "抗拉强度上限2"
                    },
                    {
                        "field_code": "BREAK_EL_MIN2",
                        "standard_name": "断裂延伸率下限2"
                    },
                    {
                        "field_code": "BREAK_EL_MAX2",
                        "standard_name": "断裂延伸率上限2"
                    },
                    {
                        "field_code": "TEST_DIRECT_CODE_3",
                        "standard_name": "拉伸方向3"
                    },
                    {
                        "field_code": "YS_MIN3",
                        "standard_name": "屈服强度下限3"
                    },
                    {
                        "field_code": "YS_MAX3",
                        "standard_name": "屈服强度上限3"
                    },
                    {
                        "field_code": "YS_EL_MAX3",
                        "standard_name": "屈服延伸率上限3"
                    },
                    {
                        "field_code": "TS_MIN3",
                        "standard_name": "抗拉强度下限3"
                    },
                    {
                        "field_code": "TS_MAX3",
                        "standard_name": "抗拉强度上限3"
                    },
                    {
                        "field_code": "BREAK_EL_MIN3",
                        "standard_name": "断裂延伸率下限3"
                    },
                    {
                        "field_code": "BREAK_EL_MAX3",
                        "standard_name": "断裂延伸率上限3"
                    },
                    {
                        "field_code": "MEASURE_RANGE_N_2",
                        "standard_name": "n值测量方法2"
                    },
                    {
                        "field_code": "N_VALUE_0_MIN_2",
                        "standard_name": "0度n值下限2"
                    },
                    {
                        "field_code": "N_VALUE_0_MAX_2",
                        "standard_name": "0度n值上限2"
                    },
                    {
                        "field_code": "N_VALUE_45_MIN_2",
                        "standard_name": "45度n值下限2"
                    },
                    {
                        "field_code": "N_VALUE_45_MAX_2",
                        "standard_name": "45度n值上限2"
                    },
                    {
                        "field_code": "N_VALUE_90_MIN_2",
                        "standard_name": "90度n值下限2"
                    },
                    {
                        "field_code": "N_VALUE_90_MAX_2",
                        "standard_name": "90度n值上限2"
                    },
                    {
                        "field_code": "N_VALUE_AVER_MIN_2",
                        "standard_name": "平均n值下限2"
                    },
                    {
                        "field_code": "N_VALUE_AVER_MAX_2",
                        "standard_name": "平均n值上限2"
                    },
                    {
                        "field_code": "BEND_ANGLE",
                        "standard_name": "弯曲角度"
                    },
                    {
                        "field_code": "BEND_DIA",
                        "standard_name": "弯曲直径"
                    },
                    {
                        "field_code": "BEND_WIDTH",
                        "standard_name": "弯曲宽度"
                    },
                    {
                        "field_code": "BEND_RESULT",
                        "standard_name": "弯曲结果"
                    },
                    {
                        "field_code": "HARDNESS_MEASURE_METHOD",
                        "standard_name": "硬度测量方法"
                    },
                    {
                        "field_code": "HARDNESS_MIN",
                        "standard_name": "硬度下限"
                    },
                    {
                        "field_code": "HARDNESS_MAX",
                        "standard_name": "硬度上限"
                    },
                    {
                        "field_code": "EXPAND_HOLE_METHOD",
                        "standard_name": "扩孔率测量方法"
                    },
                    {
                        "field_code": "EXPAND_HOLE_MIN",
                        "standard_name": "扩孔率下限"
                    },
                    {
                        "field_code": "EXPAND_HOLE_MAX",
                        "standard_name": "扩孔率上限"
                    },
                    {
                        "field_code": "SPHR_RATIO_METHOD",
                        "standard_name": "球化率测量方法"
                    },
                    {
                        "field_code": "SPHR_RATIO_MIN_C",
                        "standard_name": "横截面球化率下限"
                    },
                    {
                        "field_code": "SPHR_RATIO_MIN_L",
                        "standard_name": "纵截面球化率下限"
                    },
                    {
                        "field_code": "SPHR_RATIO_GRADE_MIN_C",
                        "standard_name": "横截面球化率等级下限"
                    },
                    {
                        "field_code": "SPHR_RATIO_GRADE_MIN_L",
                        "standard_name": "纵截面球化率等级下限"
                    },
                    {
                        "field_code": "CARB_SIZE_MIN_C",
                        "standard_name": "横截面碳化物尺寸下限"
                    },
                    {
                        "field_code": "CARB_SIZE_MIN_L",
                        "standard_name": "纵截面碳化物尺寸下限"
                    },
                    {
                        "field_code": "TOP_DECARB_THICK_MAX",
                        "standard_name": "上表面脱碳层厚度上限"
                    },
                    {
                        "field_code": "BOT_DECARB_THICK_MAX",
                        "standard_name": "下表面脱碳层厚度上限"
                    },
                    {
                        "field_code": "TOP_DECARB_THICK_PROP_MAX",
                        "standard_name": "上表面脱碳层厚度比例上限"
                    },
                    {
                        "field_code": "BOT_DECARB_THICK_PROP_MAX",
                        "standard_name": "下表面脱碳层厚度比例上限"
                    },
                    {
                        "field_code": "GHOSTY_TEST_METHOD",
                        "standard_name": "鬼带测量方法"
                    },
                    {
                        "field_code": "GHOST_GRADE",
                        "standard_name": "鬼带级别上限"
                    },
                    {
                        "field_code": "CC_NO",
                        "standard_name": "连铸处理号"
                    },
                    {
                        "field_code": "CAST_NO",
                        "standard_name": "连连浇号"
                    },
                    {
                        "field_code": "TD_NO",
                        "standard_name": "中间包号"
                    },
                    {
                        "field_code": "PREV_HEAT_ST_NO",
                        "standard_name": "前炉出钢记号"
                    },
                    {
                        "field_code": "NEXT_HEAT_ST_NO",
                        "standard_name": "后炉出钢记号"
                    },
                    {
                        "field_code": "DESIGN_DISCH_TEMP_AIM",
                        "standard_name": "设计出炉温度目标"
                    },
                    {
                        "field_code": "DESIGN_DISCH_TEMP_MIN",
                        "standard_name": "设计出炉温度下限"
                    },
                    {
                        "field_code": "DESIGN_DISCH_TEMP_MAX",
                        "standard_name": "设计出炉温度上限"
                    },
                    {
                        "field_code": "DESIGN_FM_TEMP_AIM",
                        "standard_name": "设计终轧温度目标"
                    },
                    {
                        "field_code": "DESIGN_FM_TEMP_MIN",
                        "standard_name": "设计终轧温度下限"
                    },
                    {
                        "field_code": "DESIGN_FM_TEMP_MAX",
                        "standard_name": "设计终轧温度上限"
                    },
                    {
                        "field_code": "DESIGN_CT_TEMP_AIM",
                        "standard_name": "设计卷取温度目标"
                    },
                    {
                        "field_code": "DESIGN_CT_TEMP_MIN",
                        "standard_name": "设计卷取温度下限"
                    },
                    {
                        "field_code": "DESIGN_CT_TEMP_MAX",
                        "standard_name": "设计卷取温度上限"
                    },
                    {
                        "field_code": "DESIGN_ANNEAL_DIAGRAM_CODE",
                        "standard_name": "设计退火曲线"
                    },
                    {
                        "field_code": "DESIGN_TPM_RATE_AIM_OLD",
                        "standard_name": "设计平整率目标"
                    },
                    {
                        "field_code": "DESIGN_TPM_RATE_MIN_OLD",
                        "standard_name": "设计平整率下限"
                    },
                    {
                        "field_code": "DESIGN_TPM_RATE_MAX_OLD",
                        "standard_name": "设计平整率上限"
                    },
                    {
                        "field_code": "DESIGN_RTF_TEMP_AIM",
                        "standard_name": "设计加热段温度目标"
                    },
                    {
                        "field_code": "DESIGN_RTF_TEMP_MIN",
                        "standard_name": "设计加热段温度下限"
                    },
                    {
                        "field_code": "DESIGN_RTF_TEMP_MAX",
                        "standard_name": "设计加热段温度上限"
                    },
                    {
                        "field_code": "DESIGN_SF_TEMP_AIM",
                        "standard_name": "设计均热段温度目标"
                    },
                    {
                        "field_code": "DESIGN_SF_TEMP_MIN",
                        "standard_name": "设计均热段温度下限"
                    },
                    {
                        "field_code": "DESIGN_SF_TEMP_MAX",
                        "standard_name": "设计均热段温度上限"
                    },
                    {
                        "field_code": "DESIGN_SCS_TEMP_AIM",
                        "standard_name": "设计缓冷段温度目标"
                    },
                    {
                        "field_code": "DESIGN_SCS_TEMP_MIN",
                        "standard_name": "设计缓冷段温度下限"
                    },
                    {
                        "field_code": "DESIGN_SCS_TEMP_MAX",
                        "standard_name": "设计缓冷段温度上限"
                    },
                    {
                        "field_code": "DESIGN_RCS_TEMP_AIM",
                        "standard_name": "设计快冷段温度目标"
                    },
                    {
                        "field_code": "DESIGN_RCS_TEMP_MIN",
                        "standard_name": "设计快冷段温度下限"
                    },
                    {
                        "field_code": "DESIGN_RCS_TEMP_MAX",
                        "standard_name": "设计快冷段温度上限"
                    },
                    {
                        "field_code": "DESIGN_OAS_TEMP_AIM",
                        "standard_name": "设计过时效段温度目标"
                    },
                    {
                        "field_code": "DESIGN_OAS_TEMP_MIN",
                        "standard_name": "设计过时效段温度下限"
                    },
                    {
                        "field_code": "DESIGN_OAS_TEMP_MAX",
                        "standard_name": "设计过时效段温度上限"
                    },
                    {
                        "field_code": "HR_PLAN_NO",
                        "standard_name": "热轧计划号"
                    },
                    {
                        "field_code": "FUR_NO_HR",
                        "standard_name": "热轧加热炉号"
                    },
                    {
                        "field_code": "ANNEAL_PLAN_NO",
                        "standard_name": "退火计划号"
                    },
                    {
                        "field_code": "PREC_ST_NO",
                        "standard_name": "炉次预定出钢记号"
                    },
                    {
                        "field_code": "FIN_ST_NO",
                        "standard_name": "炉次最终出钢记号"
                    },
                    {
                        "field_code": "SP_EXIT_MAT_NO",
                        "standard_name": "试验材料号"
                    },
                    {
                        "field_code": "DESIGN_TPM_RATE_AIM",
                        "standard_name": "设计平整率目标"
                    },
                    {
                        "field_code": "DESIGN_TPM_RATE_MIN",
                        "standard_name": "设计平整率下限"
                    },
                    {
                        "field_code": "DESIGN_TPM_RATE_MAX",
                        "standard_name": "设计平整率上限"
                    },
                    {
                        "field_code": "HR_START_POS",
                        "standard_name": "热轧开始位置"
                    },
                    {
                        "field_code": "HR_END_POS",
                        "standard_name": "热轧结束位置"
                    },
                    {
                        "field_code": "CR_START_POS",
                        "standard_name": "冷轧开始位置"
                    },
                    {
                        "field_code": "CR_END_POS",
                        "standard_name": "冷轧结束位置"
                    },
                    {
                        "field_code": "SOCKET_NO_BAF",
                        "standard_name": "罩式炉炉台号"
                    },
                    {
                        "field_code": "YP_ONLINE",
                        "standard_name": "在线检测屈服强度"
                    },
                    {
                        "field_code": "TS_ONLINE",
                        "standard_name": "在线检测抗拉强度"
                    },
                    {
                        "field_code": "EL_ONLINE",
                        "standard_name": "在线检测延伸率"
                    },
                    {
                        "field_code": "YPEL_ONLINE",
                        "standard_name": "在线检测屈服延伸率"
                    },
                    {
                        "field_code": "BH_ONLINE",
                        "standard_name": "在线检测BH值"
                    },
                    {
                        "field_code": "R_ONLINE",
                        "standard_name": "在线检测r90"
                    },
                    {
                        "field_code": "N_ONLINE",
                        "standard_name": "在线检测n90"
                    },
                    {
                        "field_code": "IMPACT_ENEG_1",
                        "standard_name": "冲击功1"
                    },
                    {
                        "field_code": "IMPACT_ENEG_2",
                        "standard_name": "冲击功2"
                    },
                    {
                        "field_code": "IMPACT_ENEG_3",
                        "standard_name": "冲击功3"
                    },
                    {
                        "field_code": "IMPACT_ENEG_A",
                        "standard_name": "冲击功A"
                    },
                    {
                        "field_code": "IMPACT_AVER_MIN",
                        "standard_name": "冲击平均值最小值"
                    },
                    {
                        "field_code": "IMPACT_SINGLE_MIN",
                        "standard_name": "冲击单值最小值"
                    },
                    {
                        "field_code": "REANNEAL_COUNT",
                        "standard_name": "再退火次数"
                    },
                    {
                        "field_code": "PRODUCT_FLAG",
                        "standard_name": "成品标记"
                    },
                    {
                        "field_code": "DC_AN_HC10_300",
                        "standard_name": "退火直流矫顽力(300OE)"
                    },
                    {
                        "field_code": "DC_HC10_300",
                        "standard_name": "直流矫顽力(300OE)"
                    },
                    {
                        "field_code": "OAS_MID_TEMP_ANN_AVG",
                        "standard_name": "过时效中间段温度均值"
                    },
                    {
                        "field_code": "OAS_MID_TEMP_ANN_MAX",
                        "standard_name": "过时效中间段温度最大值"
                    },
                    {
                        "field_code": "OAS_MID_TEMP_ANN_MIN",
                        "standard_name": "过时效中间段温度最小值"
                    },
                    {
                        "field_code": "SAMP_OAS_MID_TEMP_ANN_AVG",
                        "standard_name": "取样点过时效中间段温度均值"
                    },
                    {
                        "field_code": "SAMP_OAS_MID_TEMP_ANN_MAX",
                        "standard_name": "取样点过时效中间段温度最大值"
                    },
                    {
                        "field_code": "SAMP_OAS_MID_TEMP_ANN_MIN",
                        "standard_name": "取样点过时效中间段温度最小值"
                    },
                    {
                        "field_code": "SAMP_OAS_MID_TEMP_ANN_RANGE",
                        "standard_name": "取样点过时效中间段温度极差"
                    },
                    {
                        "field_code": "RT05_N",
                        "standard_name": "屈服强度Rt05"
                    },
                    {
                        "field_code": "DC_BM_8",
                        "standard_name": "直流磁感B8"
                    },
                    {
                        "field_code": "DC_BM_10",
                        "standard_name": "直流磁感B10"
                    },
                    {
                        "field_code": "DC_BM_50",
                        "standard_name": "直流磁感B50"
                    },
                    {
                        "field_code": "DC_BM_100",
                        "standard_name": "直流磁感B100"
                    },
                    {
                        "field_code": "YP_PREDICT",
                        "standard_name": "预测屈服强度"
                    },
                    {
                        "field_code": "YP_JUDGE_PREDICT",
                        "standard_name": "预测判定屈服强度"
                    },
                    {
                        "field_code": "TS_PREDICT",
                        "standard_name": "预测抗拉强度"
                    },
                    {
                        "field_code": "EL_PREDICT",
                        "standard_name": "预测延伸率"
                    },
                    {
                        "field_code": "YPEL_PREDICT",
                        "standard_name": "预测屈服延伸率"
                    },
                    {
                        "field_code": "BH_PREDICT",
                        "standard_name": "预测BH值"
                    },
                    {
                        "field_code": "R90_PREDICT",
                        "standard_name": "预测r90"
                    },
                    {
                        "field_code": "N90_PREDICT",
                        "standard_name": "预测n90"
                    },
                    {
                        "field_code": "R0_PREDICT",
                        "standard_name": "预测r0"
                    },
                    {
                        "field_code": "N0_PREDICT",
                        "standard_name": "预测n0"
                    },
                    {
                        "field_code": "R45_PREDICT",
                        "standard_name": "预测r45"
                    },
                    {
                        "field_code": "N45_PREDICT",
                        "standard_name": "预测n45"
                    },
                    {
                        "field_code": "RBAR_PREDICT",
                        "standard_name": "预测r平均"
                    },
                    {
                        "field_code": "NBAR_PREDICT",
                        "standard_name": "预测n平均"
                    },
                    {
                        "field_code": "IMPACT_PREDICT",
                        "standard_name": "预测冲击功"
                    },
                    {
                        "field_code": "DESIGN_SOURCE_HR",
                        "standard_name": "热轧工艺设计来源"
                    },
                    {
                        "field_code": "BOND_RESULT_INFO_HR",
                        "standard_name": "热轧工艺动态调整结果"
                    },
                    {
                        "field_code": "DESIGN_SOURCE_ANN",
                        "standard_name": "退火工艺设计来源"
                    },
                    {
                        "field_code": "BOND_RESULT_INFO_ANN",
                        "standard_name": "退火工艺动态调整结果"
                    },
                    {
                        "field_code": "CONT_H2_SF",
                        "standard_name": "均热段氢含量"
                    },
                    {
                        "field_code": "CONT_H2_SCS",
                        "standard_name": "缓冷段氢含量"
                    },
                    {
                        "field_code": "CONT_H2_RCS",
                        "standard_name": "快冷段氢含量"
                    },
                    {
                        "field_code": "EARING_RATE",
                        "standard_name": "制耳率"
                    },
                    {
                        "field_code": "BENDING_D",
                        "standard_name": "回弹D"
                    },
                    {
                        "field_code": "BENDING_C",
                        "standard_name": "回弹C"
                    },
                    {
                        "field_code": "BENDING_W",
                        "standard_name": "回弹W"
                    },
                    {
                        "field_code": "BENDING_A",
                        "standard_name": "回弹A"
                    },
                    {
                        "field_code": "ITEM_INC",
                        "standard_name": "纯净度测量方法"
                    },
                    {
                        "field_code": "A_CORS_TY_ASTM",
                        "standard_name": "纯净度A粗"
                    },
                    {
                        "field_code": "A_GOOD_TY_ASTM",
                        "standard_name": "纯净度A细"
                    },
                    {
                        "field_code": "B_CORS_TY_ASTM",
                        "standard_name": "纯净度B粗"
                    },
                    {
                        "field_code": "B_GOOD_TY_ASTM",
                        "standard_name": "纯净度B细"
                    },
                    {
                        "field_code": "C_CORS_TY_ASTM",
                        "standard_name": "纯净度C粗"
                    },
                    {
                        "field_code": "C_GOOD_TY_ASTM",
                        "standard_name": "纯净度C细"
                    },
                    {
                        "field_code": "D_CORS_TY_ASTM",
                        "standard_name": "纯净度D粗"
                    },
                    {
                        "field_code": "D_GOOD_TY_ASTM",
                        "standard_name": "纯净度D细"
                    },
                    {
                        "field_code": "TOP_ROUGH_D",
                        "standard_name": "上表面粗糙度D"
                    },
                    {
                        "field_code": "TOP_ROUGH_C",
                        "standard_name": "上表面粗糙度C"
                    },
                    {
                        "field_code": "TOP_ROUGH_W",
                        "standard_name": "上表面粗糙度W"
                    },
                    {
                        "field_code": "TOP_ROUGH_A",
                        "standard_name": "上表面粗糙度A"
                    },
                    {
                        "field_code": "BOT_ROUGH_D",
                        "standard_name": "下表面粗糙度D"
                    },
                    {
                        "field_code": "BOT_ROUGH_C",
                        "standard_name": "下表面粗糙度C"
                    },
                    {
                        "field_code": "BOT_ROUGH_W",
                        "standard_name": "下表面粗糙度W"
                    },
                    {
                        "field_code": "BOT_ROUGH_A",
                        "standard_name": "下表面粗糙度A"
                    },
                    {
                        "field_code": "TOP_PPI_D",
                        "standard_name": "上表面峰值数D"
                    },
                    {
                        "field_code": "TOP_PPI_C",
                        "standard_name": "上表面峰值数C"
                    },
                    {
                        "field_code": "TOP_PPI_W",
                        "standard_name": "上表面峰值数W"
                    },
                    {
                        "field_code": "TOP_PPI_A",
                        "standard_name": "上表面峰值数A"
                    },
                    {
                        "field_code": "BOT_PPI_D",
                        "standard_name": "下表面峰值数D"
                    },
                    {
                        "field_code": "BOT_PPI_C",
                        "standard_name": "下表面峰值数C"
                    },
                    {
                        "field_code": "BOT_PPI_W",
                        "standard_name": "下表面峰值数W"
                    },
                    {
                        "field_code": "BOT_PPI_A",
                        "standard_name": "下表面峰值数A"
                    },
                    {
                        "field_code": "TOP_WCA_D",
                        "standard_name": "上表面波纹度D"
                    },
                    {
                        "field_code": "TOP_WCA_C",
                        "standard_name": "上表面波纹度C"
                    },
                    {
                        "field_code": "TOP_WCA_W",
                        "standard_name": "上表面波纹度W"
                    },
                    {
                        "field_code": "TOP_WCA_A",
                        "standard_name": "上表面波纹度A"
                    },
                    {
                        "field_code": "BOT_WCA_D",
                        "standard_name": "下表面波纹度D"
                    },
                    {
                        "field_code": "BOT_WCA_C",
                        "standard_name": "下表面波纹度C"
                    },
                    {
                        "field_code": "BOT_WCA_W",
                        "standard_name": "下表面波纹度W"
                    },
                    {
                        "field_code": "BOT_WCA_A",
                        "standard_name": "下表面波纹度A"
                    },
                    {
                        "field_code": "TOP_PLATE_WT_D",
                        "standard_name": "上表面镀层重量D"
                    },
                    {
                        "field_code": "TOP_PLATE_WT_C",
                        "standard_name": "上表面镀层重量C"
                    },
                    {
                        "field_code": "TOP_PLATE_WT_W",
                        "standard_name": "上表面镀层重量W"
                    },
                    {
                        "field_code": "TOP_PLATE_WT_A",
                        "standard_name": "上表面镀层重量A"
                    },
                    {
                        "field_code": "BOT_PLATE_WT_D",
                        "standard_name": "下表面镀层重量D"
                    },
                    {
                        "field_code": "BOT_PLATE_WT_C",
                        "standard_name": "下表面镀层重量C"
                    },
                    {
                        "field_code": "BOT_PLATE_WT_W",
                        "standard_name": "下表面镀层重量W"
                    },
                    {
                        "field_code": "BOT_PLATE_WT_A",
                        "standard_name": "下表面镀层重量A"
                    },
                    {
                        "field_code": "CHAR_ELM_NAME",
                        "standard_name": "特征元素名称"
                    },
                    {
                        "field_code": "TOP_CHAR_ELM_WT_D",
                        "standard_name": "上表面特征元素重量D"
                    },
                    {
                        "field_code": "TOP_CHAR_ELM_WT_C",
                        "standard_name": "上表面特征元素重量C"
                    },
                    {
                        "field_code": "TOP_CHAR_ELM_WT_W",
                        "standard_name": "上表面特征元素重量W"
                    },
                    {
                        "field_code": "TOP_CHAR_ELM_WT_A",
                        "standard_name": "上表面特征元素重量A"
                    },
                    {
                        "field_code": "BOT_CHAR_ELM_WT_D",
                        "standard_name": "下表面特征元素重量D"
                    },
                    {
                        "field_code": "BOT_CHAR_ELM_WT_C",
                        "standard_name": "下表面特征元素重量C"
                    },
                    {
                        "field_code": "BOT_CHAR_ELM_WT_W",
                        "standard_name": "下表面特征元素重量W"
                    },
                    {
                        "field_code": "BOT_CHAR_ELM_WT_A",
                        "standard_name": "下表面特征元素重量A"
                    },
                    {
                        "field_code": "TOP_P_FILM_WT_D",
                        "standard_name": "上表面磷化膜重量D"
                    },
                    {
                        "field_code": "TOP_P_FILM_WT_C",
                        "standard_name": "上表面磷化膜重量C"
                    },
                    {
                        "field_code": "TOP_P_FILM_WT_W",
                        "standard_name": "上表面磷化膜重量W"
                    },
                    {
                        "field_code": "TOP_P_FILM_WT_A",
                        "standard_name": "上表面磷化膜重量A"
                    },
                    {
                        "field_code": "BOT_P_FILM_WT_D",
                        "standard_name": "下表面磷化膜重量D"
                    },
                    {
                        "field_code": "BOT_P_FILM_WT_C",
                        "standard_name": "下表面磷化膜重量C"
                    },
                    {
                        "field_code": "BOT_P_FILM_WT_W",
                        "standard_name": "下表面磷化膜重量W"
                    },
                    {
                        "field_code": "BOT_P_FILM_WT_A",
                        "standard_name": "下表面磷化膜重量A"
                    },
                    {
                        "field_code": "TOP_FE_D",
                        "standard_name": "上表面铁含量D"
                    },
                    {
                        "field_code": "TOP_FE_C",
                        "standard_name": "上表面铁含量C"
                    },
                    {
                        "field_code": "TOP_FE_W",
                        "standard_name": "上表面铁含量W"
                    },
                    {
                        "field_code": "TOP_FE_A",
                        "standard_name": "上表面铁含量A"
                    },
                    {
                        "field_code": "BOT_FE_D",
                        "standard_name": "下表面铁含量D"
                    },
                    {
                        "field_code": "BOT_FE_C",
                        "standard_name": "下表面铁含量C"
                    },
                    {
                        "field_code": "BOT_FE_W",
                        "standard_name": "下表面铁含量W"
                    },
                    {
                        "field_code": "BOT_FE_A",
                        "standard_name": "下表面铁含量A"
                    },
                    {
                        "field_code": "TOP_POWDER_D",
                        "standard_name": "上表面粉化试验D"
                    },
                    {
                        "field_code": "TOP_POWDER_C",
                        "standard_name": "上表面粉化试验C"
                    },
                    {
                        "field_code": "TOP_POWDER_W",
                        "standard_name": "上表面粉化试验W"
                    },
                    {
                        "field_code": "TOP_POWDER_A",
                        "standard_name": "上表面粉化试验A"
                    },
                    {
                        "field_code": "BOT_POWDER_D",
                        "standard_name": "下表面粉化试验D"
                    },
                    {
                        "field_code": "BOT_POWDER_C",
                        "standard_name": "下表面粉化试验C"
                    },
                    {
                        "field_code": "BOT_POWDER_W",
                        "standard_name": "下表面粉化试验W"
                    },
                    {
                        "field_code": "BOT_POWDER_A",
                        "standard_name": "下表面粉化试验A"
                    },
                    {
                        "field_code": "TOP_POWDER_WIDTH_D",
                        "standard_name": "上表面粉化试验宽度D"
                    },
                    {
                        "field_code": "TOP_POWDER_WIDTH_C",
                        "standard_name": "上表面粉化试验宽度C"
                    },
                    {
                        "field_code": "TOP_POWDER_WIDTH_W",
                        "standard_name": "上表面粉化试验宽度W"
                    },
                    {
                        "field_code": "TOP_POWDER_WIDTH_A",
                        "standard_name": "上表面粉化试验宽度A"
                    },
                    {
                        "field_code": "BOT_POWDER_WIDTH_D",
                        "standard_name": "下表面粉化试验宽度D"
                    },
                    {
                        "field_code": "BOT_POWDER_WIDTH_C",
                        "standard_name": "下表面粉化试验宽度C"
                    },
                    {
                        "field_code": "BOT_POWDER_WIDTH_W",
                        "standard_name": "下表面粉化试验宽度W"
                    },
                    {
                        "field_code": "BOT_POWDER_WIDTH_A",
                        "standard_name": "下表面粉化试验宽度A"
                    },
                    {
                        "field_code": "TOP_UF_FILM_D",
                        "standard_name": "上表面UF膜D"
                    },
                    {
                        "field_code": "TOP_UF_FILM_C",
                        "standard_name": "上表面UF膜C"
                    },
                    {
                        "field_code": "TOP_UF_FILM_W",
                        "standard_name": "上表面UF膜W"
                    },
                    {
                        "field_code": "TOP_UF_FILM_A",
                        "standard_name": "上表面UF膜A"
                    },
                    {
                        "field_code": "BOT_UF_FILM_D",
                        "standard_name": "下表面UF膜D"
                    },
                    {
                        "field_code": "BOT_UF_FILM_C",
                        "standard_name": "下表面UF膜C"
                    },
                    {
                        "field_code": "BOT_UF_FILM_W",
                        "standard_name": "下表面UF膜W"
                    },
                    {
                        "field_code": "BOT_UF_FILM_A",
                        "standard_name": "下表面UF膜A"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_D",
                        "standard_name": "上表面粗糙度RZ_D"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_C",
                        "standard_name": "上表面粗糙度RZ_C"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_W",
                        "standard_name": "上表面粗糙度RZ_W"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_A",
                        "standard_name": "上表面粗糙度RZ_A"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_D",
                        "standard_name": "下表面粗糙度RZ_D"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_C",
                        "standard_name": "下表面粗糙度RZ_C"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_W",
                        "standard_name": "下表面粗糙度RZ_W"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_A",
                        "standard_name": "下表面粗糙度RZ_A"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_MAX_D",
                        "standard_name": "上表面最大粗糙度RZ_D"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_MAX_C",
                        "standard_name": "上表面最大粗糙度RZ_C"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_MAX_W",
                        "standard_name": "上表面最大粗糙度RZ_W"
                    },
                    {
                        "field_code": "TOP_ROUGH_RZ_MAX_A",
                        "standard_name": "上表面最大粗糙度RZ_A"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_MAX_D",
                        "standard_name": "下表面最大粗糙度RZ_D"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_MAX_C",
                        "standard_name": "下表面最大粗糙度RZ_C"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_MAX_W",
                        "standard_name": "下表面最大粗糙度RZ_W"
                    },
                    {
                        "field_code": "BOT_ROUGH_RZ_MAX_A",
                        "standard_name": "下表面最大粗糙度RZ_A"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSK_D",
                        "standard_name": "上表面粗糙度RSK_D"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSK_C",
                        "standard_name": "上表面粗糙度RSK_C"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSK_W",
                        "standard_name": "上表面粗糙度RSK_W"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSK_A",
                        "standard_name": "上表面粗糙度RSK_A"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSK_D",
                        "standard_name": "下表面粗糙度RSK_D"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSK_C",
                        "standard_name": "下表面粗糙度RSK_C"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSK_W",
                        "standard_name": "下表面粗糙度RSK_W"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSK_A",
                        "standard_name": "下表面粗糙度RSK_A"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSM_D",
                        "standard_name": "上表面粗糙度RSM_D"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSM_C",
                        "standard_name": "上表面粗糙度RSM_C"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSM_W",
                        "standard_name": "上表面粗糙度RSM_W"
                    },
                    {
                        "field_code": "TOP_ROUGH_RSM_A",
                        "standard_name": "上表面粗糙度RSM_A"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSM_D",
                        "standard_name": "下表面粗糙度RSM_D"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSM_C",
                        "standard_name": "下表面粗糙度RSM_C"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSM_W",
                        "standard_name": "下表面粗糙度RSM_W"
                    },
                    {
                        "field_code": "BOT_ROUGH_RSM_A",
                        "standard_name": "下表面粗糙度RSM_A"
                    },
                    {
                        "field_code": "ROUGH_MEASURE_METHOD",
                        "standard_name": "粗糙度测量方法"
                    },
                    {
                        "field_code": "TOP_ROUGH_MIN",
                        "standard_name": "上表面Ra下限"
                    },
                    {
                        "field_code": "TOP_ROUGH_MAX",
                        "standard_name": "上表面Ra上限"
                    },
                    {
                        "field_code": "BOT_ROUGH_MIN",
                        "standard_name": "下表面Ra下限"
                    },
                    {
                        "field_code": "BOT_ROUGH_MAX",
                        "standard_name": "下表面Ra上限"
                    },
                    {
                        "field_code": "PPI_MIN",
                        "standard_name": "PPI下限"
                    },
                    {
                        "field_code": "PPI_MAX",
                        "standard_name": "PPI上限"
                    },
                    {
                        "field_code": "WCA_TEST_METHOD",
                        "standard_name": "WCA测量方法"
                    },
                    {
                        "field_code": "WCA_MIN",
                        "standard_name": "WCA下限"
                    },
                    {
                        "field_code": "WCA_MAX",
                        "standard_name": "WCA上限"
                    },
                    {
                        "field_code": "ROUGH_RZ_MIN",
                        "standard_name": "粗糙度RZ下限"
                    },
                    {
                        "field_code": "ROUGH_RZ_MAX",
                        "standard_name": "粗糙度RZ上限"
                    },
                    {
                        "field_code": "ROUGH_RZMAX_MIN",
                        "standard_name": "最大粗糙度RZ下限"
                    },
                    {
                        "field_code": "ROUGH_RZMAX_MAX",
                        "standard_name": "最大粗糙度RZ上限"
                    },
                    {
                        "field_code": "ROUGH_RSK_MIN",
                        "standard_name": "粗糙度RSK下限"
                    },
                    {
                        "field_code": "ROUGH_RSK_MAX",
                        "standard_name": "粗糙度RSK上限"
                    },
                    {
                        "field_code": "ROUGH_RSM_MIN",
                        "standard_name": "粗糙度RSM下限"
                    },
                    {
                        "field_code": "YS_TYPE_CODE",
                        "standard_name": "屈服强度类型"
                    },
                    {
                        "field_code": "ROUGH_RSM_MAX",
                        "standard_name": "粗糙度RSM上限"
                    }
                ]

            }
        },
    ]
    _initialize_sample_kb("professional_knowledge_kb", professional_knowledge_samples)

    #
    # 1. 数据预处理知识库
    preprocessing_samples = [
        {
            "text_for_embedding": "成分数据中，元素V含量ELM_V的缺失值通常用0填充，因为该元素含量极低。",
            "metadata": {"feature": "ELM_V", "strategy": "fill_with_zero", "reason": "该元素在工艺方面含量非常低"}
        }
    ]
    _initialize_sample_kb("data_preprocessing_kb", preprocessing_samples)

    # 2. 特征工程知识库
    feature_engineering_samples = [
        {
            "text_for_embedding": "碳当量(CE)是一种用于评估钢材焊接性的指标，通常由C, Mn, Cr, Mo, V, Ni, Cu等元素计算得到。它将多种合金元素对钢材淬硬性的影响折算成碳当量的形式。",
            "metadata": {
                "feature_name": "Carbon Equivalent (CE)",
                "formula_template": "{C} + {Mn}/6 + ({Cr}+{Mo}+{V})/5 + ({Ni}+{Cu})/15",
                "elements": ["C", "Mn", "Cr", "Mo", "V", "Ni", "Cu"],
                "mapping_hints": {
                    "C": ["C", "ELM_C"],
                    "Mn": ["Mn", "ELM_MN"],
                    "Cr": ["Cr", "ELM_CR"],
                    "Mo": ["Mo", "ELM_MO"],
                    "V": ["V", "ELM_V"],
                    "Ni": ["Ni", "ELM_NI"],
                    "Cu": ["Cu", "ELM_CU"]
                },
                "new_feature_name": "CE"
            }
        },
        {
            "text_for_embedding": "过剩钛(EX.TI)指钢中未与氮结合形成TiN的钛。这部分钛在钢中以固溶态或其他化合物形式存在，影响钢材性能。计算时需注意原子量比。",
            "metadata": {
                "feature_name": "Excess Titanium (EX.TI)",
                "formula_template": "{Ti} - 3.4 * {N}",
                "elements": ["Ti", "N"],
                "mapping_hints": {
                    "Ti": ["Ti", "ELM_TI"],
                    "N": ["N", "ELM_N"]
                },
                "new_feature_name": "EX_TI"
            }
        },
        {
            "text_for_embedding": "铝氮比(Al/N)是控制钢中氮化物析出的重要参数，影响晶粒尺寸和性能。",
            "metadata": {
                "feature_name": "Aluminum-Nitrogen Ratio",
                "formula_template": "{Al} / {N}",
                "elements": ["Al", "N"],
                "mapping_hints": {
                    "Al": ["Al", "ELM_AL"],
                    "N": ["N", "ELM_N"]
                },
                "new_feature_name": "AL_N_RATIO"
            }
        }
    ]
    _initialize_sample_kb("feature_engineering_kb", feature_engineering_samples)
    #
    # 3. 模型选择知识库
    model_selection_samples = [
        {
            "text_for_embedding": "对于小样本且特征维度不高的钢材性能预测问题，随机森林通常表现稳健且不易过拟合。",
            "metadata": {"recommended_model": "RandomForest", "sample_size": "middle", "feature_dim": "low_to_medium",
                         "pros": "robust, less_overfitting"}
        },
        {
            "text_for_embedding": "当数据量较大且特征间存在复杂非线性关系时，XGBoost配合适当的超参数调优往往能获得最佳的预测精度。",
            "metadata": {"recommended_model": "XGBoost", "sample_size": "large",
                         "feature_relations": "complex_nonlinear",
                         "notes": "requires_careful_tuning"}
        }
    ]
    _initialize_sample_kb("model_selection_kb", model_selection_samples)

    # # --- 测试知识库搜索 ---
    # # 测试数据预处理知识库
    # dp_kb = KnowledgeBaseService("data_preprocessing_kb")
    # search_query_dp = "我要构建一个针对小样本Q235B钢种数据的抗拉强度性能预报模型，在建模过程中添加碳当量特征。"
    # results_dp = dp_kb.search(search_query_dp, k=1)
    # print(f"\n搜索 '{search_query_dp}' 在 data_preprocessing_kb 中的结果:")
    # for res in results_dp:
    #     print(res)
    #
    # # 测试特征工程知识库
    # fe_kb = KnowledgeBaseService("feature_engineering_kb")
    # search_query_fe = "我要构建一个针对小样本Q235B钢种数据的抗拉强度性能预报模型，在建模过程中添加碳当量特征。"
    # results_fe = fe_kb.search(search_query_fe, k=1)
    # print(f"\n搜索 '{search_query_fe}' 在 feature_engineering_kb 中的结果:")
    # for res in results_fe:
    #     print(res)
    #
    # 测试业务数据知识库
    fe_kb = KnowledgeBaseService("professional_knowledge_kb")
    # search_query_fe = "目标性能字段列表。"
    # search_query_fe = "数据预处理时使用的目标性能指标字段列表。"
    search_query_fe = "经验意义上不适用作训练字段的特征列"
    results_fe = fe_kb.search(search_query_fe, k=1)
    print(f"\n搜索 '{search_query_fe}' 在 professional_knowledge_kb 中的结果:")
    for res in results_fe:
        print(res)
