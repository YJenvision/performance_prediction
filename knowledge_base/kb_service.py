import json

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
                    })
                else:
                    # 目前经常返回，待检查什么情况。
                    print(f"警告: 搜索返回了无效的索引 {idx}。")
            print("搜索成功。")
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
    # --- 知识库后续通过独立脚本维护 ---

    # 0. 业务数据知识库
    with open(r"professional_knowledge_kb.json", "r", encoding="utf-8") as f:
        professional_knowledge_samples = json.load(f)
    _initialize_sample_kb("professional_knowledge_kb", professional_knowledge_samples)

    #
    # 1. 数据预处理知识库
    with open(r"data_preprocessing_kb.json", "r", encoding="utf-8") as f:
        preprocessing_samples = json.load(f)
    _initialize_sample_kb("data_preprocessing_kb", preprocessing_samples)

    # 2. 特征工程知识库
    with open(r"feature_engineering_kb.json", "r", encoding="utf-8") as f:
        feature_engineering_samples = json.load(f)
    _initialize_sample_kb("feature_engineering_kb", feature_engineering_samples)

    # 3. 模型选择知识库
    with open(r"model_selection_kb.json", "r", encoding="utf-8") as f:
        model_selection_samples = json.load(f)
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
