# smallrag_db.py
from datetime import datetime
from typing import Optional, Dict, Any, List
from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.exceptions import ConnectionError
import traceback

# =========================
# Elasticsearch 数据库管理类
# =========================

class SmallRAGDB:
    def __init__(self, es_url: str = "http://localhost:9200"):
        self.es = Elasticsearch(es_url)
        self._indices = {
            "document": "smallrag_document_meta",
            "chunk": "smallrag_chunk_info",
            "qa": "smallrag_qa_history",
            "image": "smallrag_image_info"
        }

    # -------------------------
    # 1. 索引初始化
    # -------------------------

    def init_indices(self, overwrite: bool = False) -> bool:
        """初始化所有索引"""
        try:
            if not self.es.ping():
                print("❌ 无法连接 Elasticsearch")
                return False

            mappings = self._get_mappings()

            for name, index_name in self._indices.items():
                if self.es.indices.exists(index=index_name):
                    if overwrite:
                        self.es.indices.delete(index=index_name)
                        self.es.indices.create(index=index_name, body=mappings[name])
                        print(f"🔄 已覆盖重建索引: {index_name}")
                    else:
                        print(f"ℹ️ 索引已存在: {index_name}")
                else:
                    self.es.indices.create(index=index_name, body=mappings[name])
                    print(f"✅ 已创建索引: {index_name}")

            print("🎉 所有索引初始化完成！")
            return True

        except Exception as e:
            print("❌ 初始化索引失败:", str(e))
            traceback.print_exc()
            return False

    def _get_mappings(self) -> Dict[str, Dict]:
        """返回所有索引的 mapping 定义（与你原始定义一致）"""
        return {
            "document": {
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "workspace_id": {"type": "keyword"},
                        "user_username": {"type": "keyword"},
                        "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
                        "file_name": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
                        "abstract": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
                        "full_content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
                        "embedding_status": {"type": "keyword"},
                        "file_size": {"type": "integer"},
                        "file_hash": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
            },
            "chunk": {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "workspace_id": {"type": "keyword"},
                        "user_username": {"type": "keyword"},
                        "chunk_content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
                        "embedding_vector": {"type": "dense_vector", "dims": 1536, "index": True, "similarity": "cosine"},
                        "chunk_order": {"type": "integer"},
                        "page_number": {"type": "integer"},
                        "metadata": {"type": "object"},
                        "created_at": {"type": "date"}
                    }
                }
            },
            "qa": {
                "mappings": {
                    "properties": {
                        "qa_id": {"type": "keyword"},
                        "user_username": {"type": "keyword"},
                        "question": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                        "answer": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                        "qa_vector": {"type": "dense_vector", "dims": 1536, "index": True, "similarity": "cosine"},
                        "qa_concat_vector": {"type": "dense_vector", "dims": 1536, "index": True, "similarity": "cosine"},
                        "workspace_id": {"type": "keyword"},
                        "created_at": {"type": "date"}
                    }
                }
            },
            "image": {
                "mappings": {
                    "properties": {
                        "image_id": {"type": "keyword"},
                        "user_username": {"type": "keyword"},
                        "workspace_id": {"type": "keyword"},
                        "image_path": {"type": "keyword"},
                        "caption": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                        "tags": {"type": "keyword"},  # ✅ 确保是 keyword
                        "embedding_vector": {"type": "dense_vector", "dims": 512, "index": True, "similarity": "cosine"},
                        "metadata": {"type": "object"},
                        "file_size": {"type": "integer"},
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                        "format": {"type": "keyword"},
                        "created_at": {"type": "date"}
                    }
                }
            }
        }

    # -------------------------
    # 2. 文档（Document）操作
    # -------------------------

    def create_document(self, doc_id: str, data: Dict[str, Any]) -> Dict:
        return self.es.index(index=self._indices["document"], id=doc_id, body=data)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        try:
            return self.es.get(index=self._indices["document"], id=doc_id)["_source"]
        except NotFoundError:
            return None

    def update_document(self, doc_id: str, update_data: Dict[str, Any]) -> Dict:
        return self.es.update(index=self._indices["document"], id=doc_id, body={"doc": update_data})

    def delete_document(self, doc_id: str) -> Dict:
        return self.es.delete(index=self._indices["document"], id=doc_id)

    # -------------------------
    # 3. 分块（Chunk）操作
    # -------------------------

    def create_chunk(self, chunk_id: str, data: Dict[str, Any]) -> Dict:
        return self.es.index(index=self._indices["chunk"], id=chunk_id, body=data)

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        try:
            return self.es.get(index=self._indices["chunk"], id=chunk_id)["_source"]
        except NotFoundError:
            return None

    def update_chunk(self, chunk_id: str, update_data: Dict[str, Any]) -> Dict:
        return self.es.update(index=self._indices["chunk"], id=chunk_id, body={"doc": update_data})

    def delete_chunk(self, chunk_id: str) -> Dict:
        return self.es.delete(index=self._indices["chunk"], id=chunk_id)

    # -------------------------
    # 4. 问答（QA）操作
    # -------------------------

    def create_qa(self, qa_id: str, data: Dict[str, Any]) -> Dict:
        return self.es.index(index=self._indices["qa"], id=qa_id, body=data)

    def get_qa(self, qa_id: str) -> Optional[Dict]:
        try:
            return self.es.get(index=self._indices["qa"], id=qa_id)["_source"]
        except NotFoundError:
            return None

    def update_qa(self, qa_id: str, update_data: Dict[str, Any]) -> Dict:
        return self.es.update(index=self._indices["qa"], id=qa_id, body={"doc": update_data})

    def delete_qa(self, qa_id: str) -> Dict:
        return self.es.delete(index=self._indices["qa"], id=qa_id)

    # -------------------------
    # 5. 图片（Image）操作
    # -------------------------

    def create_image(self, image_id: str, data: Dict[str, Any]) -> Dict:
        return self.es.index(index=self._indices["image"], id=image_id, body=data)

    def get_image(self, image_id: str) -> Optional[Dict]:
        try:
            return self.es.get(index=self._indices["image"], id=image_id)["_source"]
        except NotFoundError:
            return None

    def update_image(self, image_id: str, update_data: Dict[str, Any]) -> Dict:
        return self.es.update(index=self._indices["image"], id=image_id, body={"doc": update_data})

    def delete_image(self, image_id: str) -> Dict:
        return self.es.delete(index=self._indices["image"], id=image_id)

    # -------------------------
    # 6. 搜索接口
    # -------------------------

    def search_documents(self, query: Dict, size: int = 10) -> List[Dict]:
        res = self.es.search(index=self._indices["document"], body=query, size=size)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def search_chunks_by_vector(self, vector: List[float], k: int = 5) -> List[Dict]:
        res = self.es.search(
            index=self._indices["chunk"],
            body={
                "knn": {
                    "field": "embedding_vector",
                    "query_vector": vector,
                    "k": k,
                    "num_candidates": max(10, k * 2)
                }
            }
        )
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def search_images_by_tags_and_caption(
        self,
        caption_keyword: str,
        tags: List[str],
        size: int = 10
    ) -> List[Dict]:
        query = {
            "query": {
                "bool": {
                    "must": [{"match": {"caption": caption_keyword}}],
                    "filter": [{"terms": {"tags": tags}}]
                }
            }
        }
        res = self.es.search(index=self._indices["image"], body=query, size=size)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def search_images_by_vector(self, vector: List[float], k: int = 5) -> List[Dict]:
        res = self.es.search(
            index=self._indices["image"],
            body={
                "knn": {
                    "field": "embedding_vector",
                    "query_vector": vector,
                    "k": k,
                    "num_candidates": max(10, k * 2)
                }
            }
        )
        return [hit["_source"] for hit in res["hits"]["hits"]]

    # -------------------------
    # 7. 工具方法
    # -------------------------

    def refresh_all(self):
        """强制刷新所有索引（测试用）"""
        for index in self._indices.values():
            self.es.indices.refresh(index=index)

if __name__ == "__main__":
    db = SmallRAGDB()

    # 初始化（测试时建议 overwrite=True）
    db.init_indices(overwrite=True)

    # 创建文档
    doc_data = {
        "doc_id": "doc_1",
        "workspace_id": "ws_1",
        "user_username": "alice",
        "title": "测试报告",
        "full_content": "内容...",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "embedding_status": "pending",
        "file_size": 1000,
        "file_hash": "xxx"
    }
    db.create_document("doc_1", doc_data)

    # 搜索图片
    images = db.search_images_by_tags_and_caption(caption_keyword="医疗", tags=["AI"])
    print("匹配图片数:", len(images))

    # 向量检索分块（假设你有真实向量）
    dummy_vec = [0.1] * 1536
    chunks = db.search_chunks_by_vector(dummy_vec, k=3)