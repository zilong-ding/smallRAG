from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import Elasticsearch, NotFoundError, ConnectionError as ESConnectionError
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import TransportError
import traceback
import logging

# 可选：配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Pydantic 模型增强：自动序列化 datetime
# -------------------------

class DocumentMeta(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    doc_id: str
    workspace_id: str
    user_username: str
    title: str
    file_name: str
    abstract: str
    full_content: str
    embedding_status: str = "pending"
    file_size: int
    file_hash: str
    created_at: datetime
    updated_at: datetime

class ChunkInfo(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    chunk_id: str
    doc_id: str
    workspace_id: str
    user_username: str
    chunk_content: str
    embedding_vector: List[float] = Field(..., min_length=1536, max_length=1536)
    chunk_order: int
    page_number: Optional[int] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime

class QAHistory(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    qa_id: str
    user_username: str
    question: str
    answer: str
    qa_vector: List[float] = Field(..., min_length=1536, max_length=1536)
    qa_concat_vector: List[float] = Field(..., min_length=1536, max_length=1536)
    workspace_id: str
    created_at: datetime

class ImageInfo(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    image_id: str
    user_username: str
    workspace_id: str
    image_path: str
    caption: str
    tags: List[str]
    embedding_vector: List[float] = Field(..., min_length=512, max_length=512)
    metadata: dict = Field(default_factory=dict)
    file_size: int
    width: int
    height: int
    format: str
    created_at: datetime

# -------------------------
# 工具函数：安全执行 ES 操作
# -------------------------

def safe_es_call(func):
    """装饰器：统一捕获 ES 异常"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ESConnectionError as e:
            logger.error(f"❌ Elasticsearch 连接失败: {e}")
            raise
        except NotFoundError:
            return None
        except TransportError as e:
            logger.error(f"❌ ES 请求错误: {e.info if hasattr(e, 'info') else e}")
            raise
        except Exception as e:
            logger.error(f"❌ 未知错误: {e}")
            traceback.print_exc()
            raise
    return wrapper

# -------------------------
# Elasticsearch 数据库管理类（增强版）
# -------------------------

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
    # 索引管理
    # -------------------------

    def init_indices(self, overwrite: bool = False) -> bool:
        try:
            if not self.es.ping():
                logger.error("❌ 无法连接 Elasticsearch")
                return False

            mappings = self._get_mappings()

            for name, index_name in self._indices.items():
                exists = self.es.indices.exists(index=index_name)
                if exists:
                    if overwrite:
                        self.es.indices.delete(index=index_name)
                        self.es.indices.create(index=index_name, body=mappings[name])
                        logger.info(f"🔄 已覆盖重建索引: {index_name}")
                    else:
                        logger.info(f"ℹ️ 索引已存在: {index_name}")
                else:
                    self.es.indices.create(index=index_name, body=mappings[name])
                    logger.info(f"✅ 已创建索引: {index_name}")

            logger.info("🎉 所有索引初始化完成！")
            return True

        except Exception as e:
            logger.error(f"❌ 初始化索引失败: {e}")
            traceback.print_exc()
            return False

    def _get_mappings(self) -> Dict[str, Dict]:
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
                        "tags": {"type": "keyword"},  # ✅ keyword 类型
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

    def indices_exist(self) -> Dict[str, bool]:
        """检查各索引是否存在"""
        return {name: self.es.indices.exists(index=index_name)
                for name, index_name in self._indices.items()}

    # -------------------------
    # 通用 CRUD 方法（带模型验证）
    # -------------------------

    def _validate_and_serialize(self, model_cls: type[BaseModel], data: Union[BaseModel, Dict]) -> Dict:
        if isinstance(data, dict):
            instance = model_cls(**data)
        elif isinstance(data, model_cls):
            instance = data
        else:
            raise TypeError(f"Expected dict or {model_cls.__name__}, got {type(data)}")
        return instance.model_dump()

    @safe_es_call
    def create_document(self, doc_id: str, data: Union[DocumentMeta, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(DocumentMeta, data)
        return self.es.index(index=self._indices["document"], id=doc_id, body=body)

    @safe_es_call
    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self.es.get(index=self._indices["document"], id=doc_id)["_source"]

    @safe_es_call
    def update_document(self, doc_id: str, update_data: Union[DocumentMeta, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(DocumentMeta, update_data)
        return self.es.update(index=self._indices["document"], id=doc_id, body={"doc": body})

    @safe_es_call
    def delete_document(self, doc_id: str) -> Dict:
        return self.es.delete(index=self._indices["document"], id=doc_id)

    # -------------------------
    # 批量操作（示例：chunks）
    # -------------------------
    @safe_es_call
    def create_chunk(self, chunk_id: str, data: Union[ChunkInfo, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(ChunkInfo, data)
        return self.es.index(index=self._indices["chunk"], id=chunk_id, body=body)

    # -------------------------
    # 3. 分块（Chunk）操作 —— 补全
    # -------------------------

    @safe_es_call
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        return self.es.get(index=self._indices["chunk"], id=chunk_id)["_source"]

    @safe_es_call
    def update_chunk(self, chunk_id: str, update_data: Union[ChunkInfo, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(ChunkInfo, update_data)
        return self.es.update(index=self._indices["chunk"], id=chunk_id, body={"doc": body})

    @safe_es_call
    def delete_chunk(self, chunk_id: str) -> Dict:
        return self.es.delete(index=self._indices["chunk"], id=chunk_id)

    # -------------------------
    # 4. 问答（QA）操作 —— 补全
    # -------------------------

    @safe_es_call
    def create_qa(self, qa_id: str, data: Union[QAHistory, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(QAHistory, data)
        return self.es.index(index=self._indices["qa"], id=qa_id, body=body)

    @safe_es_call
    def get_qa(self, qa_id: str) -> Optional[Dict]:
        return self.es.get(index=self._indices["qa"], id=qa_id)["_source"]

    @safe_es_call
    def update_qa(self, qa_id: str, update_data: Union[QAHistory, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(QAHistory, update_data)
        return self.es.update(index=self._indices["qa"], id=qa_id, body={"doc": body})

    @safe_es_call
    def delete_qa(self, qa_id: str) -> Dict:
        return self.es.delete(index=self._indices["qa"], id=qa_id)

    # -------------------------
    # 5. 图片（Image）操作 —— 补全
    # -------------------------

    @safe_es_call
    def create_image(self, image_id: str, data: Union[ImageInfo, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(ImageInfo, data)
        return self.es.index(index=self._indices["image"], id=image_id, body=body)

    @safe_es_call
    def get_image(self, image_id: str) -> Optional[Dict]:
        return self.es.get(index=self._indices["image"], id=image_id)["_source"]

    @safe_es_call
    def update_image(self, image_id: str, update_data: Union[ImageInfo, Dict[str, Any]]) -> Dict:
        body = self._validate_and_serialize(ImageInfo, update_data)
        return self.es.update(index=self._indices["image"], id=image_id, body={"doc": body})

    @safe_es_call
    def delete_image(self, image_id: str) -> Dict:
        return self.es.delete(index=self._indices["image"], id=image_id)

    # -------------------------
    # 6. 搜索接口 —— 补全缺失方法
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

    def hybrid_search_chunks(
            self,
            text_query: str,
            vector_query: List[float],
            top_k_text: int = 5,
            top_k_vector: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        混合检索：同时执行全文检索和向量检索，返回两类结果。

        Args:
            text_query: 用于全文检索的关键词或句子
            vector_query: 1536维的查询向量
            top_k_text: 全文检索返回数量
            top_k_vector: 向量检索返回数量

        Returns:
            {
                "text_hits": [...],
                "vector_hits": [...]
            }
        """
        # 1. 全文检索（BM25）
        text_res = self.es.search(
            index=self._indices["chunk"],
            body={
                "query": {
                    "match": {
                        "chunk_content": text_query
                    }
                },
                "size": top_k_text
            }
        )
        text_hits = [hit["_source"] for hit in text_res["hits"]["hits"]]

        # 2. 向量检索（KNN）
        vector_res = self.es.search(
            index=self._indices["chunk"],
            body={
                "knn": {
                    "field": "embedding_vector",
                    "query_vector": vector_query,
                    "k": top_k_vector,
                    "num_candidates": max(10, top_k_vector * 2)
                }
            }
        )
        vector_hits = [hit["_source"] for hit in vector_res["hits"]["hits"]]

        return {
            "text_hits": text_hits,
            "vector_hits": vector_hits
        }

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

    @safe_es_call
    def bulk_create_chunks(self, chunks: List[Union[ChunkInfo, Dict]]) -> Dict:
        actions = []
        for chunk in chunks:
            body = self._validate_and_serialize(ChunkInfo, chunk)
            actions.append({
                "_op_type": "index",
                "_index": self._indices["chunk"],
                "_id": body["chunk_id"],
                "_source": body
            })
        return bulk(self.es, actions)

    # 其他 bulk 方法可类似实现（qa, image 等）

    # -------------------------
    # 图片搜索：修复 tags 查询逻辑
    # -------------------------

    def search_images_by_tags_and_caption(
        self,
        caption_keyword: str,
        tags: List[str],
        size: int = 10
    ) -> List[Dict]:
        if not tags:
            query = {"query": {"match": {"caption": caption_keyword}}}
        else:
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

    # -------------------------
    # 其他方法保持不变（略），但建议也加上 @safe_es_call
    # -------------------------

    def refresh_all(self):
        for index in self._indices.values():
            self.es.indices.refresh(index=index)