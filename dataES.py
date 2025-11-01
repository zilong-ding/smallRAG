from elasticsearch import Elasticsearch
import traceback
"""
RAG 系统 Elasticsearch 索引初始化脚本

该系统包含以下索引，每个索引的设计目标与字段说明如下：

1️⃣ document_meta（文档元信息索引）
   - 用途：存储用户上传的原始文档信息，用于全文检索和管理。
   - 核心字段：
       * doc_id (keyword)：文档唯一 ID
       * workspace_id (keyword)：所属工作区 ID
       * user_username (keyword)：上传者用户名
       * title / file_name / abstract / full_content (text)：文档内容，可全文检索
       * embedding_status (keyword)：向量是否已生成（pending/completed/failed）
       * file_size / file_hash (integer/keyword)：文档文件大小及哈希
       * created_at / updated_at (date)：文档创建和更新时间
   - 检索方式：全文检索（倒排索引）

2️⃣ chunk_info（文档分块索引）
   - 用途：将长文档拆分为小块用于语义检索（向量检索），适合 RAG 检索召回阶段。
   - 核心字段：
       * chunk_id / doc_id / workspace_id / user_username (keyword)：分块 ID、所属文档、工作区及用户信息
       * chunk_content (text)：分块文本，可做关键词检索
       * embedding_vector (dense_vector)：向量表示，支持语义相似度检索
       * chunk_order / page_number (integer)：记录分块顺序与原文页码
       * metadata (object)：可扩展元信息
       * created_at (date)：创建时间
   - 检索方式：语义向量检索为主，可兼顾关键词搜索

3️⃣ qa_history（历史问答索引）
   - 用途：存储用户与大模型的问答记录，用于快速检索过去问答、知识复用与 RAG 检索。
   - 核心字段：
       * qa_id (keyword)：问答唯一 ID
       * user_username / workspace_id (keyword)：所属用户与工作区
       * question / answer (text)：问答文本，可全文检索
       * qa_vector / qa_concat_vector (dense_vector)：问答语义向量，用于语义相似度检索
       * created_at (date)：创建时间
   - 检索方式：支持关键词全文检索 + 向量语义检索，可混合排序

4️⃣ image_info（图片索引）
   - 用途：存储用户上传的图片信息，用于图像语义检索、文本搜索和标签过滤。
   - 核心字段：
       * image_id / user_username / workspace_id (keyword)：图片 ID、所属用户与工作区
       * image_path (keyword)：图片存储路径或 URL
       * caption (text)：图片文本描述，可全文检索
       * tags (keyword)：可选标签，方便过滤和聚合
       * embedding_vector (dense_vector)：图片向量（CLIP 等），支持语义检索
       * metadata (object)：可扩展元信息，如拍摄设备、颜色、EXIF 等
       * file_size / width / height / format (integer/keyword)：文件大小及图片属性
       * created_at (date)：上传时间
   - 检索方式：全文检索 + 向量语义检索 + 标签过滤

未来扩展：
- 可以增加视频数据库索引，设计方式与 image_info 类似，支持视频向量检索和元数据管理。
"""

# 连接 Elasticsearch（默认无用户名密码）
es = Elasticsearch("http://localhost:9200")

# =========================
# 1️⃣ 文档索引（全文检索）
# =========================
document_meta_mapping = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "workspace_id": {"type": "keyword"},
            "user_username": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_max_word"
            },
            "file_name": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_max_word"
            },
            "abstract": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_max_word"
            },
            "full_content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_max_word"
            },
            "embedding_status": {"type": "keyword"},  # pending/completed/failed
            "file_size": {"type": "integer"},
            "file_hash": {"type": "keyword"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"}
        }
    }
}

# =========================
# 2️⃣ 文档分块索引（语义检索）
# =========================
chunk_info_mapping = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "workspace_id": {"type": "keyword"},
            "user_username": {"type": "keyword"},
            "chunk_content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_max_word"
            },
            "embedding_vector": {
                "type": "dense_vector",
                "dims": 1536,  # 根据向量模型维度设置
                "index": True,
                "similarity": "cosine"
            },
            "chunk_order": {"type": "integer"},
            "page_number": {"type": "integer"},
            "metadata": {"type": "object"},
            "created_at": {"type": "date"}
        }
    }
}

# =========================
# 3️⃣ 历史问答索引（全文 + 语义）
# =========================
qa_history_mapping = {
    "mappings": {
        "properties": {
            "qa_id": {"type": "keyword"},
            "user_username": {"type": "keyword"},

            # 全文检索字段
            "question": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "answer": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },

            # 语义检索向量
            "qa_vector": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },
            "qa_concat_vector": {  # 问答拼接向量，用于更高质量语义检索
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine"
            },

            # 元信息
            "workspace_id": {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
}

# =========================
# 4️⃣ 图片索引（全文 + 标签 + 语义）
# =========================
image_info_mapping = {
    "mappings": {
        "properties": {
            "image_id": {"type": "keyword"},
            "user_username": {"type": "keyword"},
            "workspace_id": {"type": "keyword"},
            "image_path": {"type": "keyword"},  # 文件路径或 URL

            # 文本检索字段
            "caption": {
                "type": "text",
                "analyzer": "ik_max_word",  # 建索引时细分
                "search_analyzer": "ik_smart"  # 查询时粗分
            },
            "tags": {"type": "keyword"},  # 可选标签，方便过滤

            # 图像语义向量（CLIP embedding）
            "embedding_vector": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,
                "similarity": "cosine"
            },

            # 元信息
            "metadata": {"type": "object"},
            "file_size": {"type": "integer"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "format": {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
}


# =========================
# 初始化所有索引函数
# =========================
def init_es_indices(overwrite=False):
    """
    初始化 Elasticsearch 索引
    :param overwrite: 如果索引已存在，是否删除重建
    :return: True/False 表示初始化是否成功
    """
    try:
        if not es.ping():
            print("❌ 无法连接 Elasticsearch")
            return False

        # 所有索引及其映射
        indices = {
            "smallrag_document_meta": document_meta_mapping,
            "smallrag_chunk_info": chunk_info_mapping,
            "smallrag_qa_history": qa_history_mapping,
            "smallrag_image_info": image_info_mapping
        }

        for name, mapping in indices.items():
            if es.indices.exists(index=name):
                if overwrite:
                    es.indices.delete(index=name)
                    es.indices.create(index=name, body=mapping)
                    print(f"🔄 已覆盖重建索引: {name}")
                else:
                    print(f"ℹ️ 索引已存在: {name}")
            else:
                es.indices.create(index=name, body=mapping)
                print(f"✅ 已创建索引: {name}")

        print("🎉 所有索引已初始化完成！")
        # 验证 mapping 是否正确加载
        actual_mapping = es.indices.get_mapping(index="smallrag_image_info")
        tags_type = actual_mapping["smallrag_image_info"]["mappings"]["properties"]["tags"]["type"]
        assert tags_type == "keyword", f"tags 字段类型错误: {tags_type}"
        return True

    except Exception as e:
        print(traceback.format_exc())
        print("❌ 创建索引时出错:", str(e))
        return False


# =========================
# 调用示例
# =========================
if __name__ == "__main__":
    init_es_indices(overwrite=True)  # 👈 关键！