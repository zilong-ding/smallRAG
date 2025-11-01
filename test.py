import os
import time
from datetime import datetime, timezone
from typing import List
from testES import (  # 假设你的模型和 DB 类在 smallrag_db.py
    SmallRAGDB,
    DocumentMeta,
    ChunkInfo,
    QAHistory,
    ImageInfo
)

# 如果你的类和模型在同一文件，可改为：
# from smallrag_db import SmallRAGDB, DocumentMeta, ChunkInfo, QAHistory, ImageInfo

def now_utc():
    return datetime.now(timezone.utc)

def test_smallrag_db():
    es_url = os.getenv("ES_URL", "http://localhost:9200")
    db = SmallRAGDB(es_url=es_url)

    print("🧪 开始测试 SmallRAGDB...")

    # 1. 初始化索引（覆盖）
    print("\n1️⃣ 初始化索引...")
    assert db.init_indices(overwrite=True), "索引初始化失败"
    time.sleep(1)  # 等待 ES 内部刷新

    # 2. 测试 Document
    print("\n2️⃣ 测试 Document CRUD...")
    doc_id = "doc_001"
    doc_data = DocumentMeta(
        doc_id=doc_id,
        workspace_id="ws_123",
        user_username="alice",
        title="测试文档",
        file_name="test.pdf",
        abstract="这是一个测试摘要",
        full_content="这是完整的文档内容，用于测试。",
        file_size=1024,
        file_hash="abc123",
        created_at=now_utc(),
        updated_at=now_utc()
    )
    db.create_document(doc_id, doc_data)
    retrieved = db.get_document(doc_id)
    assert retrieved is not None
    assert retrieved["title"] == "测试文档"
    print("✅ Document 创建 & 查询成功")

    # 3. 测试 Chunk（含向量）
    print("\n3️⃣ 测试 Chunk CRUD...")
    chunk_id = "chunk_001"
    query_vector = [0.9] + [0.1] * 1535  # 唯一高维
    chunk_data = ChunkInfo(
        chunk_id=chunk_id,
        doc_id=doc_id,
        workspace_id="ws_123",
        user_username="alice",
        chunk_content="这是第一个分块内容。",
        embedding_vector=query_vector,
        chunk_order=1,
        page_number=1,
        metadata={"section": "introduction"},
        created_at=now_utc()
    )
    db.create_chunk(chunk_id, chunk_data)

    # 4. 批量创建 Chunks（使用低相似度向量）
    print("\n4️⃣ 测试批量创建 Chunks...")
    chunks = []
    for i in range(2, 5):
        # 第一个维度故意不同，降低相似度
        vec = [0.2] + [0.1] * 1535
        chunks.append(ChunkInfo(
            chunk_id=f"chunk_{i:03d}",
            doc_id=doc_id,
            workspace_id="ws_123",
            user_username="alice",
            chunk_content=f"这是第 {i} 个分块。",
            embedding_vector=vec,
            chunk_order=i,
            created_at=now_utc()
        ))
    db.bulk_create_chunks(chunks)
    db.refresh_all()
    time.sleep(2)  # 👈 等待 ES 构建 ANN 索引

    # 5. 测试 QA
    print("\n5️⃣ 测试 QA CRUD...")
    qa_id = "qa_001"
    fake_qa_vec = [0.5] * 1536
    qa_data = QAHistory(
        qa_id=qa_id,
        user_username="alice",
        question="这是什么系统？",
        answer="这是一个小型 RAG 系统。",
        qa_vector=fake_qa_vec,
        qa_concat_vector=fake_qa_vec,
        workspace_id="ws_123",
        created_at=now_utc()
    )
    db.create_qa(qa_id, qa_data)
    qa_retrieved = db.get_qa(qa_id)
    assert qa_retrieved is not None
    assert "RAG" in qa_retrieved["answer"]
    print("✅ QA 创建 & 查询成功")

    # 6. 测试 Image（含 tags 和向量）
    print("\n6️⃣ 测试 Image CRUD...")
    image_id = "img_001"
    fake_img_vec = [0.3] * 512
    img_data = ImageInfo(
        image_id=image_id,
        user_username="alice",
        workspace_id="ws_123",
        image_path="/images/test.jpg",
        caption="一只在草地上奔跑的狗",
        tags=["dog", "outdoor", "animal"],
        embedding_vector=fake_img_vec,
        file_size=20480,
        width=800,
        height=600,
        format="JPEG",
        created_at=now_utc()
    )
    db.create_image(image_id, img_data)
    img_retrieved = db.get_image(image_id)
    assert img_retrieved is not None
    assert "dog" in img_retrieved["tags"]
    db.refresh_all()
    print("✅ Image 创建 & 查询成功")

    # 7. 测试向量搜索
    print("\n7️⃣ 测试 Chunk 向量搜索...")
    results = db.search_chunks_by_vector(query_vector, k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "chunk_001", f"Expected chunk_001, got {results[0]['chunk_id']}"
    print("✅ Chunk 向量搜索成功")

    # 8. 测试图片混合搜索（caption + tags）
    print("\n8️⃣ 测试图片混合搜索...")
    img_results = db.search_images_by_tags_and_caption(
        caption_keyword="狗",
        tags=["dog", "animal"],
        size=5
    )
    assert len(img_results) >= 1
    assert img_results[0]["image_id"] == "img_001"
    print("✅ 图片混合搜索成功")

    # 9. 测试不存在的 ID
    print("\n9️⃣ 测试查询不存在的文档...")
    none_doc = db.get_document("non_existent_doc")
    assert none_doc is None
    print("✅ 不存在文档返回 None，符合预期")

    # 10. 验证 tags 字段为 keyword（可通过 mapping 检查）
    print("\n🔟 验证 image.tags 为 keyword 类型...")
    mapping = db.es.indices.get_mapping(index=db._indices["image"])
    tags_type = mapping[db._indices["image"]]["mappings"]["properties"]["tags"]["type"]
    assert tags_type == "keyword", f"tags 类型应为 keyword，实际为 {tags_type}"
    print("✅ tags 字段类型正确")

    print("\n🎉 所有测试通过！")


if __name__ == "__main__":
    test_smallrag_db()