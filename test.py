# test_smallrag_db.py
import unittest
import sys
import os
from datetime import datetime

# 将项目路径加入 sys.path（根据你的实际结构调整）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from testES import SmallRAGDB  # 假设类定义在 smallrag_db.py

class TestSmallRAGDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """测试前初始化数据库（重建索引）"""
        cls.db = SmallRAGDB(es_url="http://localhost:9200")
        success = cls.db.init_indices(overwrite=True)
        if not success:
            raise RuntimeError("Elasticsearch 初始化失败，请确保 ES 正在运行！")
        cls.dummy_text_vector = [0.1] * 1536
        cls.dummy_image_vector = [0.2] * 512

    def setUp(self):
        """每个测试前清空数据（通过删除关键文档）"""
        # 删除可能存在的测试数据（避免干扰）
        for doc_id in ["test_doc", "test_chunk", "test_qa", "test_img"]:
            try:
                self.db.delete_document(doc_id)
            except:
                pass
            try:
                self.db.delete_chunk(doc_id)
            except:
                pass
            try:
                self.db.delete_qa(doc_id)
            except:
                pass
            try:
                self.db.delete_image(doc_id)
            except:
                pass
        self.db.refresh_all()

    def test_1_create_and_get_document(self):
        """测试创建和获取文档"""
        doc_data = {
            "doc_id": "test_doc",
            "workspace_id": "ws_test",
            "user_username": "test_user",
            "title": "人工智能测试文档",
            "file_name": "test.pdf",
            "abstract": "测试摘要",
            "full_content": "完整内容用于测试。",
            "embedding_status": "pending",
            "file_size": 1024,
            "file_hash": "hash123",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        self.db.create_document("test_doc", doc_data)
        self.db.refresh_all()

        retrieved = self.db.get_document("test_doc")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["title"], "人工智能测试文档")

    def test_2_update_document(self):
        """测试更新文档"""
        self.db.create_document("test_doc", {
            "doc_id": "test_doc",
            "workspace_id": "ws_test",
            "title": "旧标题",
            "embedding_status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        })
        self.db.refresh_all()

        self.db.update_document("test_doc", {"embedding_status": "completed"})
        self.db.refresh_all()

        updated = self.db.get_document("test_doc")
        self.assertEqual(updated["embedding_status"], "completed")

    def test_3_create_and_get_chunk(self):
        """测试创建和获取分块"""
        chunk_data = {
            "chunk_id": "test_chunk",
            "doc_id": "test_doc",
            "workspace_id": "ws_test",
            "user_username": "test_user",
            "chunk_content": "大模型在医疗领域有广泛应用。",
            "embedding_vector": self.dummy_text_vector,
            "chunk_order": 1,
            "page_number": 10,
            "metadata": {"section": "introduction"},
            "created_at": datetime.utcnow().isoformat()
        }
        self.db.create_chunk("test_chunk", chunk_data)
        self.db.refresh_all()

        retrieved = self.db.get_chunk("test_chunk")
        self.assertIsNotNone(retrieved)
        self.assertIn("医疗", retrieved["chunk_content"])

    def test_4_create_and_get_qa(self):
        """测试创建和获取问答"""
        qa_data = {
            "qa_id": "test_qa",
            "user_username": "test_user",
            "question": "大模型能用于医疗吗？",
            "answer": "可以，例如辅助诊断。",
            "qa_vector": self.dummy_text_vector,
            "qa_concat_vector": self.dummy_text_vector,
            "workspace_id": "ws_test",
            "created_at": datetime.utcnow().isoformat()
        }
        self.db.create_qa("test_qa", qa_data)
        self.db.refresh_all()

        retrieved = self.db.get_qa("test_qa")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["question"], "大模型能用于医疗吗？")

    def test_5_create_and_get_image(self):
        """测试创建和获取图片"""
        image_data = {
            "image_id": "test_img",
            "user_username": "test_user",
            "workspace_id": "ws_test",
            "image_path": "/test/ai.jpg",
            "caption": "AI医疗影像示意图",
            "tags": ["医疗", "AI", "影像"],
            "embedding_vector": self.dummy_image_vector,
            "metadata": {"source": "test"},
            "file_size": 2048,
            "width": 800,
            "height": 600,
            "format": "jpg",
            "created_at": datetime.utcnow().isoformat()
        }
        self.db.create_image("test_img", image_data)
        self.db.refresh_all()

        retrieved = self.db.get_image("test_img")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["format"], "jpg")
        self.assertIn("AI", retrieved["tags"])

    def test_6_search_documents_fulltext(self):
        """测试文档全文检索"""
        self.db.create_document("test_doc", {
            "doc_id": "test_doc",
            "workspace_id": "ws_test",
            "title": "人工智能年度报告",
            "full_content": "本报告讨论AI在各行业的应用。",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "embedding_status": "completed",
            "file_size": 1000,
            "file_hash": "xxx"
        })
        self.db.refresh_all()

        results = self.db.search_documents({
            "query": {"match": {"title": "人工智能"}}
        })
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("人工智能", results[0]["title"])

    def test_7_search_chunks_by_vector(self):
        """测试分块向量检索"""
        self.db.create_chunk("test_chunk", {
            "chunk_id": "test_chunk",
            "doc_id": "test_doc",
            "workspace_id": "ws_test",
            "chunk_content": "向量检索测试内容",
            "embedding_vector": self.dummy_text_vector,
            "chunk_order": 1,
            "created_at": datetime.utcnow().isoformat()
        })
        self.db.refresh_all()

        results = self.db.search_chunks_by_vector(self.dummy_text_vector, k=1)
        self.assertGreaterEqual(len(results), 1)

    def test_8_search_images_by_tags_and_caption(self):
        """测试图片标签+全文组合检索"""
        self.db.create_image("test_img", {
            "image_id": "test_img",
            "user_username": "test_user",
            "workspace_id": "ws_test",
            "image_path": "/test/med_ai.jpg",
            "caption": "AI辅助医疗诊断系统界面",
            "tags": ["医疗", "AI", "UI"],
            "embedding_vector": self.dummy_image_vector,
            "file_size": 1024,
            "width": 1920,
            "height": 1080,
            "format": "png",
            "created_at": datetime.utcnow().isoformat()
        })
        self.db.refresh_all()

        results = self.db.search_images_by_tags_and_caption(
            caption_keyword="医疗",
            tags=["AI"]
        )
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("AI", results[0]["tags"])
        self.assertIn("医疗", results[0]["caption"])

    def test_9_search_images_by_vector(self):
        """测试图片向量检索"""
        self.db.create_image("test_img", {
            "image_id": "test_img",
            "user_username": "test_user",
            "workspace_id": "ws_test",
            "image_path": "/test/img.jpg",
            "caption": "测试图片",
            "tags": ["test"],
            "embedding_vector": self.dummy_image_vector,
            "file_size": 512,
            "width": 100,
            "height": 100,
            "format": "jpg",
            "created_at": datetime.utcnow().isoformat()
        })
        self.db.refresh_all()

        results = self.db.search_images_by_vector(self.dummy_image_vector, k=1)
        self.assertGreaterEqual(len(results), 1)

    def test_10_delete_operations(self):
        """测试删除操作"""
        # 创建
        self.db.create_qa("test_qa", {
            "qa_id": "test_qa",
            "question": "删除测试？",
            "answer": "是的。",
            "workspace_id": "ws_test",
            "created_at": datetime.utcnow().isoformat()
        })
        self.db.refresh_all()

        # 验证存在
        self.assertIsNotNone(self.db.get_qa("test_qa"))

        # 删除
        self.db.delete_qa("test_qa")
        self.db.refresh_all()

        # 验证不存在
        self.assertIsNone(self.db.get_qa("test_qa"))


if __name__ == "__main__":
    unittest.main(verbosity=2)