from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text,Integer
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime
from werkzeug.security import generate_password_hash,check_password_hash
"""
该数据库存储用户和用户下创建的对话以及上传的文档之间的关系
    用户注册之后会创建用户
    该用户可以上传文件到 Workspace，可以有多个 Workspace，每个 Workspace 管理多个上传文档的状态
    每个文档需要创建时间，上一次更新时间，标题
    该用户可以创建不同的对话，每个对话都需要创建时间，上一次更新时间，标题，具体聊天内容（JSON）

数据库设计说明：
- User: 用户
- Workspace: 用户创建的工作区（对应 ./uploads/ 下的子文件夹）
- Document: 上传到 Workspace 中的具体文档
- Conversation: 用户创建的对话

关系：
- User 1 → N Workspace
- Workspace 1 → N Document
- User 1 → N Conversation
"""


engine = create_engine('sqlite:///data.db', echo=True)
Base = declarative_base()


class User(Base):
    __tablename__ = 'users'

    username = Column(String, primary_key=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 一对多关系
    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    workspaces = relationship(
        "Workspace",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    # ===== 新增方法 =====
    def verify_password(self, password: str) -> bool:
        """验证明文密码是否与哈希匹配"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User(username='{self.username}', created_at='{self.created_at}')>"


class Workspace(Base):
    __tablename__ = 'workspaces'

    id = Column(Integer, primary_key=True, autoincrement=True)  # 自增 ID（更安全）
    name = Column(String, nullable=False)  # 工作区名称（如 "project_docs"）
    description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 外键：属于哪个用户
    user_username = Column(String, ForeignKey('users.username', ondelete="CASCADE"), nullable=False)
    user = relationship("User", back_populates="workspaces")

    # 一对多：包含多个文档
    documents = relationship(
        "Document",
        back_populates="workspace",
        cascade="all, delete-orphan"
    )

    # 确保同一用户下工作区名称唯一
    __table_args__ = (
        # SQLite 支持复合唯一约束
        # UNIQUE (name, user_username)
    )

    def __repr__(self):
        return f"<Workspace(id={self.id}, name='{self.name}', user='{self.user_username}')>"


class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)          # 文档标题（可从文件名提取）
    filename = Column(String, nullable=False)       # 原始文件名（含扩展名）
    file_path = Column(String, nullable=False)      # 相对路径，如 "uploads/project_docs/report.pdf"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    embedding_status = Column(String, default="pending")  # "pending", "completed", "failed"
    file_size = Column(Integer, default=0)          # 文件大小（字节）

    # 外键：属于哪个 Workspace
    workspace_id = Column(Integer, ForeignKey('workspaces.id', ondelete="CASCADE"), nullable=False)
    workspace = relationship("Workspace", back_populates="documents")

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}', workspace_id={self.workspace_id})>"


class Conversation(Base):
    __tablename__ = 'conversations'

    id = Column(String, primary_key=True)  # e.g., "conv_20240610_123456"
    title = Column(String, default="新对话")
    messages = Column(Text)  # JSON 字符串：[{"role": "user", "content": "..."}, ...]
    rag_enabled = Column(String, default="false")  # 保留为字符串（或改为 Boolean）
    workspace_name = Column(String, nullable=True)  # 关联的 workspace.name（非外键，因对话可能跨会话）

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 外键：属于哪个用户
    user_username = Column(String, ForeignKey('users.username', ondelete="CASCADE"), nullable=False)
    user = relationship("User", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation(id='{self.id}', user='{self.user_username}')>"

# 创建所有表
Base.metadata.create_all(engine)
dataSession = sessionmaker(bind=engine)
def get_db():
    db = dataSession()
    try:
        yield db
    finally:
        db.close()