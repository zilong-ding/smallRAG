from pydantic import BaseModel, Field, validator, field_validator,Json
import re
from datetime import datetime

# === Pydantic Models ===
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=6)
    confirm: str

    @field_validator('confirm')
    @classmethod
    def passwords_match(cls, v, info):
        # info.data 是已验证字段的字典
        if 'password' in info.data and info.data['password'] != v:
            raise ValueError('两次密码不一致')
        return v

    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError('用户名只能包含字母、数字和下划线')
        return v

class LoginRequest(BaseModel):
    username: str
    password: str
    class Config:
        orm_mode = True

class FileItem(BaseModel):
    name: str          # 文件名，如 "report.pdf"
    size: int          # 文件大小（字节）
    modified: datetime # 最后修改时间

class WorkspaceResponse(BaseModel):
    name: str
    description: str
    created_at: datetime
    user_username: str

    class Config:
        orm_mode = True

class WorkspacesResponse(BaseModel):
    names: list[str]
    class Config:
        orm_mode = True

class chatRequest(BaseModel):
    question: str
    user_name: str
    conversation_name: str  # 修正拼写：conversion → conversation
    workspace_name: str     # 保持单数，与字段一致
    conversation_id: int

class chatResponse(BaseModel):
    answer: str
    conversation_name: str
    # conversation_id: str

class ConversationsResponse(BaseModel):
    title: str
    updated_at: datetime
    conversation_id: int

# 消息模型
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime  # 这里简化为字符串格式

class UserResponse(BaseModel):
    username: str
    created_at: datetime

    class Config:
        orm_mode = True

class RegisterResponse(BaseModel):
    success: bool
    message: str
    username: str

    class Config:
        from_attributes = True  # Pydantic V2