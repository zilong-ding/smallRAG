import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import re

class User:
    def __init__(self,data_root):
        self.username = ""
        self.password_hash = ""
        self.created_at = ""
        self.data_root = data_root
        self.user_dir = os.path.join(self.data_root, "users", self.username)

    def get_conversions(self):
        return os.path.join(self.user_dir, "conversations")

    def get_uploads(self):
        return os.path.join(self.user_dir, "uploads")

    def to_dict(self):
        return {
            "password_hash": self.password_hash,
            "created_at": self.created_at
        }


class AuthManager:
    def __init__(self, data_root):
        self.users = {}
        self.data_root = data_root
        self.file_path = os.path.join(self.data_root, "auth", "users.json")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def login(self, username, password):
        """
        用户登录
        返回: 错误信息 或 登录成功消息
        """
        # print("用户登录:", username)
        with open(self.file_path, "r") as f:
            users = json.load(f)
        if username in users and check_password_hash(users[username]["password_hash"], password):
            # 创建用户目录（首次登录）
            user_dir = os.path.join(self.data_root, "users", username)
            os.makedirs(os.path.join(user_dir, "uploads"), exist_ok=True)
            os.makedirs(os.path.join(user_dir, "conversations"), exist_ok=True)
            return False, True, username, "登录成功！"
        else:
            return True,  False,  "", "用户名或密码错误！"

    def register_user(self,username: str, password: str) -> str:
        """
        注册新用户
        返回: 成功消息 或 错误信息
        """
        # 1. 输入校验
        if not username or not password:
            return "用户名和密码不能为空！"
        if len(username) < 3:
            return "用户名至少3个字符！"
        if len(password) < 6:
            return "密码至少6位！"
        if not re.match("^[a-zA-Z0-9_]+$", username):
            return "用户名只能包含字母、数字和下划线！"
        # 2. 加载现有用户
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                users = json.load(f)
        else:
            users = {}
        # 3. 用户名唯一性检查
        if username in users:
            return "用户名已存在，请换一个！"
        # 4. 密码哈希（使用 pbkdf2:sha256，默认 150000 轮迭代）
        password_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        # 5. 保存新用户
        users[username] = {
            "password_hash": password_hash,
            "created_at": datetime.now().isoformat() + "Z"
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
        # 6. 自动创建用户数据目录（避免首次登录才创建）
        user_dir = os.path.join(self.data_root, "users", username)
        os.makedirs(os.path.join(user_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(user_dir, "conversations"), exist_ok=True)
        return "注册成功！请登录。"