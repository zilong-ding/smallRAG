import os

# 清除所有代理变量
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if var in os.environ:
        del os.environ[var]

import gradio as gr
from typing import Callable, Optional
from AuthManager import AuthManager
import requests
import pandas as pd
# FastAPI 服务地址
BASE_URL = "http://localhost:8000"



class RAGChatApp:
    def __init__(
            self,
            title: str = "多用户 RAG 聊天系统"
    ):
        """
        初始化 RAG 聊天应用
        Args:
            title: 应用标题
        """
        # self.auth_manager = auth_manager
        self.title = title
        self.current_user = ""
        self.workspace = None
        # 先创建空 DataFrame（不指定 dtype）
        self.file_list_value = pd.DataFrame(
            [],
            columns=["选择", "文件名", "大小", "修改时间"]
        )

        # 再显式转换列类型
        self.file_list_value = self.file_list_value.astype({
            "选择": "bool",
            "文件名": "string",  # 推荐用 "string" 而非 "str"
            "大小": "string",
            "修改时间": "string"
        })
        self.demo = self._build_ui()

    def upload_to_fastapi(self,files):
        if not files:
            return "❌ 未选择文件",self.file_list_value, gr.update(value=None)
        workspace_name = self.workspace_dropdown.value
        results = []
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, "rb") as f:
                    # 构造 multipart/form-data 请求
                    files_payload = {
                        "file": (filename, f, "application/octet-stream")
                    }

                    # 其他参数通过 data 传递
                    data_payload = {
                        "current_user": self.current_user,
                        "workspace_name": workspace_name
                    }
                    print(f"workspace_name={workspace_name}, current_user={self.current_user}")

                    # 发送 POST 请求
                    resp = requests.post(
                        f"{BASE_URL}/workspaces/upload",
                        files=files_payload,  # 用 files 上传文件
                        data=data_payload,  # 用 data 上传普通参数
                        # headers={"Authorization": f"Bearer {token}"}  # 如需认证
                    )
                if resp.status_code == 200:
                    r = resp.json()
                    results.append(f"✅ {r["message"]} ")
                    if "更新" not in r["message"]:
                        self.file_list_value.loc[len(self.file_list_value)] = [False,r["filename"],r["size"],r["modified"]]
                else:
                    results.append(f"❌ {filename} 失败: {resp.text}")
            except Exception as e:
                results.append(f"❌ {filename} 异常: {str(e)}")

        return "\n".join(results),self.file_list_value, gr.update(value=None)

    def on_logout(self):
        return gr.update(visible=True), gr.update(visible=False), ""

    def on_login(self, username, password):
        payload = {
            "username": username,
            "password": password,
        }
        response = requests.post(f"{BASE_URL}/login", json=payload)
        if response.status_code == 200:
            # 登录成功后自动跳转到主页
            self.current_user = username
            self.workspaceChoices = self.getWorkspace()
            self.getWorkspaceFiles()
            # self.file_list.value = self.file_list_value
            return gr.update(visible=False),gr.update(visible=True),username,"登录成功",self.file_list_value
        else:
            return gr.update(),gr.update(),username,"登录失败",self.file_list_value

    def on_register(self, username, password, confirm):
        if password != confirm:
            return "两次密码不一致！",gr.update(), gr.update()
        # msg = self.auth_manager.register_user(username, password)
        payload = {
            "username": username,
            "password": password,
            "confirm": confirm
        }
        response = requests.post(f"{BASE_URL}/register", json=payload)
        if response.status_code == 200:
            # 注册成功后自动跳转到登录页
            return "注册成功", gr.update(visible=True), gr.update(visible=False)
        else:
            return "注册失败！", gr.update(), gr.update()


    def _build_ui(self):
        with gr.Blocks(title=self.title, theme=gr.themes.Soft()) as demo:
            # 全局状态：当前登录用户
            current_user = gr.State("")

            # ========== 页面容器 ==========
            with gr.Column(visible=True) as self.login_page:
                self._build_login_ui()

            with gr.Column(visible=False) as self.register_page:
                self._build_register_ui()

            with gr.Column(visible=False) as self.main_page:
                self._build_main_ui()

            # ========== 事件绑定 ==========
            # 登录页 ↔ 注册页切换
            self.to_register_btn.click(
                self._switch_to_register,
                outputs=[self.login_page, self.register_page]
            )
            self.to_login_btn.click(
                self._switch_to_login,
                outputs=[self.login_page, self.register_page]
            )

            # 功能按钮绑定
            self.login_btn.click(
                self.on_login,
                inputs=[self.login_username, self.login_password],
                outputs=[self.login_page, self.main_page, current_user, self.login_msg, self.file_list]
            )

            self.reg_btn.click(
                self.on_register,
                inputs=[self.reg_username, self.reg_password, self.reg_confirm],
                outputs=[self.reg_msg, self.login_page, self.register_page]
            )

            self.logout_btn.click(
                self.on_logout,
                outputs=[self.login_page, self.main_page, current_user]
            )

            # 用户切换时刷新主页数据（预留扩展点）
            current_user.change(
                lambda user: gr.update() if user else gr.update(),
                inputs=current_user,
                outputs=[]
            )

        return demo

    def _build_login_ui(self):
        """构建登录页面"""
        gr.Markdown("## 🔑 用户登录")
        self.login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
        self.login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
        self.login_btn = gr.Button("登录", variant="primary")
        self.login_msg = gr.Textbox(label="提示", interactive=False)
        self.to_register_btn = gr.Button("没有账号？去注册", size="sm")

    def _build_register_ui(self):
        """构建注册页面"""
        gr.Markdown("## 📝 注册新账号")
        self.reg_username = gr.Textbox(label="用户名", placeholder="3-20位，仅字母、数字、下划线")
        self.reg_password = gr.Textbox(label="密码", type="password", placeholder="至少6位")
        self.reg_confirm = gr.Textbox(label="确认密码", type="password")
        self.reg_btn = gr.Button("注册", variant="primary")
        self.reg_msg = gr.Textbox(label="提示", interactive=False)
        self.to_login_btn = gr.Button("已有账号？去登录", size="sm")

    def getWorkspace(self):
        response = requests.get(f"{BASE_URL}/workspaces", params={"current_user": self.current_user})
        if response.status_code == 200:
            workspaces = response.json()
            if workspaces:
                print("获取工作区成功")
                return workspaces
            else:
                print("获取工作区失败")
                return ["default"]
        else:
            print("http 获取工作区失败")
            return ["default"]

    def getWorkspaceFiles(self):
        response = requests.get(f"{BASE_URL}/workspaces/files", params={"workspace_name": self.workspace_dropdown.value,
                                                                        "current_user": self.current_user})
        if response.status_code == 200:
            files = response.json()
            if files:
                self.file_list_value = self.file_list_value.iloc[0:0]
                print("获取文件成功")
                for file in files:
                    self.file_list_value.loc[len(self.file_list_value)] =[False,file["name"], file["size"], file["modified"]]
            else:
                print("获取文件失败")
                # return []
        else:
            print("http 获取文件失败")
            # return []

    def _build_main_ui(self):
        """构建主页面（聊天+文件管理）"""
        gr.Markdown("# 🤖 多用户 RAG 聊天系统")
        self.workspaceChoices = self.getWorkspace()

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📁 文件管理")
                self.workspace_dropdown = gr.Dropdown(label="当前工作区", choices=self.workspaceChoices,interactive= True)
                self.file_upload = gr.File(file_count="multiple", label="上传文件",height=80)
                self.upload_btn = gr.Button("上传", variant="primary")
                self.upload_output = gr.Textbox(label="上传结果", lines=2)
                with gr.Row():
                    self.create_folder = gr.Textbox(label="新建工作区", scale=1)
                    self.create_btn = gr.Button("创建", scale=1)
                    self.rag_enabled = gr.Checkbox(label="启用 RAG", value=True,scale=1)
                self.file_list = gr.DataFrame(label="文件列表",
                                              headers=["选择", "文件名", "大小", "修改时间"],
                                              static_columns=[1, 2, 3],  # 关键：第1、2、3列（0-indexed）不可编辑
                                              datatype=["bool", "str", "str", "str"],  # 第一列为 bool → 显示为 checkbox
                                              interactive=True,  # 必须为 True 才能编辑 checkbox
                                              row_count=(0, "dynamic"),
                                              col_count=(4, "fixed")
                                              )
                self.upload_btn.click(
                    self.upload_to_fastapi,
                    inputs=self.file_upload,
                    outputs=[self.upload_output, self.file_list,self.file_upload]
                )
                self.delete_rows_btn = gr.Button("删除选中行")
                self.delete_output = gr.Textbox(label="删除结果", lines=2)
                self.delete_rows_btn.click(
                    self.delete_rows,
                    inputs=[self.file_list,self.workspace_dropdown],
                    outputs=[self.file_list,self.delete_output]
                )
                self.workspace_dropdown.change(self.change_workspace)


            with gr.Column():
                gr.Markdown("### 💬 聊天")
                self.chatbot = gr.Chatbot(height=400)
                self.msg_input = gr.Textbox(label="消息", lines=2, placeholder="输入您的问题...")
                self.send_btn = gr.Button("发送", variant="primary")
                with gr.Row():
                    self.conversion_list = gr.DataFrame(label="选择会话", headers=["标题", "修改时间"])

        self.logout_btn = gr.Button("退出登录", variant="stop")

    def change_workspace(self):
        pass

    def delete_rows(self, df: pd.DataFrame,workspace_name: str):
        message = ""
        if df.empty:
            return df, message

        # 1. 提取所有选中的文件名
        selected_mask = df["选择"] == True
        selected_files = df[selected_mask]["文件名"].tolist()

        if not selected_files:
            return df, "未选择任何文件"

        success_files = []
        failed_files = []

        # 2. 逐个发送 DELETE 请求
        for file_name in selected_files:
            try:
                url = f"{BASE_URL}/workspaces/{self.current_user}/{workspace_name}/documents/{file_name}"
                # print("url", url)
                response = requests.delete(url, timeout=10)  # 使用 DELETE

                if response.status_code == 200:
                    try:
                        results = response.json()
                        if results.get("success", False):
                            success_files.append(file_name)
                        else:
                            failed_files.append((file_name, results.get("message", "未知错误")))
                    except ValueError:
                        # 响应不是 JSON
                        failed_files.append((file_name, "响应格式错误"))
                else:
                    failed_files.append((file_name, f"HTTP {response.status_code}"))

            except requests.RequestException as e:
                failed_files.append((file_name, f"请求异常: {str(e)}"))

        # 3. 更新 DataFrame：移除所有成功删除的行
        if success_files:
            # 保留未被成功删除的行（注意：可能部分成功）
            df = df[~df["文件名"].isin(success_files)].copy()
            df = df.reset_index(drop=True)

        # 4. 构造返回消息
        parts = []
        if success_files:
            parts.append(f"成功删除 {len(success_files)} 个文件: {', '.join(success_files)}")
        if failed_files:
            fail_msgs = [f"{f}: {msg}" for f, msg in failed_files]
            parts.append(f"删除失败 ({len(failed_files)} 个): " + "; ".join(fail_msgs))

        message = "; ".join(parts) if parts else "无文件被删除"
        self.file_list_value = df

        return df, message

    @staticmethod
    def _switch_to_register():
        return gr.update(visible=False), gr.update(visible=True)

    @staticmethod
    def _switch_to_login():
        return gr.update(visible=True), gr.update(visible=False)



    def launch(self, **kwargs):
        """启动应用"""
        default_kwargs = {
            # "server_name": "0.0.0.0",
            # "server_port": 7860,
            # "show_api": False
        }
        default_kwargs.update(kwargs)
        self.demo.launch(**default_kwargs)


if __name__ == "__main__":
    app = RAGChatApp(
        title="多用户 RAG 系统",
    )
    app.launch()
