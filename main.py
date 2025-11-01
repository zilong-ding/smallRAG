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
import numpy as np
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

        self.conversion_list_value = pd.DataFrame(
            [],
            columns=["选择", "标题", "修改时间","conversion_id"]
        )
        self.conversion_list_value = self.conversion_list_value.astype({
            "选择": "bool",
            "标题": "string",
            "修改时间": "string",
            "conversion_id": "int"
        })
        self.history = []

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
            self.getConversions(self.workspace_dropdown.value)
            self.getConversation()
            # self.file_list.value = self.file_list_value
            return (gr.update(visible=False),gr.update(visible=True),
                    username,"登录成功",self.file_list_value,
                    self.conversion_list_value.iloc[:,:3],self.history)
        else:
            return (gr.update(),gr.update(),
                    username,"登录失败",self.file_list_value,
                    self.conversion_list_value.iloc[:,:3],self.history)

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
                outputs=[self.login_page, self.main_page, current_user, self.login_msg, self.file_list,self.conversion_list,self.chatbot]
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

    def getConversation(self):
        # df_sorted = df.sort_values(by=df.columns[3])
        df_sorted = self.conversion_list_value.sort_values(by=self.conversion_list_value.columns[2])
        self.current_conversion  = df_sorted.iloc[len(df_sorted)-1]
        workspace_name = self.workspace_dropdown.value
        self.setCurrentConversation(workspace_name)

    def setCurrentConversation(self,workspace_name):
        _,title,_,conversation_id = self.current_conversion.values
        print("当前转换：",workspace_name,title,conversation_id)
        responses = requests.get(f"{BASE_URL}/workspaces/{self.current_user}/{workspace_name}/{conversation_id}")
        if responses.status_code == 200:
            messages = responses.json()
            if messages:
                print("获取转换历史消息成功")
                self.history = []
                for message in messages:
                    self.history.append({"role": message["role"], "content": message["content"]})
            else:
                print("获取转换历史消息失败")
        else:
            print("http 获取转换历史消息失败")

    def select_conversion(self, df: pd.DataFrame, workspace_name: str):
        if df.empty:
            return self.history

        # 获取布尔数组（不依赖 index）
        selected_bool = df["选择"].values  # shape: (n,)
        selected_indices = np.where(selected_bool)[0]

        if len(selected_indices) != 1:
            return self.history

        pos = selected_indices[0]  # 整数位置

        # 确保 pos 在 self.conversion_list_value 范围内
        if pos >= len(self.conversion_list_value):
            return self.history

        self.current_conversion = self.conversion_list_value.iloc[pos]
        self.setCurrentConversation(workspace_name)
        return self.history



    def getConversions(self,workspace_name):
        response = requests.get(f"{BASE_URL}/workspaces/{self.current_user}/{workspace_name}")
        if response.status_code == 200:
            conversions = response.json()
            if conversions:
                print("获取转换成功")
                self.conversion_list_value = self.conversion_list_value.iloc[0:0]
                for conversion in conversions:
                    self.conversion_list_value.loc[len(self.conversion_list_value)] = [False,conversion["title"], conversion["updated_at"], conversion["conversation_id"]]
            else:
                print("获取转换失败")
        else:
            print("http 获取转换失败")
        self.conversion_list_value = self.conversion_list_value.sort_values(by=self.conversion_list_value.columns[2])

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
                self.chatbot = gr.Chatbot(height=400,type="messages",label="对话窗口")
                self.msg_input = gr.Textbox(label="消息", lines=2, placeholder="输入您的问题...")
                self.send_btn = gr.Button("发送", variant="primary")
                self.send_btn.click(
                    self.send_message,
                    inputs=[self.msg_input, self.workspace_dropdown],
                    outputs=[self.msg_input,self.chatbot]
                )
                with gr.Row():
                    self.conversion_list = gr.DataFrame(label="选择会话",
                                                        headers=["选择", "标题", "修改时间"],
                                                        static_columns=[1, 2],  # 关键：第1、2、3列（0-indexed）不可编辑
                                                        datatype=["bool", "str", "str"],
                                                        # 第一列为 bool → 显示为 checkbox
                                                        interactive=True,  # 必须为 True 才能编辑 checkbox
                                                        row_count=(0, "dynamic"),
                                                        col_count=(3, "fixed")
                                                        )
                self.select_btn = gr.Button("选择会话")
                self.select_btn.click(
                    self.select_conversion,
                    inputs=[self.conversion_list,self.workspace_dropdown],
                    outputs=[self.chatbot]
                )

        self.logout_btn = gr.Button("退出登录", variant="stop")

    def send_message(self, question:str,workspace_name: str):
        inputMessage = question
        self.history.append({"role": "user", "content": question})
        _,title,_,conversation_id = self.current_conversion.values
        # 转换为 Python 原生类型
        title = str(title) if pd.notna(title) else ""
        conversation_id = int(conversation_id)  # 或 conversation_id_raw.item()
        payload = {
            "question": question,
            "workspace_name": workspace_name,
            "user_name": self.current_user,
            "conversation_name": title,
            "conversation_id" : conversation_id
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload)
        if response.status_code == 200:
            message = response.json()
            self.history.append({"role": "assistant", "content": message["answer"]})
            inputMessage = ""
        else:
            message = "无法回答"
        return inputMessage,self.history

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
