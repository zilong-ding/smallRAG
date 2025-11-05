import os

import numpy as np
import torch

os.environ["OPENAI_API_KEY"] = "sk-ea07bf0880504b75a31b1bce38437fcf"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
import openai
from typing import Optional,List,Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

class Embedding:
    def __init__(self):
        self.model_path = "/home/dzl/PycharmProjects/SmallRag/BAAI/bge-base-zh-v1.5"
        self.model = SentenceTransformer(self.model_path)

    def embed(self,text:str)->np.ndarray:
        return self.model.encode(text)

    def check_similarity(self,embedding1,embedding2):
        similarity = self.model.similarity(embedding1, embedding2)
        return similarity.numpy()

class RankModel:
    def __init__(self):
        self.model_path = "/home/dzl/PycharmProjects/SmallRag/BAAI/bge-reranker-base"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def rank(self,question,answers:list[str])->np.ndarray:
        pairs = [[question, c] for c in answers]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            print(scores)
            return scores.cpu().numpy()



class ChatCompletion:
    def __init__(self):
        self.model = "qwen-max"
        self.system_prompt = """你是一个智能助手，请直接、准确地回答用户的问题。不要添加解释、前缀或后缀，仅输出答案本身。"""
        self.system_prompt_context = """你是一个智能助手，请根据以下规则回答用户问题：
                                        1. 如果提供的上下文与用户问题相关，请严格依据上下文内容作答，不要编造信息。
                                        2. 如果上下文与问题无关、信息不足或无法回答，请直接回答“无法根据提供的信息回答该问题。”不要自己回答
                                        3. 回答应简洁明了，仅输出答案本身，不要添加解释、前缀（如“答案是：”）或后缀。"""
        self.client = openai.OpenAI(
            api_key="sk-ea07bf0880504b75a31b1bce38437fcf",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 修复多余空格
        )

    def answer_question(
        self,
        text: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        智能问答函数，支持多轮对话历史和外部上下文。
        Args:
            text (str): 当前用户的问题。
            context (str, optional): 外部检索到的相关文档（用于 RAG）。默认为 None。
            history (List[Dict], optional): 对话历史，格式如：
                [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好！"}]
                默认为 None。
        Returns:
            str: 模型生成的答案。
        """
        # 选择 system prompt
        if context and context.strip():
            system_prompt = self.system_prompt_context
            # 将 context 注入到当前用户问题中（作为任务的一部分）
            current_user_message = f"上下文：{context}\n\n用户问题：{text}"
        else:
            system_prompt = self.system_prompt
            current_user_message = text
        # 构建完整消息列表
        messages = [{"role": "system", "content": system_prompt}]
        # 添加历史对话（如果提供）
        if history:
            # 可选：校验 history 格式（此处简化）
            messages.extend(history)
        # 添加当前用户消息
        messages.append({"role": "user", "content": current_user_message})
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"调用大模型失败: {e}") from e


if __name__ == "__main__":
    qa = ChatCompletion()
    ans = qa.answer_question("Python中如何反转列表？")
    print(ans)

    ans = qa.answer_question(
        text="作者是谁？",
        context="《三体》是中国作家刘慈欣创作的科幻小说。"
    )
    print(ans)  # 应输出：刘慈欣


    history = [
        {"role": "user", "content": "《三体》是中国作家刘慈欣创作的科幻小说。"},
        {"role": "assistant", "content": "您好！有什么可以帮您？"}
    ]

    ans = qa.answer_question(
        text="这本书的作者是谁？",
        context="李白",
        history=history
    )
    print(ans)