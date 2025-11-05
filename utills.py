from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber  # pip install pdfplumber
from typing import List,Tuple

# 推荐：显式指定适合中文的分隔符序列（从粗到细）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 每个 chunk 的最大长度（单位由 length_function 决定）
    chunk_overlap=50,        # 相邻 chunk 之间的重叠长度（建议 10%~20% 的 chunk_size）
    length_function=len,     # 当前按字符数计算（对中文基本可用，但非最精确）
    separators=[
        "\n\n",              # 段落分隔（优先尝试）
        "\n",                # 换行
        "。", "！", "？",     # 中文句末标点
        "；", "：", "，",     # 中文句中停顿标点
        " ",                 # 空格
        ""                   # 最后 fallback 到任意位置切分（避免超长）
    ],
    is_separator_regex=False,  # separators 是普通字符串，非正则表达式
    keep_separator=True,       # 保留分隔符（如句号）在 chunk 末尾，更自然
)

def split_text(text: str)->list[str]:
    return text_splitter.split_text(text)

def extract_with_pdfplumber(pdf_path: str) -> tuple[list[str], list[int]]:
    """适合含表格/复杂布局的 PDF"""
    texts = []
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # 提取文本（可设置 x/y 坐标过滤水印）
            page_text = page.extract_text(
                layout=True,  # 保留空格对齐
                x_tolerance=2,
                y_tolerance=2
            )
            if page_text:
                texts.append(page_text)
                pages.append(page.page_number)

    return texts, pages


def extract_and_split_with_pages(pdf_path: str) -> Tuple[List[List[str]], List[int]]:
    """
    解析 PDF 并按页分割文本，返回：
    - chunks: 所有文本块（List[str]）
    - page_numbers: 每个 chunk 对应的页码（List[int]，从 1 开始）

    保证 len(chunks) == len(page_numbers)
    """
    all_chunks: List[str] = []
    all_page_numbers: List[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(
                layout=True,
                x_tolerance=2,
                y_tolerance=2
            )
            if not page_text or not page_text.strip():
                continue  # 跳过空白页

            # 对当前页的文本进行分割
            page_chunks = text_splitter.split_text(page_text)

            # 将当前页的所有 chunks 添加到全局列表
            all_chunks.extend(page_chunks)
            # 每个 chunk 都标记为当前页码
            all_page_numbers.extend([page.page_number] * len(page_chunks))

    return all_chunks, all_page_numbers

if __name__ == "__main__":
    texts, pages = extract_and_split_with_pages("/home/dzl/PycharmProjects/SmallRag/data/users/dzl/uploads/default/RAG学习.pdf")
    print(texts)
    print(pages)