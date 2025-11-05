import os
# 清除所有代理变量
for var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if var in os.environ:
        del os.environ[var]
import uvicorn
from celery import Celery
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form,BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime

from werkzeug.security import generate_password_hash,check_password_hash
from dataSQL import User,get_db,Workspace,Document,Conversation
from dataSchames import (RegisterRequest,RegisterResponse,UserResponse,ConversationsResponse,
                         LoginRequest,FileItem,chatRequest,chatResponse,Message)
from model import ChatCompletion,Embedding,RankModel
from dataES import (DocumentMeta,ChunkInfo,QAHistory,ImageInfo,SmallRAGDB)
from typing import List,Dict
import shutil
import hashlib
from utills import split_text,extract_with_pdfplumber,extract_and_split_with_pages

celery_app = Celery("rag", broker="redis://localhost:6379")
app = FastAPI(title="多用户 RAG 系统 API")
llm = ChatCompletion()
embed = Embedding()
ranker = RankModel()
ESDB = SmallRAGDB(es_url="http://localhost:9200")
ESDB.init_indices(overwrite=False)


# @celery_app.task(bind=True, max_retries=3, autoretry_for=(Exception,))
def process_pdf_task(file_path:str,doc_id:int,workspace_id:int,user_username:str):
    try:
        text_blocks,all_page_numbers = extract_and_split_with_pages(file_path)
        full_text = "\n\n".join(text_blocks)
        now = datetime.utcnow()
        doc_meta = DocumentMeta(
            doc_id=str(doc_id),
            workspace_id=str(workspace_id),
            user_username=user_username,
            title=os.path.basename(file_path),
            file_name=os.path.basename(file_path),
            abstract=full_text[:500] + "..." if len(full_text) > 500 else full_text,
            full_content=full_text,  # 可选：若不需要全文检索可省略
            embedding_status="processing",
            file_size=os.path.getsize(file_path),
            file_hash="TODO_compute_hash",  # 建议计算 SHA256
            created_at=now,
            updated_at=now,
        )
        print(f" 增加 doc_meta{doc_id} 结果",ESDB.create_document(str(doc_id),doc_meta))

        chunk_list = []
        chunk_id = 0
        for page_chunks,page_num in zip(text_blocks,all_page_numbers):
            chunk_info = ChunkInfo(
                chunk_id=f"{doc_id}_chunk_{chunk_id*(page_num + 1)}",
                doc_id=str(doc_id),
                workspace_id=str(workspace_id),
                user_username=user_username,
                chunk_content=page_chunks,
                page_number=page_num,
                created_at=datetime.utcnow(),
                embedding_vector = embed.embed(page_chunks).tolist(),
                chunk_order = chunk_id*(page_num + 1),
            )
            chunk_list.append(chunk_info)
            chunk_id += 1

        ESDB.bulk_create_chunks(chunk_list)
        ESDB.refresh_all()
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误：{e}")




@app.get("/workspaces/{current_user}/{workspace_name}/{conversation_id}",response_model=List[Message])
async def get_conversation(
    current_user: str,
    workspace_name: str,
    conversation_id: int,
    db: Session = Depends(get_db)
):
    # 获取用户
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    # 获取工作区
    workspace = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    )
    if not workspace.first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="工作区不存在或无权限"
        )
    # 查询文档
    conversations = db.query(Conversation).filter(
        Conversation.user_username == current_user,
        Conversation.workspace_name == workspace_name,
        Conversation.id == conversation_id
    ).first()
    if not conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="对话不存在或无权限"
        )
    responses = []
    for message in conversations.messages:
        response = Message(
            role=message["role"],
            content=message['content'],
            timestamp = datetime.utcnow()
        )
        responses.append(
            response
        )

    return responses


@app.get("/workspaces/{current_user}/{workspace_name}",response_model=List[ConversationsResponse])
async def get_conversations(
    current_user: str,
    workspace_name: str,
    db: Session = Depends(get_db)
):
    # 查询用户
    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    # 查询工作区
    workspace = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    ).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="工作区不存在或无权限"
        )
    # 查询文档
    conversations = db.query(Conversation).filter(
        Conversation.user_username == current_user,
        Conversation.workspace_name == workspace_name
    ).all()
    returnResults = []
    for conversation in conversations:
        returnResults.append(
            {
                "title": conversation.title, # 显示使用的标题
                "updated_at": conversation.updated_at, # 显示修改时间，排序用
                "conversation_id": conversation.id
            }
        )
    return returnResults


@app.post("/chat", response_model=chatResponse)
async def chat(request: chatRequest, db: Session = Depends(get_db)):
    question = request.question
    current_user = request.user_name
    conversation_name = request.conversation_name  # 修正变量名
    workspace_name = request.workspace_name  # 修正变量名
    conversation_id = request.conversation_id

    workspace = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    ).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="工作区不存在或无权限"
        )

    # 查询是否存在该对话
    existing_conversation = db.query(Conversation).filter(
        Conversation.title == conversation_name,
        Conversation.user_username == current_user,
        Conversation.id == conversation_id
    ).first()
    question_vector = embed.embed(question).tolist()
    results = ESDB.hybrid_search_chunks(question,question_vector,
                                        workspace_id=str(workspace.id),
                                        username=current_user,
                                        top_k_text=10,top_k_vector=10)
    bm25_hits,bge_hits = results["text_hits"],results["vector_hits"]
    print("bm25_hits:",len(bm25_hits))
    print("bge_hits:",len(bge_hits))

    context = None

    # Step 1: RRF 融合（按 chunk_id）
    k_rrf = 60
    fusion_score = {}

    # 添加 BM25 结果
    for rank, hit in enumerate(bm25_hits):
        print("text: ", hit["chunk_content"])
        chunk_id = hit["chunk_id"]
        fusion_score[chunk_id] = fusion_score.get(chunk_id, 0) + 1.0 / (rank + 1 + k_rrf)

    # 添加 BGE 向量结果
    for rank, hit in enumerate(bge_hits):
        print("text: ", hit["chunk_content"])
        chunk_id = hit["chunk_id"]
        fusion_score[chunk_id] = fusion_score.get(chunk_id, 0) + 1.0 / (rank + 1 + k_rrf)

    # 按 RRF 分数排序，取 top candidates（例如前 20 用于 rerank）
    rrf_sorted = sorted(fusion_score.items(), key=lambda x: x[1], reverse=True)
    top_candidate_ids = [cid for cid, score in rrf_sorted[:10]]  # 取前10个 chunk_id

    if not top_candidate_ids:
        final_results = []
    else:
        # Step 2: 获取完整 chunk 内容（去重后）
        candidate_chunks = []
        seen = set()
        for hit in bm25_hits + bge_hits:
            cid = hit["chunk_id"]
            if cid in top_candidate_ids and cid not in seen:
                candidate_chunks.append(hit)
                seen.add(cid)
                if len(candidate_chunks) >= len(top_candidate_ids):
                    break

        # Step 3: 用 RankModel 重排序
        contents = [chunk["chunk_content"] for chunk in candidate_chunks]
        print("contents:",contents)
        rerank_scores = ranker.rank(question, contents)  # shape: (N,)

        # 绑定分数并排序
        scored_chunks = [(chunk, score) for chunk, score in zip(candidate_chunks, rerank_scores)]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Step 4: 返回最终 top-k（例如 top 5）
        final_results = [chunk for chunk, score in scored_chunks[:5]]

    context = "\n".join([chunk["chunk_content"] for chunk in final_results])
    print("最终结果：", context)

    if existing_conversation:
        conversation_id = existing_conversation.id
        # 假设 messages 是一个 JSON 列，存储 [{"role": "user", "content": "..."}, ...]
        history: List[Dict[str, str]] = existing_conversation.messages or []
        # 调用 LLM 生成回答（传入历史）
        answer = llm.answer_question(question, history=history, context=context)

        # 将新交互加入历史
        new_history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        existing_conversation.messages = new_history
        existing_conversation.updated_at = datetime.utcnow()
        db.commit()
    else:
        # 新对话：无历史
        answer = llm.answer_question(question, context=context)
        new_conversation = Conversation(
            user_username=current_user,
            title = question,
            workspace_name=workspace_name,  # 注意：模型字段名是否为 workspaces_name？
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        )
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        conversation_id = new_conversation.id

    return chatResponse(
        answer=answer,
        conversation_name=conversation_name,
    )





@app.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # 检查用户名是否已存在
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已存在"
        )

    # 创建用户
    password_hash = generate_password_hash(request.password)
    new_user = User(
        username=request.username,
        password_hash=password_hash,
        created_at=datetime.utcnow()
    )
    ws = Workspace(
        name = "default",
        description = "默认工作区",
        created_at = datetime.utcnow(),
        user_username = request.username
    )
    db.add(new_user)
    db.add(ws)
    db.commit()
    db.refresh(new_user)

    # 创建用户目录
    user_dir = f"./data/users/{request.username}"
    os.makedirs(f"{user_dir}/uploads", exist_ok=True)
    os.makedirs(f"{user_dir}/conversations", exist_ok=True)

    return RegisterResponse(
        success=True,
        message="注册成功！",
        username=new_user.username
    )

@app.post("/login", response_model=RegisterResponse)
async def login(UserInput:LoginRequest, db: Session = Depends(get_db)):
    """
    用户登录
    """
    username = UserInput.username
    password = UserInput.password
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.verify_password(password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )

    return RegisterResponse(
        success=True,
        message="登录成功！",
        username=user.username
    )

@app.get("/workspaces")
async def get_workspaces(current_user: str, db: Session = Depends(get_db)):
    """
    获取当前用户的所有工作区
    """
    workspaces = db.query(Workspace).filter(
        Workspace.user_username == current_user
    ).all()
    return [ws.name for ws in workspaces]


# 上传文件和修改 文件
@app.post("/workspaces/upload")
async def upload_file(
    workspace_name: str = Form(...),
    current_user: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. 验证工作区
    workspace = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    ).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="工作区不存在或无权限"
        )

    # 2. 流式计算文件哈希，并保存到磁盘（边读边写）
    file_hash = hashlib.sha256()
    user_upload_dir = f"./data/users/{current_user}/uploads/{workspace_name}"
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = os.path.join(user_upload_dir, file.filename)

    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):
                file_hash.update(chunk)
                f.write(chunk)
        file_size = os.path.getsize(file_path)
    except Exception as e:
        # 清理可能写入的残缺文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件保存失败: {str(e)}"
        )

    hash_hex = file_hash.hexdigest()

    # 3. 检查是否存在同名文件
    same_name_doc = db.query(Document).filter(
        Document.workspace_id == workspace.id,
        Document.filename == file.filename
    ).first()

    if same_name_doc:
        # 同名文件存在
        if same_name_doc.file_hash == hash_hex:
            # 内容相同 → 无变更
            os.remove(file_path)  # 删除刚上传的重复文件
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="文件内容未变更，无需重新上传"
            )
        else:
            # 内容不同 → 执行更新（覆盖）
            # 删除旧文件（磁盘）
            old_file_path = os.path.join("./data/users", current_user, same_name_doc.file_path)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)

            # 更新数据库记录
            same_name_doc.title = os.path.splitext(file.filename)[0]
            same_name_doc.file_path = f"uploads/{workspace_name}/{file.filename}"
            same_name_doc.file_size = file_size
            same_name_doc.file_hash = hash_hex
            same_name_doc.updated_at = datetime.utcnow()
            same_name_doc.embedding_status = "pending"  # 重置嵌入状态
            db.commit()
            db.refresh(same_name_doc)

            return {
                "success": True,
                "message": "文件已更新",
                "document_id": same_name_doc.id,
                "filename": file.filename,
                "size": file_size,
                "modified": same_name_doc.updated_at
            }

    else:
        # 无同名文件 → 检查是否内容重复（不同文件名）
        existing_doc = db.query(Document).filter(
            Document.workspace_id == workspace.id,
            Document.file_hash == hash_hex
        ).first()

        if existing_doc:
            # 内容已存在（不同文件名）
            os.remove(file_path)  # 清理刚上传的文件
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"相同内容的文件已存在（文件名：{existing_doc.filename}）"
            )

    # 4. 构建保存路径
    user_upload_dir = f"./data/users/{current_user}/uploads/{workspace_name}"
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = os.path.join(user_upload_dir, file.filename)

    # 5. 保存文件（使用已读取的内容，或重新写入）
    # 正常新增
    new_doc = Document(
        title=os.path.splitext(file.filename)[0],
        filename=file.filename,
        file_path=f"uploads/{workspace_name}/{file.filename}",
        file_size=file_size,
        file_hash=hash_hex,
        workspace_id=workspace.id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        embedding_status="pending"
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    # task = process_pdf_task.delay(
    #     file_path=str(file_path),  # 传递字符串路径
    #     doc_id=new_doc.id,
    #     workspace_id=workspace.id,
    #     user_username=current_user
    # )
    process_pdf_task(
        file_path=str(file_path),  # 传递字符串路径
        doc_id=new_doc.id,
        workspace_id=workspace.id,
        user_username=current_user
    )

    return {
        "success": True,
        "message": "文件上传成功",
        "document_id": new_doc.id,
        "filename": new_doc.filename,
        "size": file_size,
        "modified": new_doc.updated_at,
        "hash": hash_hex  # 可选返回
    }

@app.delete("/workspaces/{current_user}/{workspace_name}/documents/{document_name}")
async def delete_document(
    workspace_name: str,
    document_name: str,
    current_user: str,
    db: Session = Depends(get_db)
):
    workspace = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    ).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="工作区不存在或无权限"
        )

    doc = db.query(Document).filter(
        Document.filename == document_name,
        Document.workspace_id == workspace.id
    ).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文档不存在"
        )

    # 构建绝对路径并删除磁盘文件
    file_abs_path = os.path.join("./data/users", current_user, doc.file_path)
    try:
        if os.path.exists(file_abs_path):
            os.remove(file_abs_path)
    except Exception as e:
        # 可选：记录日志，但不中断删除数据库记录（或根据需求决定）
        print(f"警告：磁盘文件删除失败 {file_abs_path}: {e}")

    # 删除数据库记录
    db.delete(doc)
    db.commit()

    return {
        "success": True,
        "message": "文档删除成功",
        "document_name": document_name
    }




# 获取工作区文件列表
@app.get("/workspaces/files", response_model=List[FileItem])
async def get_workspace_files(
    workspace_name: str,
    current_user: str ,
    db: Session = Depends(get_db)
):
    # 验证工作区属于当前用户
    ws = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    ).first()
    if not ws:
        raise HTTPException(status_code=404, detail="工作区不存在")

    # 从数据库读取文件信息
    # documents = db.query(Document).filter(
    #     Document.workspace_id == ws.id
    # ).all()
    documents = ws.documents

    files = []
    for doc in documents:
        files.append({
            "name": doc.filename,
            "size": doc.file_size,
            "modified": doc.updated_at,
            "id": doc.id
        })
    return files

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)