from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from sqlalchemy.orm import Session
from datetime import datetime
import os
from werkzeug.security import generate_password_hash,check_password_hash
from dataSQL import User,get_db,Workspace,Document
from dataSchames import RegisterRequest,RegisterResponse,UserResponse,LoginRequest,FileItem
from typing import List
import shutil


app = FastAPI(title="多用户 RAG 系统 API")

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


@app.post("/workspaces/upload")
async def upload_file(
    workspace_name: str = Form(...),  # 通过 Form 获取 workspace_name
    current_user: str = Form(...),  # 通过 Form 获取 current_user
    file: UploadFile = File(...),  # 通过 File 获取上传的文件
    db: Session = Depends(get_db)  # 通过 Depends 获取数据库会话
):
    # 1. 验证工作区是否存在且属于当前用户
    print(f"workspace_name={workspace_name}, current_user={current_user}")
    workspace = db.query(Workspace).filter(
        Workspace.name == workspace_name,
        Workspace.user_username == current_user
    ).first()
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="工作区不存在或无权限"
        )

    # 2. 构建文件保存路径
    user_upload_dir = f"./data/users/{current_user}/uploads/{workspace_name}"
    os.makedirs(user_upload_dir, exist_ok=True)
    file_path = os.path.join(user_upload_dir, file.filename)

    # 3. 保存文件到磁盘
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件保存失败: {str(e)}"
        )
    finally:
        file.file.close()

    # 4. 获取文件元信息
    file_stat = os.stat(file_path)
    file_size = file_stat.st_size

    # 5. 在数据库中创建 Document 记录
    new_doc = Document(
        title=os.path.splitext(file.filename)[0],  # 去掉扩展名作为标题
        filename=file.filename,
        file_path=f"uploads/{workspace_name}/{file.filename}",  # 相对路径
        file_size=file_size,
        workspace_id=workspace.id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        embedding_status="pending"  # 后续由后台任务处理 embedding
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)

    return {
        "success": True,
        "message": "文件上传成功",
        "document_id": new_doc.id,
        "filename": file.filename,
        "modified":new_doc.created_at,
        "size": file_size
    }


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
            "name": doc.title,
            "size": doc.file_size,
            "modified": doc.updated_at
        })
    return files