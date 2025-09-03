# app.py (FastAPI 版本)
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录的父目录的父目录
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# 拼接automl_runs目录路径
RUNS_DIRECTORY = os.path.join(parent_parent_dir, "automl_runs")

# 创建 FastAPI 应用实例
app = FastAPI(
    title="AutoML Visualization API",
    description="用于提供 AutoML 运行可视化结果的 API 服务",
    version="1.0.0"
)

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/runs/{path:path}")
async def serve_run_files(path: str):
    """
    这个路由可以访问每次运行生成的所有文件。
    例如，URL + /runs/20230101120000/visualization/prediction_vs_actual/plot.png
    将会访问 'automl_runs/20230101120000/visualization/prediction_vs_actual/plot.png'
    """
    file_path = os.path.join(RUNS_DIRECTORY, path)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="文件未找到")

    # 检查是否为文件而不是目录
    if not os.path.isfile(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="请求的路径不是文件")

    return FileResponse(file_path)


@app.get("/")
async def root():
    """
    根路由，返回 API 基本信息
    """
    return {
        "message": "AutoML Visualization API",
        "version": "1.0.0",
        "runs_directory": RUNS_DIRECTORY,
        "usage": "使用 /runs/{path} 来访问运行结果文件"
    }


@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    # 运行服务器，绑定到所有 IP 地址
    uvicorn.run(
        "visualization_app:app",  # 使用导入字符串格式：模块名:应用实例名
        host="0.0.0.0",  # 监听所有网络接口，允许通过 IP 访问
        port=8005,
        reload=True,  # 开发模式下自动重载
        log_level="info"
    )
