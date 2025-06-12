# api/daily_task.py
from fastapi import FastAPI, Request
from datetime import datetime
import os

app = FastAPI()

@app.get("/api/daily_task")
def run_daily_task(request: Request):
    # 最佳实践：验证请求是否来自 Vercel Cron
    # 对于 Hobby 套餐，你需要自己设置一个秘密令牌
    # 对于 Pro/Enterprise 套餐，Vercel 会自动添加一个 'x-vercel-cron-secret' 头
    auth_header = request.headers.get('authorization')
    expected_secret = f"Bearer {os.environ.get('CRON_SECRET')}"

    if auth_header != expected_secret:
        return {"status": "unauthorized"}, 401

    # --- 在这里执行你的定时任务逻辑 ---
    current_time = datetime.utcnow().isoformat()
    print(f"Daily task executed at: {current_time}")
    # 例如：连接数据库、清理旧数据、发送邮件等
    # ... 任务逻辑 ...

    # 返回一个成功的响应
    return {"status": "success", "executed_at": current_time}
