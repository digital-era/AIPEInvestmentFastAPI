from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import akshare as ak
import pandas as pd

# 1. 初始化 FastAPI 应用
app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using akshare."
)

# 2. 使用 Pydantic 定义响应模型，提供数据契约和自动文档
class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")


# 3. 定义 API 端点，使用 async def
# FastAPI 会智能地在线程池中运行同步的 akshare 代码，不会阻塞事件循环
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price' or 'info'")
):
    """
    Fetches stock data based on the code and query type.
    """
    if query_type == 'price':
        try:
            stock_spot_df = ak.stock_zh_a_spot_em()
            target_stock = stock_spot_df[stock_spot_df['代码'] == code]
            
            if target_stock.empty:
                raise HTTPException(status_code=404, detail=f"Stock code '{code}' not found in real-time market data.")

            data = {
                "name": target_stock.iloc[0]['名称'],
                "latestPrice": target_stock.iloc[0]['最新价'],
                "changePercent": target_stock.iloc[0]['涨跌幅'],
                "changeAmount": target_stock.iloc[0]['涨跌额']
            }
            return PriceResponse(**data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error fetching price data: {str(e)}")

    elif query_type == 'info':
        try:
            market_code = f"sz{code}" if code.startswith(('0', '3')) else f"sh{code}"
            financial_df = ak.stock_financial_analysis_indicator(symbol=market_code)
            
            if financial_df.empty:
                raise HTTPException(status_code=404, detail=f"Financial data for stock code '{code}' not found.")

            latest_data = financial_df.iloc[0]
            
            raw_data = {
                "pe": latest_data['市盈率(TTM)'],
                "pb": latest_data.get('市净率', latest_data.get('市净率(MRQ)')),
                "roe": latest_data['净资产收益率(摊薄)']
            }
            # 清理 NaN/None 值
            clean_data = {k: (v if pd.notna(v) else None) for k, v in raw_data.items()}
            
            return InfoResponse(**clean_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error fetching financial info: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")

# 注意：在 Vercel 中，它会直接使用 app 对象，不需要下面的 main block。
# 这个 block 仅用于本地测试: uvicorn api.query:app --reload
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
