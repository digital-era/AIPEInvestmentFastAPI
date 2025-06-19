from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import akshare as ak
import pandas as pd

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using akshare."
)

class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")

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
            # --- START OF MODIFICATION ---
            
            # 1. 为股票代码添加市场前缀 (sh/sz)
            # A股主板/科创板以'6'开头，创业板以'3'开头，中小板/主板以'0'开头
            market_code = ""
            if code.startswith(('60', '68')):
                market_code = f"sh{code}"
            elif code.startswith(('00', '30')):
                market_code = f"sz{code}"
            else:
                 # 兜底处理，或者可以抛出错误
                raise HTTPException(status_code=400, detail=f"Invalid stock code format: '{code}'")

            # 2. 调用高效的单个股票查询接口
            target_stock_df = ak.stock_individual_spot_quote(symbol=market_code)

            if target_stock_df.empty:
                raise HTTPException(status_code=404, detail=f"Stock code '{code}' not found.")

            # 3. 从新的DataFrame结构中提取数据
            # ak.stock_individual_spot_quote 返回的列名是 'item' 和 'value'
            # 我们需要把它转换成我们需要的格式
            stock_data = dict(zip(target_stock_df['item'], target_stock_df['value']))

            # 4. 准备响应数据，注意数据类型转换
            data = {
                "name": stock_data.get('名称'),
                "latestPrice": float(stock_data.get('最新价', 0)),
                "changePercent": float(stock_data.get('涨跌幅', 0)),
                "changeAmount": float(stock_data.get('涨跌额', 0))
            }
            return PriceResponse(**data)
            
            # --- END OF MODIFICATION ---

        except Exception as e:
            # 捕获所有可能的异常，包括HTTPException和akshare的内部错误
            if isinstance(e, HTTPException):
                raise e # 如果是HTTPException，直接重新抛出
            print(f"Error processing price request for {code}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error fetching price data: {str(e)}")

    elif query_type == 'info':
        try:
            # (info部分代码保持不变)
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
            clean_data = {k: (v if pd.notna(v) else None) for k, v in raw_data.items()}
            
            return InfoResponse(**clean_data)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            print(f"Error processing info request for {code}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error fetching financial info: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")
