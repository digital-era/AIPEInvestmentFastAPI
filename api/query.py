from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import akshare as ak
import pandas as pd
import re

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

def get_stock_market_code(code: str) -> str:
    """获取带市场前缀的股票代码"""
    if code.startswith(('60', '68', '900')):  # 沪市A股/B股
        return f"sh{code}"
    elif code.startswith(('00', '30', '200')):  # 深市A股/B股
        return f"sz{code}"
    elif code.startswith('7'):  # 北交所
        return f"bj{code}"
    else:
        raise ValueError(f"Invalid stock code format: '{code}'")

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
            # 方案1：使用新浪财经接口（快速）->修改为雪球接口
            try:
                market_code = get_stock_market_code(code)
                #df = ak.stock_zh_a_spot_sina(symbol=market_code)
                df = ak.stock_individual_spot_xq(symbol=market_code)
                
                # 提取数据并处理百分比
                change_percent = = df.loc[df['item'] == '涨幅', 'value'].iloc[0]
               
                #change_percent = float(change_percent_str.strip('%'))
                
                data = {
                        "name": df.loc[df['item'] == '名称', 'value'].iloc[0],
                        "latestPrice": float(df.loc[df['item'] == '现价', 'value'].iloc[0]),
                        "changePercent": change_percent,
                        "changeAmount": float(df.loc[df['item'] == '涨跌', 'value'].iloc[0])
                }
                return PriceResponse(**data)
                
            except Exception as method1_error:
                print(f" method1 interface failed, trying alternative: {method1_error}")
                """
                # 方案2：备用接口（腾讯财经）
                try:
                    df = ak.stock_zh_a_spot_em()
                    target_df = df[df['代码'] == code]
                    
                    if target_df.empty:
                        raise HTTPException(status_code=404, detail=f"Stock code '{code}' not found.")
                    
                    stock_data = target_df.iloc[0]
                    change_percent_str = stock_data['涨跌幅'].rstrip('%')
                    
                    data = {
                        "name": stock_data['名称'],
                        "latestPrice": float(stock_data['最新价']),
                        "changePercent": float(change_percent_str),
                        "changeAmount": float(stock_data['涨跌额'])
                    }
                    return PriceResponse(**data)
                    
                except Exception as em_error:
                    print(f"EM interface failed: {em_error}")
                    raise
                """

        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            print(f"Error processing price request for {code}: {e}")
            detail = f"Failed to fetch price data: {str(e)}" if "HTTPException" not in str(e) else str(e)
            raise HTTPException(status_code=500, detail=detail)

    elif query_type == 'info':
        try:
            # 优化金融数据获取
            market_code = get_stock_market_code(code)
            financial_df = ak.stock_financial_analysis_indicator(symbol=market_code)
            
            if financial_df.empty:
                raise HTTPException(status_code=404, detail=f"Financial data for stock code '{code}' not found.")

            # 获取最新有效数据（跳过空行）
            latest_data = None
            for i in range(len(financial_df)):
                row = financial_df.iloc[i]
                if not pd.isna(row['净资产收益率(摊薄)']):
                    latest_data = row
                    break
            
            if latest_data is None:
                raise HTTPException(status_code=404, detail=f"No valid financial data found for {code}")
            
            # 处理可能的空值
            raw_data = {
                "pe": latest_data.get('市盈率(TTM)', None),
                "pb": latest_data.get('市净率', latest_data.get('市净率(MRQ)', None)),
                "roe": latest_data.get('净资产收益率(摊薄)', None)
            }
            
            # 清理数据（处理NaN）
            clean_data = {k: (float(v) if v and pd.notna(v) else None) for k, v in raw_data.items()}
            
            return InfoResponse(**clean_data)
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            print(f"Error processing info request for {code}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal error fetching financial info: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")
