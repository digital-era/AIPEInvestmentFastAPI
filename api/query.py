from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import akshare as ak
import pandas as pd

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using akshare."
)

# --- Pydantic Models ---
class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')
    class Config:
        allow_population_by_field_name = True # 允许使用 'latestPrice' 等别名填充

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")

# --- Helper Function ---
def get_stock_market_code(code: str) -> str:
    """获取带市场前缀的股票代码，更全面的覆盖"""
    if code.startswith(('60', '900')):
        return f"sh{code}"
    elif code.startswith('68'):  # 科创板
        return f"sh{code}"
    elif code.startswith(('00', '30', '200')):
        return f"sz{code}"
    elif code.startswith(('43', '8')):  # 北交所
        return f"bj{code}"
    else:
        # 对于无法识别的格式，直接抛出 ValueError
        raise ValueError(f"Invalid or unsupported stock code format: '{code}'")

# --- API Endpoint ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price' or 'info'")
):
    """
    Fetches stock data based on the code and query type.
    """
    try:
        market_code = get_stock_market_code(code)
    except ValueError as e:
        # 如果股票代码格式无效，返回 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e))

    if query_type == 'price':
        try:
            # 使用雪球接口获取数据
            df = ak.stock_individual_spot_xq(symbol=market_code)
            
            # 使用一个辅助函数来安全地提取和转换数据
            def get_value(item_name: str):
                try:
                    # 获取原始值
                    value = df.loc[df['item'] == item_name, 'value'].iloc[0]
                    # 尝试转换为 float，如果已经是 float 则直接返回
                    return float(value)
                except (IndexError, ValueError):
                    # 如果找不到 item 或者值无法转换为 float (例如 '--')
                    raise ValueError(f"Could not retrieve or parse '{item_name}' for code {code}")

            # 安全地构建数据字典
            data = {
                "name": df.loc[df['item'] == '名称', 'value'].iloc[0],
                "latestPrice": get_value('现价'),
                "changePercent": get_value('涨幅'), # 直接使用，它本身就是 float
                "changeAmount": get_value('涨跌')
            }
            return PriceResponse(**data)
            
        except ValueError as ve:
             # 捕获 get_value 抛出的特定错误，通常表示数据不存在或格式错误
            raise HTTPException(status_code=404, detail=str(ve))
        except Exception as e:
            # 捕获 akshare 的其他网络或API错误
            print(f"Error processing price request for {code}: {e}")
            raise HTTPException(status_code=502, detail=f"Failed to fetch data from upstream source: {e}")

    elif query_type == 'info':
        try:
            financial_df = ak.stock_financial_analysis_indicator(symbol=market_code)
            
            if financial_df.empty:
                raise HTTPException(status_code=404, detail=f"Financial data for stock code '{code}' not found.")

            latest_data = financial_df.iloc[0]
            
            # 准备数据，使用 .get() 方法安全访问，以防列名变动
            raw_data = {
                "pe": latest_data.get('市盈率(TTM)'),
                "pb": latest_data.get('市净率', latest_data.get('市净率(MRQ)')),
                "roe": latest_data.get('净资产收益率(摊薄)')
            }
            
            # 清理数据（处理 pandas 的 <NA> 或 numpy 的 NaN）
            clean_data = {k: (float(v) if pd.notna(v) else None) for k, v in raw_data.items()}
            
            return InfoResponse(**clean_data)
            
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e # 如果是已经处理过的 HTTPException，直接重新抛出
            print(f"Error processing info request for {code}: {e}")
            raise HTTPException(status_code=502, detail=f"Internal error fetching financial info: {str(e)}")
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")
