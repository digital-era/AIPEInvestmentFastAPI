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
            # 方案1：使用实时接口获取PE/PB
            spot_df = ak.stock_individual_spot_xq(symbol=market_code)
            
            # 辅助函数提取实时指标
            def get_spot_value(item_name: str):
                try:
                    value = spot_df.loc[spot_df['item'] == item_name, 'value'].iloc[0]
                    return float(value) if value != '--' else None
                except (IndexError, ValueError, TypeError):
                    return None
            
            # 获取实时PE/PB
            pe = get_spot_value('市盈率(TTM)')
            pb = get_spot_value('市净率')
            
            # 方案2：获取最新年报ROE
            roe = None
            try:
                # 使用基本面接口获取ROE
                indicator_df = ak.stock_financial_analysis_indicator(symbol=code)
                if not indicator_df.empty:
                    # 按报告期排序并取最新年报
                    indicator_df = indicator_df.sort_values('报告日期', ascending=False)
                    roe_row = indicator_df[indicator_df['报告日期'].str.contains('1231')].iloc[0]
                    roe = roe_row['净资产收益率']
                    roe_source = "年报"
            except Exception as e:
                # 如果获取失败，尝试使用PB和PE估算ROE
                if pe is not None and pb is not None and pe > 0:
                    roe = pb  / pe  # 使用公式 ROE = PB × 100 / PE
                    roe_source = "估算"
                    print(f"使用估算方法获取ROE: PB={pb}, PE={pe} => ROE={roe}")
                else:
                    print(f"无法估算ROE: PB={pb}, PE={pe}")
            
            # 如果年报获取成功但值为空，也尝试估算
            ##if roe is None and pe is not None and pb is not None and pe > 0:
            if roe is None and pe is not None and pb is not None:
                roe = pb  / pe
                roe_source = "估算"
                print(f"年报ROE为空，使用估算方法: PB={pb}, PE={pe} => ROE={roe}")
            
            return InfoResponse(pe=pe, pb=pb, roe=roe, roe_source=roe_source)
            
        except Exception as e:
            print(f"Info query error for {code}: {str(e)}")
            raise HTTPException(
                status_code=502, 
                detail=f"Financial data query failed: {str(e)}"
            )
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")
