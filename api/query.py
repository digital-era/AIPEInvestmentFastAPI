from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import akshare as ak
import pandas as pd
import yfinance as yf
from typing import Optional
import time

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using yfinance and akshare."
)

# --- Pydantic Models ---
class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')
    source: str = Field(..., description="Data source (yfinance or akshare)")
    currency: str = Field(..., description="Currency of the stock")
    class Config:
        validate_by_name = True

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")
    source: str = Field(..., description="Data source (akshare)")

# --- Helper Function ---
def get_stock_market_code(code: str) -> str:
    """获取带市场前缀的股票代码，更全面的覆盖"""
    if code.startswith(('60', '900')):
        return f"sh{code}"
    elif code.startswith('68'):  # 科创板
        return f"sh{code}"
    elif code.startswith(('00', '30', '200')):
        return f"sz{code}"
    elif code.startswith(('43', '83', '87', '88')):  # 北交所
        return f"bj{code}"
    else:
        # 对于无法识别的格式，直接抛出 ValueError
        raise ValueError(f"Invalid or unsupported stock code format: '{code}'")

def get_yfinance_ticker(code: str) -> str:
    """将A股代码转换为yfinance可识别的格式"""
    if code.startswith(('60', '68', '900')):
        return f"{code}.SS"  # 沪市
    elif code.startswith(('00', '30', '200')):
        return f"{code}.SZ"  # 深市
    elif code.startswith(('43', '83', '87', '88')):
        return f"{code}.BJ"  # 北交所
    else:
        # 对于非A股股票（如美股），直接返回代码
        return code

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据（优先使用currentPrice）"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        ticker = yf.Ticker(ticker_symbol)
        
        # 等待一小段时间确保数据加载
        time.sleep(0.2)
        
        # 优先使用currentPrice获取实时价格
        current_price = ticker.info.get('currentPrice')
        
        # 如果currentPrice不可用，尝试使用regularMarketPrice
        if current_price is None:
            current_price = ticker.info.get('regularMarketPrice')
        
        # 如果仍然不可用，尝试从历史数据中获取最新价格
        if current_price is None:
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        # 如果所有方法都无法获取价格，返回None
        if current_price is None:
            print(f"No price data available from yfinance for {ticker_symbol}")
            return None
        
        # 获取前一天收盘价
        prev_close = ticker.info.get('previousClose')
        
        # 如果无法获取前一天收盘价，尝试从历史数据中提取
        if prev_close is None:
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
            else:
                # 如果没有历史数据，使用当前价格作为前收盘价（会导致涨跌幅为0）
                prev_close = current_price
        
        # 计算涨跌额和涨跌幅
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100
        
        # 获取股票名称和货币
        name = ticker.info.get('shortName', code)
        if name is None:
            name = ticker.info.get('longName', code)
        
        currency = ticker.info.get('currency', 'USD')
        
        return PriceResponse(
            name=name,
            latestPrice=current_price,
            changePercent=change_percent,
            changeAmount=change_amount,
            source="yfinance",
            currency=currency
        )
    
    except Exception as e:
        print(f"yfinance error for {code}: {str(e)}")
        return None

def fetch_price_with_akshare(code: str, market_code: str) -> PriceResponse:
    """使用akshare获取股票实时价格数据"""
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
            "changePercent": get_value('涨幅'),  # 直接使用，它本身就是 float
            "changeAmount": get_value('涨跌'),
            "source": "akshare",
            "currency": "CNY"  # 假设akshare返回的都是人民币计价的A股数据
        }
        return PriceResponse(**data)
        
    except Exception as e:
        print(f"akshare error for {code}: {str(e)}")
        raise HTTPException(
            status_code=502, 
            detail=f"Failed to fetch data from akshare: {str(e)}"
        )

# --- API Endpoint ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price' or 'info'")
):
    """
    Fetches stock data based on the code and query type.
    """
    if query_type == 'price':
        # 先尝试使用yfinance
        yfinance_response = fetch_price_with_yfinance(code)
        if yfinance_response:
            return yfinance_response
        
        # 如果yfinance失败，则尝试使用akshare（仅适用于A股）
        print(f"Falling back to akshare for {code}")
        try:
            market_code = get_stock_market_code(code)
            return fetch_price_with_akshare(code, market_code)
        except ValueError as e:
            # 如果股票代码格式无效，返回 400 Bad Request
            raise HTTPException(status_code=400, detail=str(e))
        except HTTPException:
            raise  # 直接重新抛出已处理的HTTPException
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch price data from all sources: {str(e)}"
            )

    elif query_type == 'info':
        # 金融信息仍使用akshare（仅适用于A股）
        try:
            market_code = get_stock_market_code(code)
            
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
            except Exception as e:
                print(f"ROE extraction failed: {str(e)}")
                # 如果获取失败，尝试使用PB和PE估算ROE
                if pe is not None and pb is not None and pe != 0:
                    roe = (pb / pe) * 100  # ROE = PB/PE * 100
                    print(f"Using estimated ROE: {roe}")
            
            return InfoResponse(pe=pe, pb=pb, roe=roe, source="akshare")
            
        except Exception as e:
            print(f"Info query error for {code}: {str(e)}")
            raise HTTPException(
                status_code=502, 
                detail=f"Financial data query failed: {str(e)}"
            )
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")

import asyncio
# 获取当前事件循环
loop = asyncio.get_event_loop()

# 创建任务并等待完成
task = loop.create_task(get_stock_data("600900", "price"))
rsp = await task  # 在 Jupyter cell 中使用 "await"

# 打印结果
print("直接打印:", rsp)
print("\n字典形式:", rsp.model_dump(by_alias=True))

# 打印所有字段
print("\n所有字段:")
for field, value in rsp.model_dump(by_alias=True).items():
    print(f"{field}: {value}")
