from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional
import time

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using yfinance."
)

# --- Pydantic Models ---
class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')
    source: str = Field(..., description="Data source (yfinance)")
    currency: str = Field(..., description="Currency of the stock")
    class Config:
        validate_by_name = True  # 修正为正确配置项

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")
    source: str = Field(..., description="Data source (yfinance)")

# --- Helper Function ---
def get_yfinance_ticker(code: str) -> str:
    """将股票代码转换为yfinance可识别的格式"""
    # 港股处理（格式如: HK02899, hk00005, HK03690）
    if code.upper().startswith('HK'):
        # 提取数字部分，移除前导零，并添加 .HK 后缀
        num_part = code[2:].lstrip('0')  # 移除开头的所有零
        if not num_part:  # 如果所有数字都是零（如HK00000）
            num_part = "0"
        return f"{num_part}.HK"
    
    # A股处理
    if code.startswith(('60', '68', '900')):  # 沪市
        return f"{code}.SS"
    elif code.startswith(('00', '30', '200')):  # 深市
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87', '88')):  # 北交所
        return f"{code}.BJ"
    else:  # 美股及其他市场
        # 美股处理（如 AAPL, MSFT）
        return code

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        
        # 等待一小段时间确保数据加载
        time.sleep(0.2)
        
        # 获取基本信息
        info = ticker.info
        
        # 优先使用currentPrice获取实时价格
        current_price = info.get('currentPrice')
        
        # 如果currentPrice不可用，尝试使用regularMarketPrice
        if current_price is None:
            current_price = info.get('regularMarketPrice')
        
        # 如果仍然不可用，尝试从历史数据中获取最新价格
        if current_price is None:
            print("Falling back to historical data for current price")
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        # 如果所有方法都无法获取价格，返回None
        if current_price is None:
            print(f"No price data available from yfinance for {ticker_symbol}")
            return None
        
        # 获取前一天收盘价
        prev_close = info.get('previousClose')
        
        # 如果无法获取前一天收盘价，尝试从历史数据中提取
        if prev_close is None:
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
            else:
                # 如果没有历史数据，使用当前价格作为前收盘价
                prev_close = current_price
        
        # 计算涨跌额和涨跌幅
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100
        
        # 获取股票名称和货币
        name = info.get('shortName', info.get('longName', code))
        currency = info.get('currency', 'USD')
        
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

def fetch_financial_info_with_yfinance(code: str) -> Optional[InfoResponse]:
    """使用yfinance获取金融信息"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching financial info with yfinance for {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 获取市盈率、市净率和ROE
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        
        # ROE转换为百分比形式（如果存在）
        if roe is not None:
            roe = roe   # 转换为百分比
        elif pe is not None and pb is not None and pe != 0:
            roe = (pb / pe) * 100  # ROE = PB/PE * 100
            print(f"Using estimated ROE: {roe}")

        return InfoResponse(
            pe=pe,
            pb=pb,
            roe=roe,
            source="yfinance"
        )
    
    except Exception as e:
        print(f"yfinance error for financial info {code}: {str(e)}")
        return None

# --- API Endpoint ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price' or 'info'")
):
    """
    Fetches stock data based on the code and query type using yfinance.
    """
    if query_type == 'price':
        response = fetch_price_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Price data not found for {code}"
            )

    elif query_type == 'info':
        response = fetch_financial_info_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Financial info not found for {code}"
            )
            
    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price' or 'info'.")
