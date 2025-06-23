from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
import datetime

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using yfinance."
)

# --- Pydantic Models ---
class DailyData(BaseModel):
    date: str
    change: str
    price: float

class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')
    source: str = Field(..., description="Data source (yfinance)")
    currency: str = Field(..., description="Currency of the stock")
    dailydata: Optional[List[DailyData]] = Field(None, description="Recent 5 trading days data for ETFs")
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
        # 提取数字部分，最多只移除开头的1个零
        num_part = code[2:]
        if num_part.startswith('0') and len(num_part) > 1:
            num_part = num_part[1:]  # 只移除开头的第一个零
        return f"{num_part}.HK"
    elif code.upper().startswith('ETF'):  # ETF
        # 提取数字部分，最多只移除开头的ETF
        num_part = code[3:]
        if num_part.startswith('5') and len(num_part) == 6:
            return f"{num_part}.SS"
        elif num_part.startswith('15') and len(num_part) == 6:
            return f"{num_part}.SZ"
    
    # A股处理
    if code.startswith(('60', '68', '900')):  # 沪市
        return f"{code}.SS"
    elif code.startswith(('00', '30', '200')):  # 深市
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87', '88')):  # 北交所
        return f"{code}.BJ"
    elif code.startswith(('58', '55', '51')):  # 上证ETF
        return f"{code}.SS"
    elif code.startswith(('15')):  # 深证ETF
        return f"{code}.SZ"
    else:  # 美股及其他市场
        # 美股处理（如 AAPL, MSFT）
        return code

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        current_price = None
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
            print("prev_close is None")
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
        
        # 检查是否为ETF
        is_etf = (
            code.upper().startswith('58') or 
            code.upper().startswith('56') or 
            code.upper().startswith('51') or 
            code.upper().startswith('15')
        )
        
        daily_data = []
        if is_etf:
            # 获取最近5个交易日的ETF数据
            print(f"Fetching 5-day ETF data for {ticker_symbol}")
            data = ticker.history(period="5d")
            
            if not data.empty:
                # 按日期降序排序（最新日期在前）
                data = data.sort_index(ascending=False)
                
                # 准备每日数据
                for i, (date, row) in enumerate(data.iterrows()):
                    # 计算涨跌额（与前一天比较）
                    if i == 0:  # 最新一天
                        daily_change = row['Close'] - prev_close
                    elif i < len(data) - 1:  # 中间日期
                        daily_change = row['Close'] - data.iloc[i+1]['Close']
                    else:  # 最早一天
                        daily_change = 0.0
                    
                    date_str = date.strftime('%Y-%m-%d')
                    daily_data.append({
                        "date": date_str,
                        "change": f"{daily_change:.2f}",
                        "price": row['Close']
                    })
        
        return PriceResponse(
            name=name,
            latestPrice=current_price,
            changePercent=change_percent,
            changeAmount=change_amount,
            source="yfinance",
            currency=currency,
            dailydata=daily_data if is_etf else None
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
