from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
import datetime
import json
import pandas as pd # 导入 pandas 用于数据处理

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using yfinance."
)

# --- Pydantic Models (保持不变) ---
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
    
    # 修正 pydantic v2 的配置方式
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "name": "Apple Inc.",
                "latestPrice": 150.0,
                "changePercent": 1.5,
                "changeAmount": 2.25,
                "source": "yfinance",
                "currency": "USD"
            }
        }


class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")
    source: str = Field(..., description="Data source (yfinance)")

# --- Helper Functions (保持不变) ---
def get_yfinance_ticker(code: str) -> str:
    """将股票代码转换为yfinance可识别的格式"""
    if code.upper().startswith('HK'):
        num_part = code[2:]
        if num_part.startswith('0') and len(num_part) > 1:
            num_part = num_part[1:]
        return f"{num_part}.HK"
    elif code.upper().startswith('US'):
        code_part = code[2:]
        return f"{code_part}"
    
    if code.startswith(('60', '68', '900')):
        return f"{code}.SS"
    elif code.startswith(('00', '30', '200')):
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87', '88')):
        return f"{code}.BJ"
    elif code.startswith(('58', '55', '51')):
        return f"{code}.SS"
    elif code.startswith(('15')):
        return f"{code}.SZ"
    else:
        return code

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if current_price is None:
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        if current_price is None:
            return None
        
        prev_close = info.get('previousClose')
        if prev_close is None:
            hist = ticker.history(period="2d")
            prev_close = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
        
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100 if prev_close != 0 else 0
        
        name = info.get('shortName', info.get('longName', code))
        currency = info.get('currency', 'USD')
        
        is_etf_or_us = code.upper().startswith(('58', '56', '51', '15', 'US'))
        
        daily_data = []
        if is_etf_or_us:
            data = ticker.history(period="5d").sort_index(ascending=False)
            if not data.empty:
                closes = data['Close'].tolist()
                for i in range(len(closes)):
                    prev_day_close = closes[i+1] if i + 1 < len(closes) else (data['Open'].iloc[-1] if 'Open' in data.columns else closes[-1])
                    daily_change = ((closes[i] - prev_day_close) / prev_day_close) * 100 if prev_day_close != 0 else 0.0
                    daily_data.append(DailyData(
                        date=data.index[i].strftime('%Y-%m-%d'),
                        change=f"{daily_change:.2f}",
                        price=closes[i]
                    ))

        return PriceResponse(
            name=name,
            latestPrice=current_price,
            changePercent=change_percent,
            changeAmount=change_amount,
            source="yfinance",
            currency=currency,
            dailydata=daily_data if is_etf_or_us else None
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
        
        pe = info.get('trailingPE')
        pb = info.get('priceToBook')
        roe = info.get('returnOnEquity')
        
        if roe is None and pe is not None and pb is not None and pe != 0:
            roe = (pb / pe) # yfinance 的 roe 是小数，所以这里不乘以100

        return InfoResponse(
            pe=pe,
            pb=pb,
            roe=roe,
            source="yfinance"
        )
    
    except Exception as e:
        print(f"yfinance error for financial info {code}: {str(e)}")
        return None

# --- API Endpoint (修改部分) ---
@app.get("/api/rtStockQueryProxy")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price', 'info', 'movingaveragedata', or 'intraday'")
):
    """
    Fetches stock data based on the code and query type using yfinance.
    """
    if query_type == 'price':
        response = fetch_price_with_yfinance(code)
        if response:
            return response
        raise HTTPException(status_code=404, detail=f"Price data not found for {code}")

    elif query_type == 'info':
        response = fetch_financial_info_with_yfinance(code)
        if response:
            return response
        raise HTTPException(status_code=404, detail=f"Financial info not found for {code}")

    elif query_type == 'movingaveragedata':
        try:
            ticker_symbol = get_yfinance_ticker(code)
            ticker = yf.Ticker(ticker_symbol)
            # 增加 back_adjust=True 来更好地处理分红配股对历史价格的影响
            hist_data = ticker.history(period="2y", auto_adjust=False, back_adjust=True)
            if hist_data.empty:
                raise HTTPException(status_code=404, detail="No historical data found")

            ma_periods = [5, 10, 20, 30, 60, 120, 250]
            for period in ma_periods:
                hist_data[f'MA_{period}'] = hist_data['Close'].rolling(window=period).mean()

            # 只返回最近一年的数据给前端，减轻前端负担
            plot_data = hist_data.tail(252).copy()
            json_output = plot_data.reset_index().to_json(orient='records', date_format='iso')
            
            return Response(content=json_output, media_type="application/json")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ==============================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>   新增的分支逻辑   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================
    elif query_type == 'intraday':
        try:
            ticker_symbol = get_yfinance_ticker(code)
            print(f"Fetching intraday (1m) data for {ticker_symbol}")
            ticker = yf.Ticker(ticker_symbol)

            # 获取最近一个交易日的1分钟数据
            # 注意: yfinance对1分钟数据的支持最长为7天，period="1d"是安全的
            # auto_adjust=False 确保我们能拿到 'Volume' 列
            intraday_data = ticker.history(period="1d", interval="1m", auto_adjust=False)

            if intraday_data.empty:
                raise HTTPException(status_code=404, detail=f"No intraday data found for {code}. It might be a non-trading day or the ticker is not supported for intraday data.")

            # --- 计算累计均价 (VWAP) ---
            # 1. 计算每分钟的 (价格 * 成交量)
            intraday_data['PriceVolume'] = intraday_data['Close'] * intraday_data['Volume']
            # 2. 计算累计成交量
            intraday_data['CumulativeVolume'] = intraday_data['Volume'].cumsum()
            # 3. 计算累计的 (价格 * 成交量)
            intraday_data['CumulativePriceVolume'] = intraday_data['PriceVolume'].cumsum()
            # 4. 计算均价 (VWAP)
            intraday_data['avg_price'] = intraday_data['CumulativePriceVolume'] / intraday_data['CumulativeVolume']

            # --- 准备返回给前端的数据 ---
            # 1. 格式化时间为 'HH:MM:SS'
            intraday_data['time'] = intraday_data.index.strftime('%H:%M:%S')
            
            # 2. 选择并重命名列以匹配前端的期望
            result_df = intraday_data[['time', 'Close', 'avg_price', 'Volume']].rename(columns={
                'Close': 'price',
                'Volume': 'volume'
            })
            
            # 3. 填充可能出现的 NaN 值（例如，如果某分钟成交量为0）
            result_df = result_df.fillna(method='ffill') # 使用前一个有效值填充
            
            # 4. 将 DataFrame 转换为 JSON 记录列表
            json_output = result_df.to_json(orient='records')
            
            return Response(content=json_output, media_type="application/json")

        except Exception as e:
            print(f"Error fetching intraday data for {code}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    # ==============================================================================
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>     新增逻辑结束     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==============================================================================

    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price', 'info', 'movingaveragedata', or 'intraday'.")
