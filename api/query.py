from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import traceback

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using tushare with fallback to yfinance."
)

# Tushare配置 - 请在这里填入你的token
TUSHARE_TOKEN = "18592c39e9b5e8319cefadf056b3fc8d87c83579c7ca375f26de087c"  # 请替换为实际的token

# 由于在serverless环境中tushare可能无法正常工作，我们将主要依赖yfinance
# 但我们可以尝试使用tushare的HTTP API作为替代方案
TUSHARE_ENABLED = True  # 可以根据需要禁用tushare

# --- Pydantic Models (完全保持原始状态) ---
class DailyData(BaseModel):
    date: str
    change: str
    price: float

class PriceResponse(BaseModel):
    name: str
    latest_price: float = Field(..., alias='latestPrice')
    change_percent: float = Field(..., alias='changePercent')
    change_amount: float = Field(..., alias='changeAmount')
    source: str = Field(..., description="Data source (yfinance/tushare)")
    currency: str = Field(..., description="Currency of the stock")
    dailydata: Optional[List[DailyData]] = Field(None, description="Recent 5 trading days data for ETFs")
    class Config:
        validate_by_name = True

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")
    source: str = Field(..., description="Data source (yfinance)")

# --- Helper Functions ---
def get_yfinance_ticker(code: str) -> str:
    """将股票代码转换为yfinance可识别的格式"""
    # 港股处理（格式如: HK02899, hk00005, HK03690）
    if code.upper().startswith('HK'):
        num_part = code[2:]
        if num_part.startswith('0') and len(num_part) > 1:
            num_part = num_part[1:]
        return f"{num_part}.HK"
    elif code.upper().startswith('US'):  # 美股
        code_part = code[2:]
        return f"{code_part}"
    
    # A股处理
    if code.startswith(('60', '68', '900')):  # 沪市
        return f"{code}.SS"
    elif code.startswith(('00', '30', '200')):  # 深市
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87', '88')):  # 北交所
        return f"{code}.BJ"
    elif code.startswith(('58', '56','55', '51')):  # 上证ETF
        return f"{code}.SS"
    elif code.startswith(('15')):  # 深证ETF
        return f"{code}.SZ"
    else:
        return code

# --- 简化的Tushare HTTP API Functions ---
def fetch_price_with_tushare(code: str) -> Optional[PriceResponse]:
    """使用tushare HTTP API获取股票实时价格数据"""
    if not TUSHARE_ENABLED:
        return None
        
    try:
        import requests
        
        # 构建tushare代码
        if code.upper().startswith('HK'):
            num_part = code[2:]
            tushare_code = f"{num_part.zfill(5)}.HK"
        elif code.upper().startswith('US'):
            tushare_code = code[2:]
        elif code.startswith(('60', '68', '900')):
            tushare_code = f"{code}.SH"
        elif code.startswith(('00', '30', '200')):
            tushare_code = f"{code}.SZ"
        elif code.startswith(('43', '83', '87', '88')):
            tushare_code = f"{code}.BJ"
        elif code.startswith(('58', '56', '55', '51')):
            tushare_code = f"{code}.SH"
        elif code.startswith('15'):
            tushare_code = f"{code}.SZ"
        else:
            tushare_code = code
            
        print(f"Fetching price data with tushare HTTP API for {tushare_code}")
        
        # 使用tushare的HTTP API
        url = "http://api.tushare.pro"
        
        # 获取股票基本信息
        data = {
            "api_name": "stock_basic",
            "token": TUSHARE_TOKEN,
            "params": {"ts_code": tushare_code},
            "fields": "ts_code,name"
        }
        
        response = requests.post(url, json=data, timeout=5)
        if response.status_code != 200:
            print(f"Tushare HTTP API error: {response.status_code}")
            return None
            
        result = response.json()
        if result.get('code') != 0 or not result.get('data', {}).get('items'):
            print(f"No basic info from tushare for {tushare_code}")
            return None
            
        items = result['data']['items']
        name = items[0][1] if items else code
        
        # 获取日线数据
        today = datetime.now().strftime('%Y%m%d')
        data = {
            "api_name": "daily",
            "token": TUSHARE_TOKEN,
            "params": {"ts_code": tushare_code, "start_date": today, "end_date": today},
            "fields": "trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
        }
        
        response = requests.post(url, json=data, timeout=5)
        if response.status_code != 200:
            print(f"Tushare daily data error: {response.status_code}")
            return None
            
        result = response.json()
        if result.get('code') != 0 or not result.get('data', {}).get('items'):
            # 尝试获取最近的数据
            data["params"] = {"ts_code": tushare_code, "limit": 2}
            response = requests.post(url, json=data, timeout=5)
            if response.status_code != 200:
                return None
                
            result = response.json()
            if result.get('code') != 0 or not result.get('data', {}).get('items'):
                print(f"No daily data from tushare for {tushare_code}")
                return None
        
        items = result['data']['items']
        if not items:
            return None
            
        # 解析数据
        latest_item = items[0]
        fields = result['data']['fields']
        
        # 创建字段映射
        field_map = {}
        for i, field in enumerate(fields):
            field_map[field] = latest_item[i] if i < len(latest_item) else None
        
        current_price = field_map.get('close')
        prev_close = field_map.get('pre_close')
        
        if current_price is None or prev_close is None:
            print(f"Incomplete data from tushare for {tushare_code}")
            return None
        
        # 计算涨跌幅和涨跌额
        change_percent = ((current_price - prev_close) / prev_close) * 100 if prev_close else 0
        change_amount = current_price - prev_close if prev_close else 0
        
        # 获取日线数据用于ETF/US股票
        is_etf = (
            code.upper().startswith('58') or 
            code.upper().startswith('56') or 
            code.upper().startswith('55') or 
            code.upper().startswith('51') or 
            code.upper().startswith('15')
        )
        is_us = code.upper().startswith('US')
        
        daily_data = []
        if is_etf or is_us:
            # 获取最近5个交易日数据
            data = {
                "api_name": "daily",
                "token": TUSHARE_TOKEN,
                "params": {"ts_code": tushare_code, "limit": 5},
                "fields": "trade_date,close"
            }
            
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                result = response.json()
                if result.get('code') == 0 and result.get('data', {}).get('items'):
                    history_items = result['data']['items']
                    history_fields = result['data']['fields']
                    
                    for i, item in enumerate(history_items):
                        item_dict = dict(zip(history_fields, item))
                        close_price = item_dict.get('close')
                        trade_date = item_dict.get('trade_date')
                        
                        if close_price and trade_date:
                            if i == 0:
                                daily_change = change_percent
                            elif i == 1:
                                prev_close_2 = history_items[1][history_fields.index('close')] if len(history_items) > 1 else close_price
                                daily_change = ((close_price - prev_close_2) / prev_close_2) * 100 if prev_close_2 else 0
                            else:
                                daily_change = 0
                            
                            daily_data.append({
                                "date": trade_date,
                                "change": f"{daily_change:.2f}",
                                "price": close_price
                            })
        
        # 处理ETF名称映射
        if is_etf:
            file_path = "data/etf_name_data.json"    
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name_map = {str(item['code']): item['name'] for item in data}
                    predefined_name = name_map.get(code)
                    if predefined_name:
                        name = predefined_name
                except Exception as e:
                    print(f"Error loading name map: {e}")
        
        # 确定货币
        if code.upper().startswith('HK'):
            currency = "HKD"
        elif code.upper().startswith('US'):
            currency = "USD"
        else:
            currency = "CNY"
        
        return PriceResponse(
            name=name,
            latestPrice=current_price,
            changePercent=change_percent,
            changeAmount=change_amount,
            source="tushare",
            currency=currency,
            dailydata=daily_data if is_etf or is_us else None
        )
    
    except Exception as e:
        print(f"Tushare HTTP API error for {code}: {e}")
        traceback.print_exc()
        return None

def fetch_intraday_with_tushare(code: str) -> Optional[Response]:
    """使用tushare HTTP API获取分时数据"""
    if not TUSHARE_ENABLED:
        return None
        
    try:
        import requests
        
        # 构建tushare代码
        if code.upper().startswith('HK'):
            num_part = code[2:]
            tushare_code = f"{num_part.zfill(5)}.HK"
        elif code.upper().startswith('US'):
            tushare_code = code[2:]
        elif code.startswith(('60', '68', '900')):
            tushare_code = f"{code}.SH"
        elif code.startswith(('00', '30', '200')):
            tushare_code = f"{code}.SZ"
        elif code.startswith(('43', '83', '87', '88')):
            tushare_code = f"{code}.BJ"
        elif code.startswith(('58', '56', '55', '51')):
            tushare_code = f"{code}.SH"
        elif code.startswith('15'):
            tushare_code = f"{code}.SZ"
        else:
            tushare_code = code
            
        print(f"Fetching intraday data with tushare HTTP API for {tushare_code}")
        
        # 使用tushare的HTTP API
        url = "http://api.tushare.pro"
        
        # 尝试获取分时数据
        today = datetime.now().strftime('%Y%m%d')
        data = {
            "api_name": "realtime_quote",
            "token": TUSHARE_TOKEN,
            "params": {"ts_code": tushare_code},
            "fields": "ts_code,trade_time,price,change,volume"
        }
        
        response = requests.post(url, json=data, timeout=5)
        if response.status_code != 200:
            print(f"Tushare realtime_quote error: {response.status_code}")
            return None
            
        result = response.json()
        if result.get('code') != 0 or not result.get('data', {}).get('items'):
            print(f"No realtime data from tushare for {tushare_code}")
            return None
        
        items = result['data']['items']
        fields = result['data']['fields']
        
        if not items:
            return None
        
        # 处理分时数据
        intraday_records = []
        cumulative_volume = 0
        cumulative_price_volume = 0
        
        for item in items:
            item_dict = dict(zip(fields, item))
            trade_time = item_dict.get('trade_time')
            price = item_dict.get('price')
            volume = item_dict.get('volume', 0)
            
            if trade_time and price is not None:
                # 计算累计均价 (VWAP)
                price_volume = price * volume
                cumulative_volume += volume
                cumulative_price_volume += price_volume
                
                avg_price = cumulative_price_volume / cumulative_volume if cumulative_volume > 0 else price
                
                # 解析时间
                try:
                    # tushare的时间格式可能是 "09:30:00" 或 "2024-01-20 09:30:00"
                    if ' ' in str(trade_time):
                        trade_datetime = datetime.strptime(str(trade_time), '%Y-%m-%d %H:%M:%S')
                        date_str = trade_datetime.strftime('%Y-%m-%d')
                        time_str = trade_datetime.strftime('%H:%M:%S')
                    else:
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        time_str = str(trade_time)
                except:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    time_str = str(trade_time)
                
                intraday_records.append({
                    'date': date_str,
                    'time': time_str,
                    'price': price,
                    'avg_price': avg_price,
                    'volume': volume
                })
        
        if not intraday_records:
            return None
        
        # 转换为DataFrame并填充NaN值
        result_df = pd.DataFrame(intraday_records)
        result_df = result_df.fillna(method='ffill')
        
        json_output = result_df.to_json(orient='records')
        
        return Response(content=json_output, media_type="application/json")
    
    except Exception as e:
        print(f"Tushare intraday HTTP API error for {code}: {e}")
        traceback.print_exc()
        return None

# --- 原有的yfinance函数 ---
def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据（回落机制）"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol} (fallback)")
        ticker = yf.Ticker(ticker_symbol)
        
        # 尝试快速获取数据
        current_price = None
        try:
            current_price = ticker.fast_info.get('last_price')
            fast_prev_close = ticker.fast_info.get('previous_close')
        except:
            fast_prev_close = None
        
        time.sleep(0.1)
        
        info = ticker.info

        if current_price is None:
            current_price = info.get('currentPrice')
        
        if current_price is None:
            current_price = info.get('regularMarketPrice')

        if current_price is None:
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        if current_price is None:
            print(f"No price data available from yfinance for {ticker_symbol}")
            return None
        
        prev_close = fast_prev_close if fast_prev_close is not None else info.get('previousClose')
        
        if prev_close is None:
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
            else:
                prev_close = current_price
        
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100 if prev_close else 0
        
        name = info.get('shortName', info.get('longName', code))
        currency = info.get('currency', 'USD')
        
        is_etf = (
            code.upper().startswith('58') or 
            code.upper().startswith('56') or 
            code.upper().startswith('55') or 
            code.upper().startswith('51') or 
            code.upper().startswith('15')
        )
        is_us = code.upper().startswith('US')
        
        daily_data = []
        if is_etf or is_us:
            data = ticker.history(period="5d")
            
            if not data.empty:
                data = data.sort_index(ascending=False)
                
                for i, (date, row) in enumerate(data.iterrows()):
                    if i == 0:
                        daily_change = ((row['Close'] - prev_close) / prev_close) * 100 if prev_close else 0
                    elif i < len(data) - 1:
                        daily_change = ((row['Close'] - data.iloc[i+1]['Close']) / data.iloc[i+1]['Close']) * 100 if data.iloc[i+1]['Close'] else 0
                    else:
                        daily_change = 0
                    
                    date_str = date.strftime('%Y-%m-%d')
                    daily_data.append({
                        "date": date_str,
                        "change": f"{daily_change:.2f}",
                        "price": row['Close']
                    })

        if is_etf:
            file_path = "data/etf_name_data.json"    
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name_map = {str(item['code']): item['name'] for item in data}
                    predefined_name = name_map.get(code)
                    if predefined_name:
                        name = predefined_name
                except Exception as e:
                    print(f"Error loading name map: {e}")
        
        return PriceResponse(
            name=name,
            latestPrice=current_price,
            changePercent=change_percent,
            changeAmount=change_amount,
            source="yfinance",
            currency=currency,
            dailydata=daily_data if is_etf or is_us else None
        )
    
    except Exception as e:
        print(f"Yfinance error for {code}: {e}")
        traceback.print_exc()
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
        
        if roe is not None:
            roe = roe
        elif pe is not None and pb is not None and pe != 0:
            roe = (pb / pe) * 100
            print(f"Using estimated ROE: {roe}")

        return InfoResponse(
            pe=pe,
            pb=pb,
            roe=roe,
            source="yfinance"
        )
    
    except Exception as e:
        print(f"Yfinance error for financial info {code}: {e}")
        traceback.print_exc()
        return None

# --- API Endpoint ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price', 'info', 'movingaveragedata', or 'intraday'")
):
    """
    Fetches stock data based on the code and query type using tushare with fallback to yfinance.
    """
    print(f"Received request: code={code}, type={query_type}")
    
    try:
        if query_type == 'price':
            # 先尝试tushare HTTP API
            response = fetch_price_with_tushare(code)
            if response:
                return response
            
            print(f"Tushare failed, falling back to yfinance for {code}")
            response = fetch_price_with_yfinance(code)
            if response:
                return response
            
            raise HTTPException(
                status_code=404, 
                detail=f"Price data not found for {code}"
            )

        elif query_type == 'info':
            response = fetch_financial_info_with_yfinance(code)
            if response:
                return response
            
            raise HTTPException(
                status_code=404, 
                detail=f"Financial info not found for {code}"
            )

        elif query_type == 'movingaveragedata':
            try:
                ticker_symbol = get_yfinance_ticker(code)
                ticker = yf.Ticker(ticker_symbol)
                hist_data = ticker.history(period="2y", auto_adjust=False, back_adjust=True)
                
                if hist_data.empty:
                    raise HTTPException(status_code=404, detail="No historical data found")

                ma_periods = [5, 10, 20, 30, 60, 120, 250]
                for period in ma_periods:
                    hist_data[f'MA_{period}'] = hist_data['Close'].rolling(window=period).mean()

                plot_data = hist_data.tail(252).copy()
                json_output = plot_data.reset_index().to_json(orient='records', date_format='iso')
                
                return Response(content=json_output, media_type="application/json")
                
            except HTTPException:
                raise
            except Exception as e:
                print(f"Error in movingaveragedata: {e}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

        elif query_type == 'intraday':
            # 先尝试tushare HTTP API
            tushare_response = fetch_intraday_with_tushare(code)
            if tushare_response:
                return tushare_response
            
            print(f"Tushare intraday failed, falling back to yfinance for {code}")
            
            try:
                ticker_symbol = get_yfinance_ticker(code)
                ticker = yf.Ticker(ticker_symbol)

                intraday_data = ticker.history(period="1d", interval="1m", auto_adjust=False)

                if intraday_data.empty:
                    raise HTTPException(
                        status_code=404, 
                        detail=f"No intraday data found for {code}. It might be a non-trading day."
                    )

                intraday_data['PriceVolume'] = intraday_data['Close'] * intraday_data['Volume']
                intraday_data['CumulativeVolume'] = intraday_data['Volume'].cumsum()
                intraday_data['CumulativePriceVolume'] = intraday_data['PriceVolume'].cumsum()
                intraday_data['avg_price'] = intraday_data['CumulativePriceVolume'] / intraday_data['CumulativeVolume']

                intraday_data['date'] = intraday_data.index.strftime('%Y-%m-%d')
                intraday_data['time'] = intraday_data.index.strftime('%H:%M:%S')
                
                result_df = intraday_data[['date', 'time', 'Close', 'avg_price', 'Volume']].rename(columns={
                    'Close': 'price',
                    'Volume': 'volume'
                })
                
                result_df = result_df.fillna(method='ffill')
                json_output = result_df.to_json(orient='records')
                
                return Response(content=json_output, media_type="application/json")

            except HTTPException:
                raise
            except Exception as e:
                print(f"Yfinance intraday error: {e}")
                traceback.print_exc()
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error fetching intraday data: {str(e)}"
                )

        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid 'type' parameter. Use 'price', 'info', 'movingaveragedata', or 'intraday'."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in API endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 根路径端点
@app.get("/")
async def root():
    return {"message": "Stock Query API is running", "timestamp": datetime.now().isoformat()}
