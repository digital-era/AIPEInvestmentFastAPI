from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
import datetime
import json
import pandas as pd
import os
import tushare as ts
from datetime import datetime, timedelta
import numpy as np

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using tushare with fallback to yfinance."
)

# Tushare配置 - 请在这里填入你的token
TUSHARE_TOKEN = "请在这里填入你的tushare_token"  # 请替换为实际的token

# 初始化tushare pro接口
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

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
def get_tushare_code(code: str) -> str:
    """将股票代码转换为tushare可识别的格式"""
    # 港股处理
    if code.upper().startswith('HK'):
        # 提取数字部分，移除开头的HK
        num_part = code[2:]
        # 港股代码格式：如02899.HK
        return f"{num_part.zfill(5)}.HK"
    elif code.upper().startswith('US'):  # 美股
        # 提取数字部分，移除开头的US
        code_part = code[2:]
        return f"{code_part}"
    
    # A股处理
    if code.startswith(('60', '68', '900')):  # 沪市
        return f"{code}.SH"
    elif code.startswith(('00', '30', '200')):  # 深市
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87', '88')):  # 北交所
        return f"{code}.BJ"
    elif code.startswith(('58', '56', '55', '51')):  # 上证ETF
        return f"{code}.SH"
    elif code.startswith(('15')):  # 深证ETF
        return f"{code}.SZ"
    else:  # 其他市场
        return code

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
    else:  # 其他市场
        return code

# --- Tushare API Functions (使用Python接口) ---
def fetch_price_with_tushare(code: str) -> Optional[PriceResponse]:
    """使用tushare获取股票实时价格数据"""
    try:
        tushare_code = get_tushare_code(code)
        print(f"Fetching price data with tushare for {tushare_code}")
        
        # 获取实时行情数据
        # 首先尝试获取股票基本信息
        df_basic = pro.stock_basic(ts_code=tushare_code)
        if df_basic.empty:
            print(f"No basic info found in tushare for {tushare_code}")
            return None
        
        name = df_basic.iloc[0]['name']
        
        # 获取实时行情
        # 注意：实时行情需要积分，这里使用日线数据作为替代
        today = datetime.now().strftime('%Y%m%d')
        
        # 获取当日和前一日的数据
        df_daily = pro.daily(ts_code=tushare_code, start_date=today, end_date=today)
        
        if df_daily.empty:
            # 如果当日无数据，可能是非交易日，获取最近交易日的数据
            df_daily = pro.daily(ts_code=tushare_code, limit=2)
            if df_daily.empty:
                print(f"No daily data found in tushare for {tushare_code}")
                return None
        
        # 获取最新数据
        latest_data = df_daily.iloc[0]
        current_price = latest_data['close']
        
        # 获取前收盘价
        if len(df_daily) > 1:
            prev_close = df_daily.iloc[1]['close']
        else:
            # 如果只有一天数据，使用当日开盘价作为参考
            prev_close = latest_data['open']
        
        # 计算涨跌幅和涨跌额
        change_percent = ((current_price - prev_close) / prev_close) * 100
        change_amount = current_price - prev_close
        
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
            # 获取最近5个交易日的ETF数据
            df_history = pro.daily(ts_code=tushare_code, limit=5)
            
            if not df_history.empty:
                # 按日期降序排序（最新日期在前）
                df_history = df_history.sort_values('trade_date', ascending=False)
                
                for i, row in df_history.iterrows():
                    close_price = row['close']
                    trade_date = row['trade_date']
                    
                    # 计算涨跌幅
                    if i == 0:  # 最新一天
                        daily_change = change_percent
                    elif i == 1:  # 前一天
                        prev_close_2 = df_history.iloc[1]['close'] if len(df_history) > 1 else close_price
                        daily_change = ((close_price - prev_close_2) / prev_close_2) * 100 if prev_close_2 != 0 else 0.0
                    else:
                        prev_close_i = df_history.iloc[i]['close']
                        next_close_i = df_history.iloc[i-1]['close']
                        daily_change = ((next_close_i - prev_close_i) / prev_close_i) * 100 if prev_close_i != 0 else 0.0
                    
                    daily_data.append({
                        "date": trade_date,
                        "change": f"{daily_change:.2f}",
                        "price": close_price
                    })
        
        # 处理ETF名称映射
        if is_etf:
            file_path = "data/etf_name_data.json"    
            name_map = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name_map = {str(item['code']): item['name'] for item in data}
                    print(f"Successfully loaded {len(name_map)} names from {file_path}")
                except Exception as e:
                    print(f"Warning: Could not load or parse name map file at {file_path}. Error: {e}")
                    name_map = {}
            else:
                print(f"Info: Name map file not found at {file_path}.")
            
            predefined_name = name_map.get(code)
            if predefined_name:
                name = predefined_name
        
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
        print(f"tushare error for {code}: {str(e)}")
        return None

def fetch_intraday_with_tushare(code: str) -> Optional[Response]:
    """使用tushare获取分时数据"""
    try:
        tushare_code = get_tushare_code(code)
        print(f"Fetching intraday data with tushare for {tushare_code}")
        
        # 获取当日分时数据
        # 注意：实时分时数据可能需要积分，这里使用日线分钟数据
        today = datetime.now().strftime('%Y%m%d')
        
        # 使用通用行情接口获取分钟数据
        # 注意：这个接口可能需要特定的权限
        try:
            # 尝试获取当日分钟数据
            df_intraday = pro.bo_daily(ts_code=tushare_code, trade_date=today)
            
            if df_intraday.empty:
                # 如果当日没有数据，尝试获取最近交易日的分钟数据
                # 首先获取最近交易日
                df_trade_cal = pro.trade_cal(exchange='', start_date=today, end_date=today)
                if not df_trade_cal.empty and df_trade_cal.iloc[0]['is_open'] == 1:
                    # 如果是交易日但没有数据，返回空
                    print(f"No intraday data available for {tushare_code} on {today}")
                    return None
                else:
                    # 如果不是交易日，获取最近一个交易日的分钟数据
                    # 这里简化处理，直接返回空，由yfinance接管
                    return None
        except Exception as e:
            print(f"tushare intraday API error: {e}")
            # 如果接口不可用，返回None，让yfinance处理
            return None
        
        # 处理分时数据
        intraday_records = []
        cumulative_volume = 0
        cumulative_price_volume = 0
        
        # 按时间排序
        df_intraday = df_intraday.sort_values('trade_time')
        
        for _, row in df_intraday.iterrows():
            trade_time = row.get('trade_time')
            price = row.get('price')
            volume = row.get('vol', 0)
            
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
                        date_str = today[:4] + '-' + today[4:6] + '-' + today[6:8]
                        time_str = str(trade_time)
                except:
                    date_str = today[:4] + '-' + today[4:6] + '-' + today[6:8]
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
        print(f"tushare intraday error for {code}: {str(e)}")
        return None

# --- 原有的yfinance函数保持不变 ---
def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据（回落机制）"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol} (fallback)")
        ticker = yf.Ticker(ticker_symbol)
        current_price = None

        try:
            current_price = ticker.fast_info['last_price']
            fast_prev_close = ticker.fast_info['previous_close']
        except Exception:
            current_price = None
            fast_prev_close = None
        
        time.sleep(0.2)        
        
        info = ticker.info

        if current_price is None:
            current_price = info.get('currentPrice')
        
        if current_price is None:
            current_price = info.get('regularMarketPrice')

        if current_price is None:
            print("Falling back to historical data for current price")
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        if current_price is None:
            print(f"No price data available from yfinance for {ticker_symbol}")
            return None        
        
        prev_close = fast_prev_close if fast_prev_close is not None else info.get('previousClose')
        
        if prev_close is None:
            print("prev_close is None")
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
            else:
                prev_close = current_price
        
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100
        
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
            print(f"Fetching 5-day ETF data for {ticker_symbol}")
            data = ticker.history(period="5d")
            
            if not data.empty:
                data = data.sort_index(ascending=False)
                
                for i, (date, row) in enumerate(data.iterrows()):
                    if i == 0:
                        daily_change = ((row['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
                    elif i < len(data) - 1:
                        daily_change = ((row['Close'] - data.iloc[i+1]['Close']) / data.iloc[i+1]['Close']) * 100 if data.iloc[i+1]['Close'] != 0 else 0.0
                    else:
                        daily_change = 0.0
                    
                    date_str = date.strftime('%Y-%m-%d')
                    daily_data.append({
                        "date": date_str,
                        "change": f"{daily_change:.2f}",
                        "price": row['Close']
                    })

        if is_etf:
            file_path = "data/etf_name_data.json"    
            name_map = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name_map = {str(item['code']): item['name'] for item in data}
                    print(f"Successfully loaded {len(name_map)} names from {file_path}")
                except Exception as e:
                    print(f"Warning: Could not load or parse name map file at {file_path}. Error: {e}")
                    name_map = {}
            else:
                print(f"Info: Name map file not found at {file_path}.")
            
            predefined_name = name_map.get(code)
            if predefined_name:
                name = predefined_name        
            
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
        print(f"yfinance error for financial info {code}: {str(e)}")
        return None

# --- 修改后的API Endpoint ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price', 'info', 'movingaveragedata', or 'intraday'")
):
    """
    Fetches stock data based on the code and query type using tushare with fallback to yfinance.
    """
    if query_type == 'price':
        # 先尝试tushare，失败则回落到yfinance
        response = fetch_price_with_tushare(code)
        if response:
            return response
        else:
            print(f"tushare failed, falling back to yfinance for {code}")
            response = fetch_price_with_yfinance(code)
            if response:
                return response
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Price data not found for {code}"
                )

    elif query_type == 'info':
        # info查询保持不变，仍然使用yfinance
        response = fetch_financial_info_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"Financial info not found for {code}"
            )

    elif query_type == 'movingaveragedata':
        # 保持不变
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
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif query_type == 'intraday':
        # 先尝试tushare，失败则回落到yfinance
        tushare_response = fetch_intraday_with_tushare(code)
        if tushare_response:
            return tushare_response
        else:
            print(f"tushare intraday failed, falling back to yfinance for {code}")
            try:
                ticker_symbol = get_yfinance_ticker(code)
                ticker = yf.Ticker(ticker_symbol)

                intraday_data = ticker.history(period="1d", interval="1m", auto_adjust=False)

                if intraday_data.empty:
                    raise HTTPException(status_code=404, detail=f"No intraday data found for {code}")

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

            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter. Use 'price', 'info', 'movingaveragedata', or 'intraday'.")
