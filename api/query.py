import os
import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
from datetime import datetime, timedelta
import pytz

# ==============================================================================
# >>>>>>>>>>>>>>>   核心修复：Vercel 环境兼容性处理   <<<<<<<<<<<<<<<
# ==============================================================================
# 必须在导入 mootdx 之前执行
os.environ['HOME'] = '/tmp'
mootdx_config_dir = '/tmp/.mootdx'
if not os.path.exists(mootdx_config_dir):
    os.makedirs(mootdx_config_dir, exist_ok=True)
# ==============================================================================

from mootdx.quotes import Quotes

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks."
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
    source: str = Field(..., description="Data source")
    currency: str = Field(..., description="Currency")
    dailydata: Optional[List[DailyData]] = Field(None, description="5 days data")
    
    class Config:
        validate_by_name = True

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="P/E Ratio")
    pb: float | None = Field(None, description="P/B Ratio")
    roe: float | None = Field(None, description="ROE")
    source: str = Field(..., description="Data source")

# --- Helper Functions ---

def is_a_share_market(code: str) -> bool:
    code = code.upper()
    if (code.startswith(('58', '56', '55', '51', '15')) or  
        code.startswith(('60', '68', '900')) or           
        code.startswith(('00', '30', '200')) or           
        code.startswith(('43', '83', '87', '88'))):       
        return True
    return False

def get_yfinance_ticker(code: str) -> str:
    code = code.upper()
    if code.startswith('HK'):
        num_part = code[2:]
        if num_part.startswith('0') and len(num_part) > 1:
            num_part = num_part[1:]
        return f"{num_part}.HK"
    elif code.startswith('US'):
        return code[2:]
    
    if code.startswith(('60', '68', '900')):
        return f"{code}.SS"
    elif code.startswith(('00', '30', '200')):
        return f"{code}.SZ"
    elif code.startswith(('43', '83', '87', '88')):
        return f"{code}.BJ"
    elif code.startswith(('58', '56','55', '51')):
        return f"{code}.SS"
    elif code.startswith(('15')):
        return f"{code}.SZ"
    else:
        return code

def fetch_price_with_mootdx(code: str) -> Optional[PriceResponse]:
    client = None
    try:
        # --- 1. 确定市场 ---
        if code.startswith(('43', '83', '87', '88')):
            market = 'bj'
        else:
            market = 'std'

        # --- 2. ⚠️ 手动指定服务器 IP 和端口 (硬编码) ---
        # 这是解决 "got 0" 报错的关键，不再依赖库自动获取
        SERVER_IP = "113.105.152.49"  # 常用的通达信行情 IP
        SERVER_PORT = 7709

        # --- 3. 强制使用指定的 IP 创建客户端 ---
        # 这样会跳过配置文件和自动测速逻辑
        client = Quotes.factory(market=market, ip=SERVER_IP, port=SERVER_PORT)
        
        if not client:
            raise Exception("无法初始化客户端")

        result = client.quotes(symbol=[code])
        
        if result is None or result.empty:
            print(f"mootdx: 未获取到 {code} 的数据")
            return None

        row = result.iloc[0]
        
        # --- 4. 安全提取数据 ---
        price_val = row.get('price')
        yesterday_val = row.get('yesterday') or row.get('pre_close') or row.get('open')
        
        if price_val is None or yesterday_val is None:
            print("数据字段缺失")
            return None

        try:
            current_price = float(price_val)
            prev_close = float(yesterday_val)
        except (ValueError, TypeError):
            print("数据类型转换失败")
            return None

        # --- 5. 计算涨跌幅 ---
        if prev_close == 0:
            change_percent = 0.0
            change_amount = 0.0
        else:
            change_amount = current_price - prev_close
            change_percent = (change_amount / prev_close) * 100

        return PriceResponse(
            name=str(row.get('name', code)),
            latestPrice=current_price,
            changePercent=round(change_percent, 2),
            changeAmount=change_amount,
            source="mootdx",
            currency="CNY",
            dailydata=None 
        )
        
    except Exception as e:
        print(f"mootdx error: {e}")
        return None
    finally:
        if client:
            try:
                client.exit()
            except:
                pass

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    try:
        ticker_symbol = get_yfinance_ticker(code)
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
            data = ticker.history(period="1d")
            if not data.empty:
                current_price = data['Close'].iloc[-1]
        
        if current_price is None:
            return None        
        
        prev_close = fast_prev_close or info.get('previousClose')
        
        if prev_close is None:
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev_close = hist['Close'].iloc[-2]
            else:
                prev_close = current_price
        
        if prev_close is None or prev_close == 0:
            prev_close = current_price

        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100 if prev_close != 0 else 0.0
        
        name = info.get('shortName', info.get('longName', code))
        currency = info.get('currency', 'USD')
        
        is_etf = code.upper().startswith(('58', '56', '55', '51', '15'))
        is_us = code.upper().startswith('US')
        
        daily_data = []
        if is_etf or is_us:
            data = ticker.history(period="5d")
            if not data.empty:
                data = data.sort_index(ascending=False)
                for i, (date, row) in enumerate(data.iterrows()):
                    if i == 0:
                        daily_change = ((row['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
                    elif i < len(data) - 1:
                        prev_day_close = data.iloc[i+1]['Close']
                        daily_change = ((row['Close'] - prev_day_close) / prev_day_close) * 100 if prev_day_close != 0 else 0.0
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
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    name_map = {str(item['code']): item['name'] for item in data}
                    predefined_name = name_map.get(code)
                    if predefined_name:
                        name = predefined_name        
                except Exception as e:
                    print(f"Warning: {e}")

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

# --- API Endpoint ---

@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="Stock code"),
    query_type: str = Query(..., alias="type", description="Query type")
):
    code = code.strip()
    
    if query_type == 'price':
        if is_a_share_market(code):
            response = fetch_price_with_mootdx(code)
            if response:
                return response
            else:
                print(f"mootdx failed for {code}, falling back to yfinance")
        
        response = fetch_price_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail=f"Price not found: {code}")

    elif query_type == 'info':
        response = fetch_price_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail=f"Info not found: {code}")

    elif query_type == 'intraday':
        try:
            ticker_symbol = get_yfinance_ticker(code)
            ticker = yf.Ticker(ticker_symbol)
            
            intraday_data = ticker.history(period="1d", interval="1m")
            
            if intraday_data.empty:
                intraday_data = ticker.history(period="5d", interval="5m")
                if intraday_data.empty:
                    return Response(content="[]", media_type="application/json")

            # 计算均价 (VWAP)
            if 'Volume' in intraday_data.columns and 'Close' in intraday_data.columns:
                volume = intraday_data['Volume'].replace(0, 1e-10)
                intraday_data['avg_price'] = (intraday_data['Close'] * volume).cumsum() / volume.cumsum()
            else:
                intraday_data['avg_price'] = intraday_data['Close']

            intraday_data = intraday_data.reset_index()
            intraday_data['date'] = intraday_data['Datetime'].dt.strftime('%Y-%m-%d')
            intraday_data['time'] = intraday_data['Datetime'].dt.strftime('%H:%M:%S')
            
            columns_to_keep = ['date', 'time', 'Close', 'avg_price']
            if 'Volume' in intraday_data.columns:
                columns_to_keep.append('Volume')
                
            result_df = intraday_data[columns_to_keep].rename(columns={
                'Close': 'price',
                'Volume': 'volume'
            })
            
            result_df = result_df.fillna(method='ffill').fillna(0)
            json_output = result_df.to_json(orient='records')
            return Response(content=json_output, media_type="application/json")

        except Exception as e:
            print(f"Intraday error: {e}")
            return Response(content="[]", media_type="application/json")

    else:
        raise HTTPException(status_code=400, detail="Invalid type parameter")
