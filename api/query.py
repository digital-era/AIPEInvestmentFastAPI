import os

# --- ⚠️ 必须放在文件顶部，用于修复 Vercel 只读文件系统错误 ---
# mootdx 会在启动时尝试创建 .mootdx 配置目录，默认路径是 ~/
# Vercel 环境是只读的，所以必须将其指向 /tmp (唯一可写的目录)
os.environ['HOME'] = '/tmp'
# 确保 /tmp/.mootdx 目录存在
mootdx_config_dir = '/tmp/.mootdx'
if not os.path.exists(mootdx_config_dir):
    os.makedirs(mootdx_config_dir, exist_ok=True)
# ------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
import json
import pandas as pd
import os

# 现在再导入 mootdx，它会使用 /tmp/.mootdx 作为配置目录
from mootdx.quotes import Quotes
from mootdx.exceptions import TdxConnectionError, TdxFunctionCallError

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks."
)

# --- Pydantic Models (完全保持原始状态，用于验证输出格式) ---
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

# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>   新增：MooTDX 适配层 (完全隔离)   <<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================

def is_a_share_market(code: str) -> bool:
    """判断是否为A股、ETF或北交所股票 (仅这部分走mootdx)"""
    code = code.upper()
    if (code.startswith(('58', '56', '55', '51', '15')) or  # ETF
        code.startswith(('60', '68', '900')) or           # 沪市
        code.startswith(('00', '30', '200')) or           # 深市
        code.startswith(('43', '83', '87', '88'))):       # 北交所
        return True
    return False

def fetch_price_with_mootdx(code: str) -> Optional[PriceResponse]:
    """
    仅用于A股/ETF/北交所。
    目标：返回的数据结构必须和 yfinance 的 PriceResponse 完全一致。
    """
    client = None
    try:
        # 1. 选择市场
        if code.startswith(('43', '83', '87', '88')):
            market = 'bj'
        else:
            market = 'std' # 沪深
        
        # 2. 连接行情服务器
        client = Quotes.factory(market=market, bestip=True)
        if not client:
            raise ConnectionError("无法连接到通达信服务器")

        # 3. 获取实时行情
        # 注意：传入的是纯数字代码
        result = client.quotes(symbol=[code])
        if result is None or result.empty:
            return None

        # 4. 解析数据 (核心：字段映射)
        row = result.iloc[0]
        
        # --- 字段提取 (增加了严格的类型转换和异常捕获) ---
        try:
            # 现价
            current_price = float(row['price'])
            # 昨收 (通达信标准字段是 'yesterday')
            prev_close = float(row['yesterday'])
            # 名称
            name = str(row['name']).strip()
        except (KeyError, ValueError) as e:
            print(f"Data parse error in mootdx: {e}")
            return None

        # --- 计算涨跌 ---
        # 防止除以0
        if prev_close == 0:
            change_percent = 0.0
        else:
            change_amount = current_price - prev_close
            change_percent = (change_amount / prev_close) * 100
        change_amount = current_price - prev_close

        # --- 构造响应 ---
        # 必须使用 **kwargs 或者严格按字段名传参，以匹配 alias
        return PriceResponse(
            name=name,
            latestPrice=current_price,       # 对应 latest_price
            changePercent=change_percent,    # 对应 change_percent
            changeAmount=change_amount,      # 对应 change_amount
            source="mootdx",                 # 标记数据源
            currency="CNY",
            dailydata=None # mootdx实时接口不包含日线列表，设为None
        )
        
    except Exception as e:
        print(f"mootdx error: {e}")
        return None
    finally:
        # 必须关闭连接，否则会堆积句柄
        if client:
            try:
                client.exit()
            except:
                pass

# ==============================================================================
# >>>>>>>>>>>>>>>   以下是原始 yfinance 代码 (保持原封不动)   <<<<<<<<<<<<<<<
# ==============================================================================

def get_yfinance_ticker(code: str) -> str:
    """将股票代码转换为yfinance可识别的格式"""
    # 港股处理（格式如: HK02899, hk00005, HK03690）
    if code.upper().startswith('HK'):
        # 提取数字部分，最多只移除开头的1个零
        num_part = code[2:]
        if num_part.startswith('0') and len(num_part) > 1:
            num_part = num_part[1:]  # 只移除开头的第一个零
        return f"{num_part}.HK"
    elif code.upper().startswith('US'):  # 美股
        # 提取数字部分，最多只移除开头的US
        code_part = code[2:]
        return f"{code_part}"
    
    # A股处理 (这里只是辅助，因为A股主要走mootdx，但作为降级方案保留)
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

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据 (原始逻辑，未修改)"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol}")
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
        is_us = (
            code.upper().startswith('US') 
        )
        
        daily_data = []
        if is_etf or is_us:
            print(f"Fetching 5-day ETF data for {ticker_symbol}")
            data = ticker.history(period="5d")
            
            if not data.empty:
                data = data.sort_index(ascending=False)
                
                for i, (date, row) in enumerate(data.iterrows()):
                    if i == 0:  # 最新一天
                        daily_change = ((row['Close'] - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0
                    elif i < len(data) - 1:  # 中间日期
                        daily_change = ((row['Close'] - data.iloc[i+1]['Close']) / data.iloc[i+1]['Close']) * 100 if data.iloc[i+1]['Close'] != 0 else 0.0
                    else:  # 最早一天
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
                print(f"Info: Name map file not found at {file_path}. Names will be fetched from yfinance.")
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
    """使用yfinance获取金融信息 (原始逻辑，未修改)"""
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

        return InfoResponse(
            pe=pe,
            pb=pb,
            roe=roe,
            source="yfinance"
        )
    
    except Exception as e:
        print(f"yfinance error for financial info {code}: {str(e)}")
        return None

# --- API Endpoint (修改处：增加了路由判断) ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price', 'info', 'movingaveragedata', or 'intraday'")
):
    """
    路由逻辑：
    1. 如果是查询 'price' 且是 A股/ETF/北交所 -> 使用 mootdx (速度快)
    2. 其他所有情况 (美股、港股、基本面、K线) -> 使用 yfinance (兼容性好)
    """
    code = code.strip()
    
    if query_type == 'price':
        # --- 路由分发 ---
        if is_a_share_market(code):
            print(f"[mootdx] Handling A-share: {code}")
            response = fetch_price_with_mootdx(code)
            if response:
                return response
            else:
                # 如果mootdx失败(例如服务器挂了)，降级到yfinance，保证接口不死
                print(f"[Fallback] mootdx failed for {code}, using yfinance")
        
        # 非A股，或者mootdx降级
        response = fetch_price_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail=f"Price data not found for {code}")

    elif query_type == 'info':
        # 基本面数据统一走yfinance，mootdx获取财务数据较慢且复杂
        response = fetch_financial_info_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail=f"Financial info not found for {code}")

    # --- 以下原功能保持完全不变 ---
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
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif query_type == 'intraday':
        try:
            ticker_symbol = get_yfinance_ticker(code)
            ticker = yf.Ticker(ticker_symbol)
            intraday_data = ticker.history(period="1d", interval="1m", auto_adjust=False)

            if intraday_data.empty:
                raise HTTPException(status_code=404, detail=f"No intraday data found for {code}")

            # 计算均价逻辑
            intraday_data['PriceVolume'] = intraday_data['Close'] * intraday_data['Volume']
            intraday_data['CumulativeVolume'] = intraday_data['Volume'].cumsum()
            intraday_data['CumulativePriceVolume'] = intraday_data['PriceVolume'].cumsum()
            # 防止除以0
            intraday_data['CumulativeVolume'] = intraday_data['CumulativeVolume'].replace(0, 1e-10)
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
        raise HTTPException(status_code=400, detail="Invalid 'type' parameter.")
