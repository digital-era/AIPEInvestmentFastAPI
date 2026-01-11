from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Optional, List
import time
import datetime
import json
import pandas as pd
import os

# 新增导入：mootdx
from mootdx.quotes import Quotes
from mootdx.exceptions import TdxConnectionError, TdxFunctionCallError

app = FastAPI(
    title="Stock Query API",
    description="An API to fetch real-time price and financial info for stocks using yfinance."
)

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
    source: str = Field(..., description="Data source (yfinance)")
    currency: str = Field(..., description="Currency of the stock")
    dailydata: Optional[List[DailyData]] = Field(None, description="Recent 5 trading days data for ETFs")
    class Config:
        validate_by_name = True  # 保持您原始的配置

class InfoResponse(BaseModel):
    pe: float | None = Field(None, description="Price-to-Earnings Ratio (TTM)")
    pb: float | None = Field(None, description="Price-to-Book Ratio")
    roe: float | None = Field(None, description="Return on Equity")
    source: str = Field(..., description="Data source (yfinance)")

# --- Helper Function (完全保持原始状态) ---
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

def fetch_price_with_yfinance(code: str) -> Optional[PriceResponse]:
    """使用yfinance获取股票实时价格数据"""
    try:
        ticker_symbol = get_yfinance_ticker(code)
        print(f"Fetching price data with yfinance for {ticker_symbol}")
        ticker = yf.Ticker(ticker_symbol)
        current_price = None

        # --- [新增/修改 1] 优先尝试从 fast_info 获取更精确的实时数据 ---
        try:
            # fast_info 的 last_price 在盘中通常比 info 更新
            current_price = ticker.fast_info['last_price']
            # 暂存 fast_info 的昨收，供后面使用
            fast_prev_close = ticker.fast_info['previous_close']
        except Exception:
            # 如果版本过低或获取失败，保持为 None，后续逻辑会自动回退
            current_price = None
            fast_prev_close = None
        # --------------------------------------------------------
        
        # 等待一小段时间确保数据加载
        time.sleep(0.2)        
        
        # 获取基本信息
        info = ticker.info

        # --- [修改 2] 仅当 fast_info 没拿到价格时，才尝试 info ---
        if current_price is None:
            current_price = info.get('currentPrice')
        # -----------------------------------------------------
        
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
        
        # --- [修改 3] 获取前一天收盘价 (优先用 fast_info 的数据) ---
        prev_close = fast_prev_close if fast_prev_close is not None else info.get('previousClose')
        # ---------------------------------------------------------
        
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
            code.upper().startswith('55') or 
            code.upper().startswith('51') or 
            code.upper().startswith('15')
        )
        # 检查是否为US stock
        is_us = (
            code.upper().startswith('US') 
        )
        
        daily_data = []
        if is_etf or is_us:
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

        if(is_etf):
            # 优先从我们自己的映射表中获取中文名称
            file_path = "data/etf_name_data.json"    
            name_map = {}
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # 使用字典推导式高效地创建映射
                    name_map = {str(item['code']): item['name'] for item in data}
                    print(f"Successfully loaded {len(name_map)} names from {file_path}")
                except Exception as e:
                    print(f"Warning: Could not load or parse name map file at {file_path}. Error: {e}")
                    name_map = {} # 加载失败则使用空字典
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

# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>   新增/修改：MooTDX 适配逻辑   <<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================

def is_a_share_market(code: str) -> bool:
    """判断是否为A股、ETF或北交所股票"""
    code = code.upper()
    if (code.startswith(('58', '56', '55', '51', '15')) or  # ETF
        code.startswith(('60', '68', '900')) or           # 沪市
        code.startswith(('00', '30', '200')) or           # 深市
        code.startswith(('43', '83', '87', '88'))):       # 北交所
        return True
    return False

def fetch_price_with_mootdx(code: str) -> Optional[PriceResponse]:
    """使用mootdx获取A股/ETF/北交所数据 (安全版本)"""
    client = None
    try:
        # 1. 确定市场类型
        if code.startswith(('43', '83', '87', '88')):
            market = 'bj' # 北交所
        else:
            market = 'std' # 沪深A股及ETF
        
        # 2. 连接服务器
        client = Quotes.factory(market=market, bestip=True)
        if not client:
            raise ConnectionError("无法连接到行情服务器")

        # 3. 获取行情
        result = client.quotes(symbol=[code])
        if result is None or result.empty:
            return None

        row = result.iloc[0]
        
        # 4. 解析数据
        # 现价通常字段名为 'price'
        current_price = float(row['price'])
        
        # --- 【安全逻辑】获取昨收价 ---
        # 不同版本或协议下，昨收的字段名可能不同
        prev_close = None
        # 常见的通达信字段名列表
        possible_keys = ['yesterday', 'pre_close', 'last_close', 'close_yesterday']
        
        for key in possible_keys:
            if key in row:
                try:
                    prev_close = float(row[key])
                    break
                except (ValueError, TypeError):
                    continue
        
        # 如果都没找到，回退到开盘价（保底方案，防止报错）
        if prev_close is None:
            prev_close = float(row['open'])
        
        # 股票名称
        name = str(row['name']).strip()
        
        # 计算涨跌额和涨跌幅
        change_amount = current_price - prev_close
        change_percent = (change_amount / prev_close) * 100 if prev_close != 0 else 0.0

        # 5. 构造响应
        return PriceResponse(
            name=name,
            latestPrice=current_price,
            changePercent=change_percent,
            changeAmount=change_amount,
            source="mootdx",
            currency="CNY",
            dailydata=None # mootdx 实时接口通常不包含历史5日，除非单独查
        )
        
    except Exception as e:
        print(f"mootdx error for {code}: {str(e)}")
        return None
    finally:
        # 6. 释放连接资源
        if client:
            try:
                client.exit()
            except:
                pass

# --- API Endpoint (修改处) ---
@app.get("/api/query")
async def get_stock_data(
    code: str = Query(..., description="The stock code, e.g., '600900' or 'AAPL'"),
    query_type: str = Query(..., alias="type", description="Type of query: 'price', 'info', 'movingaveragedata', or 'intraday'")
):
    """
    Fetches stock data based on the code and query type.
    A股/ETF/北交所 -> mootdx (高速)
    其他 -> yfinance (兼容)
    """
    # 去除可能的空格
    code = code.strip()
    
    if query_type == 'price':
        # >>>>>>> 路由逻辑：先尝试mootdx，失败则降级 <<<<<<<<
        if is_a_share_market(code):
            response = fetch_price_with_mootdx(code)
            if response:
                return response
            else:
                print(f"mootdx failed for {code}, falling back to yfinance")
        
        # 如果不是A股，或者mootdx失败，使用yfinance
        response = fetch_price_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail=f"Price data not found for {code}")

    elif query_type == 'info':
        # 基本面数据统一走yfinance
        response = fetch_financial_info_with_yfinance(code)
        if response:
            return response
        else:
            raise HTTPException(status_code=404, detail=f"Financial info not found for {code}")

    elif query_type == 'movingaveragedata':
        # K线图数据，为了保持格式统一，继续用yfinance
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
            
            # 将 DataFrame 转换为 JSON，同时处理日期索引
            json_output = plot_data.reset_index().to_json(orient='records', date_format='iso')
            
            # 直接返回JSON响应
            return Response(content=json_output, media_type="application/json")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif query_type == 'intraday':
        try:
            ticker_symbol = get_yfinance_ticker(code)
            ticker = yf.Ticker(ticker_symbol)

            intraday_data = ticker.history(period="1d", interval="1m", auto_adjust=False)

            if intraday_data.empty:
                raise HTTPException(status_code=404, detail=f"No intraday data found for {code}. It might be a non-trading day.")

            # --- 计算累计均价 (VWAP) ---
            intraday_data['PriceVolume'] = intraday_data['Close'] * intraday_data['Volume']
            intraday_data['CumulativeVolume'] = intraday_data['Volume'].cumsum()
            intraday_data['CumulativePriceVolume'] = intraday_data['PriceVolume'].cumsum()
            intraday_data['avg_price'] = intraday_data['CumulativePriceVolume'] / intraday_data['CumulativeVolume']

            # --- 【核心修正】在返回数据中同时包含日期和时间 ---
            intraday_data['date'] = intraday_data.index.strftime('%Y-%m-%d') # 新增：提取日期
            intraday_data['time'] = intraday_data.index.strftime('%H:%M:%S') # 保留：提取时间
            
            # 在返回的列中增加 'date'
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
