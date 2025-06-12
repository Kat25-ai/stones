import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
from bs4 import BeautifulSoup
from textblob import TextBlob
import pytz
import time

# --- Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="Professional Forex Fundamental Analyzer",
    page_icon="📊"
)

# --- API Keys (replace with your own) ---
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "YOUR_NEWSAPI_KEY")
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "YOUR_ALPHA_VANTAGE_KEY")
OANDA_TOKEN = st.secrets.get("OANDA_TOKEN", "YOUR_OANDA_TOKEN")

# --- Constants ---
CENTRAL_BANKS = {
    "Fed": "Federal Reserve (USD)",
    "ECB": "European Central Bank (EUR)",
    "BOE": "Bank of England (GBP)",
    "BOJ": "Bank of Japan (JPY)",
    "BOC": "Bank of Canada (CAD)",
    "RBA": "Reserve Bank of Australia (AUD)",
    "RBNZ": "Reserve Bank of New Zealand (NZD)",
    "SNB": "Swiss National Bank (CHF)"
}

# --- Data Fetching Functions ---
def get_economic_calendar():
    """Get high-impact events from Forex Factory RSS"""
    try:
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        feed = feedparser.parse(url)
        
        events = []
        for entry in feed.entries:
            if 'High' in entry.get('impact', ''):
                event_time = datetime.strptime(entry.get('date'), "%a %b %d %H:%M:%S %Y")
                
                # Only show upcoming or recently past events
                if event_time > datetime.now() - timedelta(hours=6):
                    events.append({
                        "date": event_time.strftime("%Y-%m-%d %H:%M"),
                        "currency": entry.get('currency', ''),
                        "title": entry.title,
                        "forecast": entry.get('forecast', 'N/A'),
                        "previous": entry.get('previous', 'N/A'),
                        "actual": entry.get('actual', 'Pending'),
                        "timestamp": event_time,
                        "impact": entry.get('impact', 'Medium')
                    })
        return pd.DataFrame(events)
    except:
        return pd.DataFrame()

def get_news_sentiment(query="interest rates"):
    """Get recent news sentiment from NewsAPI"""
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
        response = requests.get(url).json()
        
        articles = []
        for article in response.get('articles', [])[:15]:
            text = f"{article['title']}. {article.get('description', '')}"
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            
            articles.append({
                "source": article['source']['name'],
                "title": article['title'],
                "sentiment": polarity,
                "sentiment_label": "Bullish" if polarity > 0.1 else "Bearish" if polarity < -0.1 else "Neutral",
                "url": article['url'],
                "published": article['publishedAt'][:19].replace("T", " "),
                "color": "green" if polarity > 0.1 else "red" if polarity < -0.1 else "gray"
            })
        return pd.DataFrame(articles)
    except:
        return pd.DataFrame()

def get_central_bank_speeches():
    """Fetches latest central bank communications"""
    try:
        speeches = []
        
        # Federal Reserve (Fed)
        fed_url = "https://www.federalreserve.gov/newsevents.htm"
        fed_response = requests.get(fed_url)
        fed_soup = BeautifulSoup(fed_response.text, 'html.parser')
        
        for item in fed_soup.select('.eventlist__item')[:5]:
            title = item.select_one('.eventlist__title').text.strip()
            date_str = item.select_one('.eventlist__time').text.strip()
            link = "https://www.federalreserve.gov" + item.find('a')['href']
            
            # Parse date (e.g., "Nov 6, 2023 | Speech")
            event_date = datetime.strptime(date_str.split(" | ")[0], "%b %d, %Y")
            
            speeches.append({
                "central_bank": "Fed",
                "title": title,
                "date": event_date.strftime("%Y-%m-%d"),
                "url": link,
                "timestamp": event_date
            })
        
        # Add more central banks here following similar pattern
        
        return pd.DataFrame(speeches)
    except:
        return pd.DataFrame()

# --- Analysis Functions ---
class CentralBankDecoder:
    def __init__(self):
        self.hawkish_phrases = {
            "inflation concern": 0.9,
            "tighten policy": 0.95,
            "strong economy": 0.8,
            "overheating": 0.85,
            "rate hike": 1.0,
            "higher for longer": 0.9
        }
        
        self.dovish_phrases = {
            "patient approach": 0.9,
            "growth risks": 0.85,
            "accommodative": 0.95,
            "monitor closely": 0.75,
            "rate cut": 1.0,
            "easing policy": 0.85
        }
        
        self.hedge_phrases = [
            "data dependent", "uncertain", "transitory", 
            "crosscurrents", "balance risks", "depending on"
        ]
    
    def analyze_speech(self, title):
        """Mock analysis - in real app, would analyze full speech text"""
        # Simple pattern matching for demo
        text = title.lower()
        
        hawk_score = sum(weight for phrase, weight in self.hawkish_phrases.items() 
                         if phrase in text)
        dovish_score = sum(weight for phrase, weight in self.dovish_phrases.items() 
                           if phrase in text)
        
        # Calculate confidence
        confidence = min(100, max(hawk_score, dovish_score) * 20)
        
        if hawk_score > dovish_score:
            return "Hawkish", confidence
        elif dovish_score > hawk_score:
            return "Dovish", confidence
        return "Neutral", confidence

def calculate_safety_score(event):
    """Calculate trading safety score (0-100)"""
    score = 50
    
    # Positive factors
    if event['actual'] != 'Pending' and event['actual'] != event['forecast']:
        score += 25  # Surprise factor
    if "rate" in event['title'].lower():
        score += 15  # Interest rate events are more reliable
    
    # Negative factors
    if "Fed" in event['title'] or "ECB" in event['title']:
        score -= 10  # High volatility expected
    if "speech" in event['title'].lower():
        score -= 20  # Unpredictable outcomes
        
    return max(0, min(100, score))

def generate_trade_signals(events, speeches):
    """Generate trade signals based on events and speeches"""
    signals = []
    
    # Process events
    for _, event in events.iterrows():
        if event['actual'] == 'Pending' or event['currency'] == '':
            continue
            
        safety = calculate_safety_score(event)
        surprise = 0
        
        try:
            # Calculate surprise factor
            actual_val = float(event['actual'])
            forecast_val = float(event['forecast'])
            surprise = (actual_val - forecast_val) / forecast_val * 100
        except:
            surprise = 0
            
        if safety > 65 and abs(surprise) > 10:
            signal = {
                "type": "EVENT",
                "pair": f"{event['currency']}/USD",
                "direction": "BUY" if surprise > 0 else "SELL",
                "strength": min(100, abs(surprise) * 2),
                "reason": event['title'],
                "confidence": safety,
                "expiry": (event['timestamp'] + timedelta(hours=4)).strftime("%H:%M")
            }
            signals.append(signal)
    
    # Process central bank speeches
    decoder = CentralBankDecoder()
    for _, speech in speeches.iterrows():
        direction, confidence = decoder.analyze_speech(speech['title'])
        if confidence > 60:
            currency = CENTRAL_BANKS[speech['central_bank']].split()[-1].strip("()")
            signal = {
                "type": "SPEECH",
                "pair": f"{currency}/USD",
                "direction": "BUY" if direction == "Hawkish" else "SELL",
                "strength": confidence,
                "reason": speech['title'],
                "confidence": confidence,
                "expiry": (speech['timestamp'] + timedelta(days=1)).strftime("%Y-%m-%d")
            }
            signals.append(signal)
    
    return signals

# --- Dashboard UI ---
st.title("💰 Professional Forex Fundamental Analyzer")
st.caption("Real-time News, Events, and Central Bank Analysis")

# --- Data Loading ---
calendar_df = get_economic_calendar()
news_df = get_news_sentiment()
speeches_df = get_central_bank_speeches()
trade_signals = generate_trade_signals(calendar_df, speeches_df)

# --- Layout Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Market Overview", 
    "🗓️ Economic Calendar", 
    "🏦 Central Bank Analysis",
    "🚦 Trading Signals"
])

with tab1:
    st.subheader("Market Sentiment Dashboard")
    
    # Sentiment summary
    if not news_df.empty:
        avg_sentiment = news_df['sentiment'].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Sentiment", 
                   "Bullish 🚀" if avg_sentiment > 0.1 else "Bearish 🐻" if avg_sentiment < -0.1 else "Neutral ➖",
                   f"{avg_sentiment:.2f}")
        
        # News sentiment timeline
        news_df['published'] = pd.to_datetime(news_df['published'])
        fig = px.scatter(
            news_df, 
            x='published', 
            y='sentiment',
            color='sentiment_label',
            color_discrete_map={
                "Bullish": "green",
                "Bearish": "red",
                "Neutral": "gray"
            },
            hover_data=['title', 'source'],
            title="News Sentiment Timeline"
        )
        fig.update_layout(yaxis_range=[-1,1], height=300)
        col2.plotly_chart(fig, use_container_width=True)
        
        # Sentiment distribution
        sentiment_dist = news_df['sentiment_label'].value_counts()
        fig = px.pie(
            sentiment_dist,
            names=sentiment_dist.index,
            values=sentiment_dist.values,
            title="Sentiment Distribution",
            hole=0.5
        )
        col3.plotly_chart(fig, use_container_width=True)
        
        # Display news articles
        st.subheader("Latest Market News")
        for _, article in news_df.iterrows():
            with st.expander(f"{article['source']}: {article['title']}"):
                st.caption(f"Published: {article['published']} | Sentiment: {article['sentiment_label']}")
                st.markdown(f"[Read Full Article]({article['url']})")
    else:
        st.warning("No financial news available")

with tab2:
    st.subheader("High-Impact Economic Events")
    
    if calendar_df.empty:
        st.warning("No upcoming high-impact events")
    else:
        # Current time marker
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.info(f"Current Time: {now} | Timezone: UTC")
        
        # Display events in timeline
        for _, event in calendar_df.iterrows():
            safety = calculate_safety_score(event)
            color = "green" if safety > 65 else "orange" if safety > 40 else "red"
            time_diff = (event['timestamp'] - datetime.now()).total_seconds()/3600
            
            with st.container(border=True):
                cols = st.columns([1, 4, 1])
                cols[0].subheader(event['currency'])
                cols[0].caption(event['timestamp'].strftime("%H:%M"))
                
                cols[1].write(f"**{event['title']}**")
                cols[1].caption(f"Forecast: {event['forecast']} | Actual: {event['actual']}")
                
                cols[2].metric("Safety", f"{safety}/100", delta_color="off")
                cols[2].progress(safety/100, text=f"{safety}%")
                
                # Time indicator
                if time_diff > 0:
                    cols[1].caption(f"⏱️ In {abs(time_diff):.1f} hours")
                else:
                    cols[1].caption(f"🕒 {abs(time_diff):.1f} hours ago")

with tab3:
    st.subheader("Central Bank Communications Analysis")
    
    if speeches_df.empty:
        st.warning("No recent central bank communications")
    else:
        decoder = CentralBankDecoder()
        
        # Central bank status overview
        st.subheader("Central Bank Policy Stance")
        policy_cols = st.columns(len(speeches_df['central_bank'].unique()))
        
        for idx, bank in enumerate(speeches_df['central_bank'].unique()):
            bank_speeches = speeches_df[speeches_df['central_bank'] == bank]
            direction, confidence = decoder.analyze_speech(
                bank_speeches.iloc[0]['title']
            )
            
            with policy_cols[idx]:
                st.metric(
                    bank, 
                    direction,
                    f"{confidence}% confidence",
                    delta_color="off"
                )
                st.progress(confidence/100)
        
        # Detailed speeches
        st.subheader("Recent Speeches & Statements")
        for _, speech in speeches_df.iterrows():
            direction, confidence = decoder.analyze_speech(speech['title'])
            color = "green" if direction == "Hawkish" else "red" if direction == "Dovish" else "gray"
            
            with st.expander(f"{speech['central_bank']}: {speech['title']}"):
                st.markdown(f"**Date**: {speech['date']}")
                st.markdown(f"**Analysis**: :{color}[**{direction}**] with {confidence}% confidence")
                st.markdown(f"**Expected Impact**: Currencies may respond with **{'strength' if direction == 'Hawkish' else 'weakness'}**")
                st.markdown(f"[Read Full Speech]({speech['url']})")

with tab4:
    st.subheader("Trade Signal Dashboard")
    
    if not trade_signals:
        st.info("No strong trading signals detected")
    else:
        # Sort by confidence
        trade_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Display signals
        for signal in trade_signals:
            color = "green" if signal['direction'] == "BUY" else "red"
            emoji = "🚀" if signal['direction'] == "BUY" else "🛑"
            
            with st.container(border=True):
                cols = st.columns([1, 3, 1])
                cols[0].subheader(f":{color}[{signal['pair']}]")
                cols[0].write(f"### :{color}[{signal['direction']}] {emoji}")
                
                cols[1].write(f"**{signal['reason']}**")
                cols[1].progress(signal['confidence']/100, text=f"Confidence: {signal['confidence']}%")
                cols[1].caption(f"Signal Type: {signal['type']} | Expires: {signal['expiry']}")
                
                cols[2].metric("Strength", f"{signal['strength']}/100", delta_color="off")
                
                # Trading advice
                if signal['strength'] > 80:
                    cols[2].success("Strong Signal")
                elif signal['strength'] > 60:
                    cols[2].warning("Moderate Signal")
                else:
                    cols[2].info("Weak Signal")
        
        # Trading strategy tips
        st.subheader("Trading Strategy Recommendations")
        st.markdown("""
        - **Entry Strategy**: 
          - Wait for price confirmation (1-5 minute closing above entry for BUY, below for SELL)
          - Use limit orders at key support/resistance levels
          
        - **Risk Management**:
          - Set stop-loss at 1.0-1.5x ATR
          - Risk no more than 1% per trade
          - Take profit at 2.0-3.0x risk
          
        - **Position Sizing**:
          ```python
          def calculate_position_size(account_size, risk_percent, stop_loss_pips):
              risk_amount = account_size * (risk_percent / 100)
              position_size = risk_amount / (stop_loss_pips * 10)
              return round(position_size, 2)
          ```
        """)

# --- Sidebar ---
st.sidebar.header("Professional Trading Tools")
st.sidebar.subheader("Risk Calculator")
account_size = st.sidebar.number_input("Account Size ($)", 1000, 1000000, 10000)
risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.1, 10.0, 1.0)
stop_loss = st.sidebar.number_input("Stop Loss (pips)", 5, 100, 30)

position_size = account_size * (risk_percent / 100) / (stop_loss * 10)
st.sidebar.metric("Position Size (Lots)", round(position_size, 2))

st.sidebar.subheader("Market Status")
market_status = st.sidebar.selectbox("Current Market Phase", [
    "Trending (Strong Direction)",
    "Ranging (Sideways)",
    "Breakout (Volatile)",
    "News-Driven (Event Risk)"
])

if "Trending" in market_status:
    st.sidebar.info("✅ Favor trend-following strategies")
elif "Ranging" in market_status:
    st.sidebar.info("🔄 Use range-bound strategies")
elif "Breakout" in market_status:
    st.sidebar.warning("⚠️ Trade breakouts with confirmation")
else:
    st.sidebar.error("❗ Reduce position size, avoid trading during events")

st.sidebar.divider()
st.sidebar.caption("Refresh data every 5 minutes")
if st.sidebar.button("Refresh Data"):
    st.rerun()

# --- Auto-refresh ---
# st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")