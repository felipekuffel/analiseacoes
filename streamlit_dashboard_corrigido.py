
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from finvizfinance.screener.overview import Overview
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(layout="wide")
def plot_ativo(df, ticker, nome_empresa, vcp_detectado=False):
    df['DataStr'] = df.index.strftime("%d %b")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['DataStr'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Candlestick'))
    fig.add_trace(go.Scatter(x=df['DataStr'], y=df['SMA50'], mode='lines',
                             line=dict(color='white'), name='Média 50 dias'))
    fig.add_trace(go.Scatter(x=df['DataStr'], y=df['High20'], line=dict(color='orange', dash='dot'),
                             name='Resistência'))
    fig.add_trace(go.Scatter(x=df['DataStr'], y=df['suporte'], line=dict(color='blue', dash='dot'),
                             name='Suporte'))
    fig.add_trace(go.Bar(x=df['DataStr'], y=df['momentum'],
                         marker_color=['aqua' if m > 0 else 'red' for m in df['momentum']],
                         name='Momentum'))
    df_up = df[df['momentum_up']]
    df_rompe = df[df['rompe_resistencia']]
    fig.add_trace(go.Scatter(x=df_up['DataStr'], y=df_up['High'] * 1.02,
                             mode='markers', marker=dict(symbol='triangle-up', color='violet', size=10),
                             name='Momentum Up'))
    fig.add_trace(go.Scatter(x=df_rompe['DataStr'], y=df_rompe['High'] * 1.05,
                             mode='markers', marker=dict(symbol='diamond', color='lime', size=10),
                             name='Rompimento'))

    if vcp_detectado:
        last_index = df['DataStr'].iloc[-1]
        last_price = df['Close'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[last_index],
            y=[last_price * 1.08],
            mode='markers',
            marker=dict(symbol='star-diamond', color='magenta', size=14),
            name='Padrão VCP'
        ))

    PP, suportes, resistencias = calcular_pivot_points(df)
    for s in suportes:
        fig.add_shape(type='rect', x0=df['DataStr'].iloc[0], x1=df['DataStr'].iloc[-1],
                      y0=s * 0.997, y1=s * 1.003, fillcolor='rgba(0,0,255,0.1)', layer='below', line_width=0)
    for r in resistencias:
        fig.add_shape(type='rect', x0=df['DataStr'].iloc[0], x1=df['DataStr'].iloc[-1],
                      y0=r * 0.997, y1=r * 1.003, fillcolor='rgba(255,0,0,0.1)', layer='below', line_width=0)

    fig.update_layout(template='plotly_dark', height=600, hovermode='x unified',
                      xaxis_rangeslider_visible=False,
                      xaxis=dict(type='category', tickangle=-45),
                      yaxis=dict(side='right', title='Preço'))
    return fig

# ---------------------- FUNÇÕES DE INDICADORES ----------------------

def pine_linreg(series, length):
    def linreg_last(x):
        idx = np.arange(length)
        slope, intercept = np.polyfit(idx, x, 1)
        return intercept + slope * (length - 1)
    return series.rolling(length).apply(linreg_last, raw=True)

def calcular_indicadores(df, length=20, momentum_threshold=0.07):
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df[(df['High'] > df['Low']) & (df['Open'] != df['Close'])]
    df['High20'] = df['High'].rolling(length).max().shift(1)
    df['Low20'] = df['Low'].rolling(length).min()
    df['SMA50'] = df['Close'].rolling(50).mean()
    centro = ((df['High20'] + df['Low20']) / 2 + df['Close'].rolling(length).mean()) / 2
    df['linreg_close'] = pine_linreg(df['Close'], length)
    df['momentum'] = df['linreg_close'] - centro
    df['momentum_up'] = (df['momentum'].shift(1) <= 0) & (df['momentum'] > momentum_threshold)
    df['rompe_resistencia'] = df['Close'] > df['High20']
    df['suporte'] = df['Low'].rolling(length).min()
    return df

def detectar_vcp(df):
    if 'Volume' not in df.columns or len(df) < 60:
        return False
    closes = df['Close']
    std_20 = closes.rolling(20).std()
    contracoes = (std_20 < std_20.shift(1)) & (std_20.shift(1) < std_20.shift(2))
    volume_decrescente = (df['Volume'].rolling(20).mean() < df['Volume'].rolling(20).mean().shift(5))
    return contracoes.iloc[-1] and volume_decrescente.iloc[-1]

def calcular_pivot_points(df):
    high = df['High'].iloc[-2]
    low = df['Low'].iloc[-2]
    close = df['Close'].iloc[-2]
    PP = (high + low + close) / 3
    R1 = 2 * PP - low
    S1 = 2 * PP - high
    R2 = PP + (R1 - S1)
    S2 = PP - (R1 - S1)
    R3 = high + 2 * (PP - low)
    S3 = low - 2 * (high - PP)
    return PP, [S1, S2, S3], [R1, R2, R3]

def classificar_tendencia(close):
    x = np.arange(len(close))
    slope, _ = np.polyfit(x, close, 1)
    if slope > 0.05:
        return "Alta"
    elif slope < -0.05:
        return "Baixa"
    else:
        return "Lateral"

def avaliar_risco(df):
    preco_atual = df['Close'].iloc[-1]
    sup10 = df['Low'].rolling(10).min().iloc[-1]
    res10 = df['High'].rolling(10).max().iloc[-1]
    sup20 = df['Low'].rolling(20).min().iloc[-1]
    res20 = df['High'].rolling(20).max().iloc[-1]
    dist_sup = min(abs(preco_atual - sup10), abs(preco_atual - sup20))
    dist_res = min(abs(preco_atual - res10), abs(preco_atual - res20))
    base_risk = (dist_res / preco_atual) * 10 if dist_sup < dist_res else 8 + (dist_sup / preco_atual) * 2
    if df['rompe_resistencia'].iloc[-1]: base_risk += 1
    if df['momentum_up'].iloc[-1]: base_risk -= 1
    return int(min(max(round(base_risk), 1), 10))

def gerar_comentario(df, risco, tendencia, vcp):
    comentario = f"Tendência: {tendencia}. "
    if df['momentum_up'].iloc[-1] and df['rompe_resistencia'].iloc[-1]:
        comentario += "Sinal de força técnica detectado (momentum + rompimento). "
    elif df['momentum_up'].iloc[-1]:
        comentario += "Momentum positivo recente. "
    elif df['rompe_resistencia'].iloc[-1]:
        comentario += "Rompimento de resistência detectado. "
    else:
        comentario += "Sem sinais fortes de entrada. "
    if vcp:
        comentario += " | Padrão VCP detectado 🔍"

    PP, suportes, resistencias = calcular_pivot_points(df)
    preco_atual = df['Close'].iloc[-1]
    comentario += f" Preço atual: {preco_atual:.2f}"
    comentario += " | Suportes: " + ', '.join(f"{s:.2f} ({((s - preco_atual)/preco_atual)*100:+.2f}%)" for s in suportes)
    comentario += " | Resistências: " + ', '.join(f"{r:.2f} ({((r - preco_atual)/preco_atual)*100:+.2f}%)" for r in resistencias)
    return comentario

# ---------------------- INTERFACE STREAMLIT ----------------------


st.title("🚀 Resultados com IA e Filtros Técnicos")

# --- SIDEBAR CONFIG ---
threshold = st.sidebar.slider("⚡ Limite de momentum", 0.01, 0.2, 0.07)
dias_breakout = st.sidebar.slider("📈 Breakout da máxima dos últimos X dias", 10, 60, 20)
lookback = st.sidebar.slider("📊 Candles recentes analisados", 3, 10, 5)
sinal = st.sidebar.selectbox("🎯 Filtrar por sinal", ["Todos", "Ambos", "Momentum", "Breakout"])
performance = st.sidebar.selectbox("📊 Filtro de desempenho", [
    "Quarter Up", "Quarter +10%", "Quarter +20%", "Quarter +30%", "Quarter +50%",
    "Half Up", "Half +10%", "Half +20%", "Half +30%", "Half +50%", "Half +100%",
    "Year Up", "Year +10%", "Year +20%", "Year +30%", "Year +50%", "Year +100%",
    "Year +200%", "Year +300%", "Year +500%"
], index=15)
mostrar_vcp = st.sidebar.checkbox("🔎 Mostrar apenas ativos com padrão VCP", value=False, key="checkbox_vcp")
executar = st.sidebar.button("🔍 Iniciar análise")
ticker_manual = st.sidebar.text_input("📌 Ver gráfico de um ticker específico (ex: AAPL)", key="textinput_ticker_manual").upper()

if executar:
    st.session_state.recomendacoes = []
    screener = Overview()
    screener.set_filter(filters_dict={"Performance": performance, "Average Volume": "Over 300K"})
    tickers = screener.screener_view()['Ticker'].tolist()

    total_momentum = 0
    total_breakout = 0
    total_ambos = 0

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = calcular_indicadores(df, dias_breakout, threshold)

            momentum_cond = df['momentum_up'].iloc[-lookback:].any()
            breakout_cond = df['rompe_resistencia'].iloc[-lookback:].any()
            ambos_cond = momentum_cond and breakout_cond

            vcp_detectado = detectar_vcp(df)
            if mostrar_vcp and not vcp_detectado:
                continue

            match sinal:
                case "Momentum": cond = momentum_cond
                case "Breakout": cond = breakout_cond
                case "Ambos": cond = ambos_cond
                case _: cond = True

            if not cond: continue

            nome = yf.Ticker(ticker).info.get("shortName", ticker)
            risco = avaliar_risco(df)
            tendencia = classificar_tendencia(df['Close'].tail(20))
            comentario = gerar_comentario(df, risco, tendencia, vcp_detectado)

            st.subheader(f"{ticker} - {nome}")
            st.plotly_chart(plot_ativo(df, ticker, nome, vcp_detectado), use_container_width=True)
            st.markdown(f"**🧠 Análise IA:** {comentario}")
            st.markdown(f"**📉 Risco (1–10):** `{risco}` — **Tendência:** `{tendencia}`")

            st.session_state.recomendacoes.append({
                "Ticker": ticker, "Empresa": nome, "Risco": risco,
                "Tendência": tendencia, "Comentário": comentario
            })
        except Exception as e:
            st.warning(f"Erro com {ticker}: {e}")

    if st.session_state.recomendacoes:
        st.subheader("📋 Tabela Final de Recomendações")
        df_final = pd.DataFrame(st.session_state.recomendacoes).sort_values(by="Risco")
        st.dataframe(df_final, use_container_width=True)
        st.download_button("⬇️ Baixar CSV", df_final.to_csv(index=False).encode(), file_name="recomendacoes_ia.csv")

# Gráfico individual
if ticker_manual:
    df = yf.download(ticker_manual, period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = calcular_indicadores(df, dias_breakout, threshold)
    vcp_detectado = detectar_vcp(df)
    nome = yf.Ticker(ticker_manual).info.get("shortName", ticker_manual)
    risco = avaliar_risco(df)
    tendencia = classificar_tendencia(df['Close'].tail(20))
    comentario = gerar_comentario(df, risco, tendencia, vcp_detectado)
    st.subheader(f"{ticker_manual} - {nome}")
    st.plotly_chart(plot_ativo(df, ticker_manual, nome, vcp_detectado), use_container_width=True)
    st.markdown(f"**🧠 Análise IA:** {comentario}")
    st.markdown(f"**📉 Risco (1–10):** `{risco}` — **Tendência:** `{tendencia}`")
