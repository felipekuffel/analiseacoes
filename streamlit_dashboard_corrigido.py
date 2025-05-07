import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from finvizfinance.screener.overview import Overview
import plotly.graph_objects as go
import plotly.express as px
import time
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

def get_nome_empresa(ticker):
    try:
        return yf.Ticker(ticker).fast_info.get("shortName", ticker)
    except Exception:
        return ticker

# --- Função de earnings detalhado ---
def get_earnings_info_detalhado(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        if ticker_obj.calendar is not None and not ticker_obj.calendar.empty:
            if 'Earnings Date' in ticker_obj.calendar.index:
                earnings_date = ticker_obj.calendar.loc['Earnings Date'][0]
                if isinstance(earnings_date, pd.Timestamp) and not pd.isna(earnings_date):
                    now = pd.Timestamp.now(tz="UTC").tz_convert("America/New_York")
                    delta = (earnings_date - now).days
                    data_str = earnings_date.strftime('%d %b %Y')
                    if delta >= 0:
                        return f"Próx: {data_str} (em {delta}d)", earnings_date, delta
                    else:
                        return f"Último: {data_str} (há {-delta}d)", earnings_date, delta
        return "Indisponível", None, None
    except Exception as e:
        return"Erro: {str(e)}", None, None


def plot_ativo(df, ticker, nome_empresa, vcp_detectado=False):
    df = df.tail(150).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df['index_str'] = df.index.strftime('%Y-%m-%d')

    df['pct_change'] = df['Close'].pct_change() * 100
    df['DataStr'] = df.index.strftime("%d %b")
    df["previousClose"] = df["Close"].shift(1)
    df["color"] = np.where(df["Close"] > df["previousClose"], "#2736e9", "#de32ae")
    df["Percentage"] = df["Volume"] * 100 / df['Volume'].sum()

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "xy"}]],
        vertical_spacing=0.02,
        shared_xaxes=True
    )

    hovertext = df.apply(lambda row: f"{row['DataStr']}<br>Open: {row['Open']:.2f}<br>High: {row['High']:.2f}<br>Low: {row['Low']:.2f}<br>Close: {row['Close']:.2f}<br>Variação: {row['pct_change']:.2f}%" if pd.notna(row['pct_change']) else row['DataStr'], axis=1)

   # Médias móveis (primeiro, para ficarem atrás)
    fig.add_trace(go.Scatter(x=df['index_str'], y=df['SMA50'], mode='lines',
                            line=dict(color='rgba(0, 153, 255, 0.42)', width=1), name='SMA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['index_str'], y=df['EMA20'], mode='lines',
                            line=dict(color='rgba(0,255,0,0.4)', width=1), name='EMA20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['index_str'], y=df['SMA150'], mode='lines',
                            line=dict(color='rgba(255,165,0,0.4)', width=1), name='SMA150'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['index_str'], y=df['SMA200'], mode='lines',
                            line=dict(color='rgba(253, 76, 76, 0.4)', width=1), name='SMA200'), row=1, col=1)

    # OHLC (candles) por último para ficar por cima
    fig.add_trace(go.Ohlc(
        x=df['index_str'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color="#2736e9", decreasing_line_color="#de32ae", line_width=3,
        showlegend=False, text=hovertext, hoverinfo='text'), row=1, col=1)

    

    df_up = df[df['momentum_up']]
    df_rompe = df[df['rompe_resistencia']]
    fig.add_trace(go.Scatter(x=df_up['index_str'], y=df_up['High'] * 1.03, mode='markers', marker=dict(symbol='diamond', color='violet', size=6), name='Momentum Up'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_rompe['index_str'], y=df_rompe['High'] * 1.03, mode='markers', marker=dict(symbol='triangle-up', color='lime', size=6), name='Rompimento'), row=1, col=1)

    if vcp_detectado:
        last_index = df['index_str'].iloc[-1]
        last_price = df['Close'].iloc[-1]
        fig.add_trace(go.Scatter(x=[last_index], y=[last_price * 1.06], mode='markers', marker=dict(symbol='star-diamond', color='magenta', size=8), name='Padrão VCP', text=hovertext, hoverinfo='x+text'), row=1, col=1)

    fig.add_trace(go.Bar(x=df['index_str'], y=df['Volume'], text=df['Percentage'], marker_line_color=df['color'], marker_color=df['color'], name="Volume", texttemplate="%{text:.2f}%", hoverinfo="x+y", textfont=dict(color="white")), row=2, col=1)
    fig.add_trace(go.Bar(x=df['index_str'], y=df['momentum'], marker=dict(color=['rgba(23, 36, 131, 0.5)' if m > 0 else 'rgba(84, 14, 77, 0.50)' for m in df['momentum']], line=dict(width=0)), name='Momentum'), row=3, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)

    pct_text = f" ({df['pct_change'].iloc[-1]:+.2f}%)"
    fig.add_hline(y=df['Close'].iloc[-1], line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot'), row=1, col=1)
    pct_price = df['Close'].iloc[-1]

    fig.update_layout(
        xaxis=dict(type='category'),
        xaxis2=dict(type='category'),
        xaxis3=dict(type='category'),
        title=f"{ticker} - {nome_empresa} - {pct_price:.2f}{pct_text}",
        template='plotly_dark',
        height=900,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        yaxis=dict(title='', side='right', type='linear', showgrid=False, zeroline=False),
        yaxis2=dict(side='right', showgrid=False, zeroline=False),
        yaxis3=dict(side='right', showgrid=False, zeroline=False),
        legend=dict(x=1.05, y=1, traceorder='reversed', font_size=12),
        bargap=0.1
    )

    # --- FLAT BASE (conforme já estava implementado) ---
    zonas_flat = []
    i = 0
    while i < len(df) - 14:
        max_candles = 90
        min_candles = 14
        j = i + min_candles

        base_salva = None

        while j < len(df) and (j - i) <= max_candles:
            sub_df = df.iloc[i:j]
            high = sub_df['High'].max()
            low = sub_df['Low'].min()
            diff_pct = (high - low) / high * 100

            if diff_pct > 20:
                break

            if (j - i) >= min_candles:
                inicio = sub_df['index_str'].iloc[0]
                fim = sub_df['index_str'].iloc[-1]
                resistencia = round(high, 2)
                suporte = round(low, 2)
                duracao = j - i
                base_salva = (inicio, fim, resistencia, suporte, duracao)

            j += 1

        if base_salva:
            zonas_flat.append(base_salva)
            i = j
        else:
            i += 1

    for inicio, fim, resistencia, suporte, duracao in zonas_flat:
        fig.add_trace(go.Scatter(x=[inicio, inicio, fim, fim], y=[suporte, resistencia, resistencia, suporte], fill="toself", fillcolor="rgba(255, 255, 255, 0)", line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[inicio, fim], y=[resistencia, resistencia], mode="lines", line=dict(color="green", width=2, dash="dot"), hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[inicio, fim], y=[suporte, suporte], mode="lines", line=dict(color="green", width=2, dash="dot"), hoverinfo="skip", showlegend=False), row=1, col=1)
        variacao_pct = ((resistencia - suporte) / resistencia) * 100
        fig.add_annotation(x=inicio, y=resistencia, text=f"{resistencia:.2f} | {variacao_pct:.1f}% | {duracao} d ", showarrow=False, font=dict(color="green", size=10), bgcolor="rgba(255, 255, 255, 0)", yanchor="bottom", xanchor="left")
        # Anotação inferior: suporte
        fig.add_annotation(x=inicio, y=suporte,text=f"{suporte:.2f}",showarrow=False,font=dict(color="green", size=10),bgcolor="rgba(255, 255, 255, 0)",yanchor="top", xanchor="left")


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
    
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA150'] = df['Close'].rolling(150).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    centro = ((df['High20'] + df['Low20']) / 2 + df['Close'].rolling(length).mean()) / 2
    df['linreg_close'] = pine_linreg(df['Close'], length)
    df['momentum'] = df['linreg_close'] - centro
    df['momentum_up'] = (df['momentum'].shift(1) <= 0) & (df['momentum'] > momentum_threshold)
    df['rompe_resistencia'] = df['Close'] > df['High20']
    df['suporte'] = df['Low'].rolling(length).min()
    return df


def detectar_vcp(df):
    if 'Volume' not in df.columns or len(df) < 40:
        return False

    closes = df['Close']
    highs = df['High']
    lows = df['Low']
    volumes = df['Volume']
    sma50 = closes.rolling(50).mean()

    # 1. Dois pivôs descendentes
    max1 = highs[-40:-20].max()
    max2 = highs[-20:].max()
    if pd.isna(max1) or pd.isna(max2) or not (max1 > max2):
        return False

    min1 = lows[-40:-20].min()
    min2 = lows[-20:].min()
    if pd.isna(min1) or pd.isna(min2) or not (min1 < min2):
        return False

    # 2. Volume médio geral caindo
    vol_ant = volumes[-40:-20].mean()
    vol_rec = volumes[-20:].mean()
    if pd.isna(vol_ant) or pd.isna(vol_rec) or not (vol_ant > vol_rec):
        return False

    # 3. Range (amplitude) caindo
    range_ant = (highs[-40:-20] - lows[-40:-20]).mean()
    range_rec = (highs[-20:] - lows[-20:]).mean()
    if pd.isna(range_ant) or pd.isna(range_rec) or not (range_ant > range_rec):
        return False

    # 4. Preço ao menos na média de 50
    if pd.isna(sma50.iloc[-1]) or closes.iloc[-1] < sma50.iloc[-1] * 0.97:
        return False

    return True

    min1 = lows[-60:-40].min()
    min2 = lows[-40:-20].min()
    min3 = lows[-20:].min()
    if not (min1 < min2 < min3):
        return False

    # 2. Volume médio caindo
    vol1 = volumes[-60:-40].mean()
    vol2 = volumes[-40:-20].mean()
    vol3 = volumes[-20:].mean()
    if not (vol1 > vol2 > vol3):
        return False

    # 3. Redução do range (amplitude)
    range1 = (highs[-60:-40] - lows[-60:-40]).mean()
    range2 = (highs[-40:-20] - lows[-40:-20]).mean()
    range3 = (highs[-20:] - lows[-20:]).mean()
    if not (range1 > range2 > range3):
        return False

    # 4. Preço acima da média de 50 períodos
    if closes.iloc[-1] < sma50.iloc[-1]:
        return False

    return True

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


st.title("🚀🔍")

threshold = st.sidebar.slider("⚡ Limite de momentum", 0.01, 0.2, 0.07)
dias_breakout = st.sidebar.slider("\U0001F4C8 Breakout da máxima dos últimos X dias", 10, 60, 20)
lookback = st.sidebar.slider("\U0001F4CA Candles recentes analisados", 3, 10, 5)
sinal = st.sidebar.selectbox("\U0001F3AF Filtrar por sinal", ["Todos", "Ambos", "Momentum", "Breakout"])
performance = st.sidebar.selectbox("\U0001F4CA Filtro de desempenho", [
    "Quarter Up", "Quarter +10%", "Quarter +20%", "Quarter +30%", "Quarter +50%",
    "Half Up", "Half +10%", "Half +20%", "Half +30%", "Half +50%", "Half +100%",
    "Year Up", "Year +10%", "Year +20%", "Year +30%", "Year +50%", "Year +100%",
    "Year +200%", "Year +300%", "Year +500%"], index=15)
mostrar_vcp = st.sidebar.checkbox("\U0001F50E Mostrar apenas ativos com padrão VCP", value=False, key="checkbox_vcp")
ordenamento_mm = st.sidebar.checkbox("\U0001F4D0 EMA20 > SMA50 > SMA150 > SMA200", value=False)
sma200_crescente = st.sidebar.checkbox("\U0001F4C8 SMA200 maior que há 30 dias", value=False)
executar = st.sidebar.button("\U0001F50D Iniciar análise")
ticker_manual = st.sidebar.text_input("\U0001F4CC Ver gráfico de um ticker específico (ex: AAPL)", key="textinput_ticker_manual").upper()

if executar:
    st.session_state.recomendacoes = []
    screener = Overview()
    screener.set_filter(filters_dict={"Performance": performance, "Average Volume": "Over 300K"})
    tickers = screener.screener_view()['Ticker'].tolist()

    progress = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"\U0001F50D Analisando {ticker} ({i+1}/{len(tickers)})...")
        try:
            df = yf.download(ticker, period="18mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = calcular_indicadores(df, dias_breakout, threshold)

            if ordenamento_mm:
                if not (df['EMA20'].iloc[-1] > df['SMA50'].iloc[-1] > df['SMA150'].iloc[-1] > df['SMA200'].iloc[-1]):
                    continue

            if sma200_crescente:
                if len(df) < 30 or df['SMA200'].iloc[-1] <= df['SMA200'].iloc[-30]:
                    continue

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

            if not cond:
                continue

            nome = yf.Ticker(ticker).info.get("shortName", ticker)
            risco = avaliar_risco(df)
            tendencia = classificar_tendencia(df['Close'].tail(20))
            comentario = gerar_comentario(df, risco, tendencia, vcp_detectado)
            earnings_str, _, _ = get_earnings_info_detalhado(ticker)

            st.subheader(f"{ticker} - {nome}")
            with st.spinner(f"Carregando gráfico de {ticker}..."):
                fig = plot_ativo(df, ticker, nome, vcp_detectado)
                st.plotly_chart(fig, use_container_width=True, key=f"plot_{ticker}")
            st.markdown(f"📅 {earnings_str}")
            st.markdown(f"**\U0001F9E0 Análise IA:** {comentario}")
            st.markdown(f"**📉 Risco (1–10):** {risco} — **Tendência:** {tendencia}")

            st.session_state.recomendacoes.append({
                "Ticker": ticker,
                "Empresa": nome,
                "Risco": risco,
                "Tendência": tendencia,
                "Comentário": comentario,
                "Earnings": earnings_str
            })

        except Exception as e:
            st.warning(f"Erro com {ticker}: {e}")

        progress.progress((i + 1) / len(tickers))

    status_text.empty()
    progress.empty()

    if st.session_state.recomendacoes:
        st.subheader("📋 Tabela Final de Recomendações")
        df_final = pd.DataFrame(st.session_state.recomendacoes).sort_values(by="Risco")
        st.dataframe(df_final, use_container_width=True)
        st.download_button("⬇️ Baixar CSV", df_final.to_csv(index=False).encode(), file_name="recomendacoes_ia.csv")

# Gráfico individual
if ticker_manual:
    df = yf.download(ticker_manual, period="18mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = calcular_indicadores(df, dias_breakout, threshold)
    vcp_detectado = detectar_vcp(df)
    nome = yf.Ticker(ticker_manual).info.get("shortName", ticker_manual)
    risco = avaliar_risco(df)
    tendencia = classificar_tendencia(df['Close'].tail(20))
    comentario = gerar_comentario(df, risco, tendencia, vcp_detectado)
    earnings_str, _, _ = get_earnings_info_detalhado(ticker_manual)

    st.subheader(f"{ticker_manual} - {nome}")
    with st.spinner(f"Carregando gráfico de {ticker_manual}..."):
        fig = plot_ativo(df, ticker_manual, nome, vcp_detectado)
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{ticker_manual}_manual")
    st.markdown(f"📅 {earnings_str}")
    st.markdown(f"**\U0001F9E0 Análise IA:** {comentario}")
    st.markdown(f"**📉 Risco (1–10):** {risco} — **Tendência:** {tendencia}")
