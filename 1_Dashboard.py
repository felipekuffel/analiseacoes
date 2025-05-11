    #st.title("Dashboard de Análise Técnica")
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from finvizfinance.screener.overview import Overview
import plotly.graph_objects as go
import plotly.express as px
import time
from plotly.subplots import make_subplots
import datetime
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from cryptography.hazmat.primitives import serialization
from streamlit_autorefresh import st_autorefresh
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import re
from finvizfinance.screener.overview import Overview
import requests



st.set_page_config(layout="wide")



# --- Função de earnings detalhado ---
def get_earnings_info_detalhado(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        calendar = ticker_obj.calendar

        if isinstance(calendar, dict) or isinstance(calendar, pd.Series):
            earnings = calendar.get("Earnings Date", None)

            # Se for uma lista de datas, pegamos a primeira futura
            if isinstance(earnings, list) and earnings:
                earnings = earnings[0]
            if isinstance(earnings, (pd.Timestamp, datetime.datetime, datetime.date)):
                earnings_date = pd.to_datetime(earnings).tz_localize("America/New_York") if pd.to_datetime(earnings).tzinfo is None else pd.to_datetime(earnings)
                now = pd.Timestamp.now(tz="America/New_York")
                delta = (earnings_date - now).days
                data_str = earnings_date.strftime('%d %b %Y')
                if delta >= 0:
                    return f" {data_str} (em {delta}d)", earnings_date, delta
                else:
                    return f"Último: {data_str} (há {-delta}d)", earnings_date, delta

        return "Indisponível", None, None
    except Exception as e:
        return f"Erro: {e}", None, None


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
        increasing_line_color="#2736e9", decreasing_line_color="#de32ae", line_width=2.5,
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
    showlegend=False,  # ❌ desabilita a legenda
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

    try:
        earnings_df = yf.Ticker(ticker).quarterly_financials.T
        earnings_dates = [d.strftime('%Y-%m-%d') for d in earnings_df.index]
        print("Datas earnings:", earnings_dates)
        print("Datas no gráfico:", df['index_str'].tolist())


        for date in earnings_dates:
            if date in df['index_str'].values:
                for date in earnings_dates:
                    if date in df['index_str'].values:
                        fig.add_shape(
                            type="line",
                            x0=date, x1=date,
                            yref="paper", y0=0, y1=1,
                            line=dict(color="rgba(128,128,128,0.5)", dash="dot", width=1),
                        )
                        fig.add_annotation(
                            x=date, y=1,
                            xref="x", yref="paper",
                            text="", showarrow=False,
                            font=dict(color="rgba(128,128,128,0.5)", size=10),
                            xanchor="left"
                        )

    except Exception as e:
        print("Erro ao adicionar marcações de earnings:", e)

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


# --- Nova função de risco aprimorada ---
def avaliar_risco(df):
    preco_atual = df['Close'].iloc[-1]
    suporte = df['Low'].rolling(20).min().iloc[-1]
    resistencia = df['High'].rolling(20).max().iloc[-1]
    risco = 5  # ponto base

    # ATR (volatilidade)
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    atr = df['TR'].rolling(14).mean().iloc[-1]
    if atr / preco_atual > 0.05:
        risco += 1  # ativo volátil
    else:
        risco -= 1  # ativo estável

    # Distância até suporte
    if (preco_atual - suporte) / preco_atual > 0.05:
        risco += 1

    # Proximidade da resistência
    if (resistencia - preco_atual) / preco_atual < 0.03:
        risco += 1

    # Preço abaixo da média de 200
    if preco_atual < df['SMA200'].iloc[-1]:
        risco += 1

    # Quedas consecutivas nos últimos 30 dias
    closes = df['Close'].tail(30).reset_index(drop=True)
    quedas = sum(closes.diff() < 0)
    if quedas >= 3:
        risco += 1

    # Queda com volume alto nos últimos 30 dias
    recent_df = df.tail(30)
    media_volume = recent_df['Volume'].mean()
    dias_queda_volume_alto = recent_df[(recent_df['Close'] < recent_df['Close'].shift(1)) & (recent_df['Volume'] > media_volume)]
    if not dias_queda_volume_alto.empty:
        risco += 1

    # Rompimento de topo com volume alto
    if df['rompe_resistencia'].iloc[-1] and df['Volume'].iloc[-1] > df['Volume'].rolling(20).mean().iloc[-1]:
        risco -= 1

    # Médias alinhadas
    if df['EMA20'].iloc[-1] > df['SMA50'].iloc[-1] > df['SMA150'].iloc[-1] > df['SMA200'].iloc[-1]:
        risco -= 1

    return int(min(max(round(risco), 1), 10))

# --- Função de análise IA aprimorada ---
def gerar_comentario(df, risco, tendencia, vcp):
    comentario = "📊 Ativo em zona de observação técnica"

    sinais = []
    if df['momentum_up'].iloc[-1]:
        sinais.append("Momentum")
    if df['rompe_resistencia'].iloc[-1]:
        sinais.append("Rompimento")
    if vcp:
        sinais.append("Padrão VCP")

    if sinais:
        comentario += f"\n📈 Sinais técnicos: {', '.join(sinais)}"

    return comentario


def get_quarterly_growth_table_yfinance(ticker):
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.quarterly_financials.T

    if df.empty or "Total Revenue" not in df.columns or "Net Income" not in df.columns:
        return None

    df = df[["Total Revenue", "Net Income"]].dropna()
    df.sort_index(ascending=False, inplace=True)

    rows = []
    for i in range(4):
        try:
            atual = df.iloc[i]
            trimestre_data = df.index[i].date()
            receita_atual = atual["Total Revenue"]
            lucro_atual = atual["Net Income"]

            receita_pct = None
            lucro_pct = None
            if i + 4 < len(df):
                receita_ant = df.iloc[i + 4]["Total Revenue"]
                lucro_ant = df.iloc[i + 4]["Net Income"]
                if receita_ant:
                    receita_pct = (receita_atual - receita_ant) / receita_ant * 100
                if lucro_ant:
                    lucro_pct = (lucro_atual - lucro_ant) / abs(lucro_ant) * 100

            margem = (lucro_atual / receita_atual) * 100 if receita_atual else None

            def fmt_pct(val):
                if val is None:
                    return ""
                emoji = " 🚀" if val > 18 else ""
                return f"{val:+.1f}%{emoji}"

            rows.append({
                "Trimestre": trimestre_data.strftime("%b %Y"),
                "Receita (B)": f"${receita_atual / 1e9:.2f}B",
                "Receita YoY": fmt_pct(receita_pct),
                "Lucro (B)": f"${lucro_atual / 1e9:.2f}B",
                "Lucro YoY": fmt_pct(lucro_pct),
                "Margem (%)": f"{margem:.1f}%" if margem is not None else ""
            })
        except Exception:
            continue

    df_final = pd.DataFrame(rows).set_index("Trimestre")
    return df_final









# ---------------------- INTERFACE STREAMLIT ----------------------


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
# Elementos finais da barra lateral
st.sidebar.markdown("---")


# ✅ Botão de logout sempre visível
if st.sidebar.button("🚪 Sair"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def inserir_preco_no_meio(niveis: list, preco: float) -> pd.DataFrame:
    df = pd.DataFrame(niveis)
    df["Valor"] = df["Valor"].map(lambda x: float(f"{x:.2f}"))
    df["DistânciaReal"] = (df["Valor"] - preco) / preco
    df["Distância"] = (df["DistânciaReal"] * 100).map("{:+.2f}%".format)
    df["Valor"] = df["Valor"].map("{:.2f}".format)
    df.drop(columns=["DistânciaReal"], inplace=True)
    df = df.dropna(how="any")

    df_temp = df.copy()
    df_temp["Valor_float"] = df_temp["Valor"].astype(float)

    inserido = False
    linhas_ordenadas = []

    for _, row in df_temp.sort_values(by="Valor_float", ascending=False).iterrows():
        if not inserido and float(row["Valor"]) < preco:
            linhas_ordenadas.append({
                "Nível": "💰 Preço Atual",
                "Valor": f"{preco:.2f}",
                "Distância": "{:+.2f}%".format(0)
            })
            inserido = True
        linhas_ordenadas.append(row[["Nível", "Valor", "Distância"]].to_dict())

    if not inserido:
        linhas_ordenadas.append({
            "Nível": "💰 Preço Atual",
            "Valor": f"{preco:.2f}",
            "Distância": "{:+.2f}%".format(0)
        })

    df_final = pd.DataFrame(linhas_ordenadas).set_index("Nível")
    return df_final

if executar:
    st.session_state.recomendacoes = []

    status_text = st.empty()
    progress_bar = st.progress(0)
    f = io.StringIO()

    with redirect_stdout(f), redirect_stderr(f):
        with st.spinner("🔄 Buscando ativos..."):
            screener = Overview()
            screener.set_filter(filters_dict={"Performance": performance, "Average Volume": "Over 300K"})
            tickers_df = screener.screener_view()

    # Captura o log impresso pela finvizfinance
    log_output = f.getvalue()
    matches = re.findall(r'loading page.*?\[(.*?)\].*?(\d+)/(\d+)', log_output)

    # Atualiza progresso com base na última linha de progresso (se houver)
    if matches:
        current, total = map(int, matches[-1][1:])
        percent = current / total
        progress_bar.progress(percent)
        status_text.text(f"📄 Página {current} de {total} ({int(percent * 100)}%)")
    else:
        status_text.text("✅ Ativos carregados.")

    # Não exibe o texto bruto das páginas
    tickers = tickers_df['Ticker'].tolist()
    st.success(f"✅ {len(tickers)} ativos carregados.")

    # --- Análise técnica por ticker
    progress = st.progress(0)
    status_text = st.empty()

    for i, ticker in enumerate(tickers):
        status_text.text(f"🔍 Analisando {ticker} ({i+1}/{len(tickers)})...")
        try:
            df = yf.download(ticker, period="18mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df = calcular_indicadores(df, dias_breakout, threshold)

            if ordenamento_mm and not (df['EMA20'].iloc[-1] > df['SMA50'].iloc[-1] > df['SMA150'].iloc[-1] > df['SMA200'].iloc[-1]):
                continue

            if sma200_crescente and (len(df) < 30 or df['SMA200'].iloc[-1] <= df['SMA200'].iloc[-30]):
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

            with st.container():
                st.subheader(f"{ticker} - {nome}")
                col1, col2 = st.columns([3, 2])

                with col1:
                    with st.spinner(f"📊 Carregando gráfico de {ticker}..."):
                        fig = plot_ativo(df, ticker, nome, vcp_detectado)
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{ticker}")

                with col2:
                    st.markdown(comentario)
                    st.markdown(f"📅 **Resultado:** {earnings_str}")
                    st.markdown(f"📉 **Risco:** `{risco}`")

                    preco = df["Close"].iloc[-1]
                    PP, suportes, resistencias = calcular_pivot_points(df)
                    dists_resist = [(r, ((r - preco) / preco) * 100) for r in resistencias]
                    dists_suportes = [(s, ((s - preco) / preco) * 100) for s in suportes]

                    resist_ordenado = sorted([r for r in dists_resist if r[0] > preco], key=lambda x: x[0])[:3]
                    suporte_ordenado = sorted([s for s in dists_suportes if s[0] < preco], key=lambda x: -x[0])[:3]

                    niveis = []

                    for i, (valor, _) in enumerate(resist_ordenado):
                        niveis.append({"Nível": f"🔺 {i + 1}ª Resistência", "Valor": valor})

                    for i, (valor, _) in enumerate(suporte_ordenado):
                        niveis.append({"Nível": f"🔻 {i + 1}º Suporte", "Valor": valor})

                    swing_high = df["High"].rolling(40).max().iloc[-1]
                    swing_low = df["Low"].rolling(40).min().iloc[-1]
                    retracao_382 = swing_high - (swing_high - swing_low) * 0.382
                    retracao_618 = swing_high - (swing_high - swing_low) * 0.618

                    indicadores = {
                        "SMA 20": df["SMA20"].iloc[-1],
                        "SMA 50": df["SMA50"].iloc[-1],
                        "SMA 150": df["SMA150"].iloc[-1],
                        "SMA 200": df["SMA200"].iloc[-1],
                        "Máxima 52s": df["High"].rolling(252).max().iloc[-1],
                        "Mínima 52s": df["Low"].rolling(252).min().iloc[-1],
                        "Retração 38.2% (últ. 40d)": retracao_382,
                        "Retração 61.8% (últ. 40d)": retracao_618
                    }

                    for nome_ind, valor in indicadores.items():
                        if "SMA" in nome_ind:
                            nivel_nome = f"🟣 {nome_ind}"
                        elif "Retração" in nome_ind:
                            nivel_nome = f"📏 {nome_ind}"
                        elif "Máxima" in nome_ind:
                            nivel_nome = f"📈 {nome_ind}"
                        elif "Mínima" in nome_ind:
                            nivel_nome = f"📉 {nome_ind}"
                        else:
                            nivel_nome = nome_ind
                        niveis.append({"Nível": nivel_nome, "Valor": valor})

                    df_niveis = inserir_preco_no_meio(niveis, preco)

                    def highlight_niveis(row):
                        nivel = row.name
                        if "Preço Atual" in nivel:
                            return ["background-color: #fff3b0; font-weight: bold;"] * len(row)
                        elif "🔺" in nivel:
                            return ["color: #1f77b4; font-weight: bold;"] * len(row)
                        elif "🔻" in nivel:
                            return ["color: #2ca02c; font-weight: bold;"] * len(row)
                        elif any(tag in nivel for tag in ["🟣", "📏", "📈", "📉"]):
                            return ["color: #9467bd; font-style: italic;"] * len(row)
                        return [""] * len(row)

                    styled_table = df_niveis.style.apply(highlight_niveis, axis=1)
                    st.dataframe(styled_table, use_container_width=True, height=565)
                    df_resultado = get_quarterly_growth_table_yfinance(ticker)
                    if df_resultado is not None:
                        st.markdown("📊 **Histórico Trimestral (YoY)**")
                        st.table(df_resultado)
                    else:
                        st.warning("❌ Histórico de crescimento YoY não disponível.")


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
    col1, col2 = st.columns([3, 2])


    with col1:
        with st.spinner(f"Carregando gráfico de {ticker_manual}..."):
            fig = plot_ativo(df, ticker_manual, nome, vcp_detectado)
            st.plotly_chart(fig, use_container_width=True, key=f"plot_{ticker_manual}_manual")

    with col2:
        st.markdown(comentario)
        st.markdown(f"📅 Resultado: {earnings_str}")

        preco = df["Close"].iloc[-1]
        PP, suportes, resistencias = calcular_pivot_points(df)
        dists_resist = [(r, ((r - preco) / preco) * 100) for r in resistencias]
        dists_suportes = [(s, ((s - preco) / preco) * 100) for s in suportes]

        resist_ordenado = sorted([r for r in dists_resist if r[0] > preco], key=lambda x: x[0])[:3]
        suporte_ordenado = sorted([s for s in dists_suportes if s[0] < preco], key=lambda x: -x[0])[:3]

        niveis = []

        for i, (valor, _) in enumerate(resist_ordenado):
            niveis.append({"Nível": f"🔺 {i + 1}ª Resistência", "Valor": valor})

        for i, (valor, _) in enumerate(suporte_ordenado):
            niveis.append({"Nível": f"🔻 {i + 1}º Suporte", "Valor": valor})

        # Fibonacci retrações
        swing_high = df["High"].rolling(40).max().iloc[-1]
        swing_low = df["Low"].rolling(40).min().iloc[-1]
        retracao_382 = swing_high - (swing_high - swing_low) * 0.382
        retracao_618 = swing_high - (swing_high - swing_low) * 0.618

        indicadores = {
            "SMA 20": df["SMA20"].iloc[-1],
            "SMA 50": df["SMA50"].iloc[-1],
            "SMA 150": df["SMA150"].iloc[-1],
            "SMA 200": df["SMA200"].iloc[-1],
            "Máxima 52s": df["High"].rolling(252).max().iloc[-1],
            "Mínima 52s": df["Low"].rolling(252).min().iloc[-1],
            "Retração 38.2%": retracao_382,
            "Retração 61.8%": retracao_618
        }

        for nome, valor in indicadores.items():
            if "SMA" in nome:
                nivel_nome = f"🟣 {nome}"
            elif "Retração" in nome:
                nivel_nome = f"📏 {nome}"
            elif "Máxima" in nome:
                nivel_nome = f"📈 {nome}"
            elif "Mínima" in nome:
                nivel_nome = f"📉 {nome}"
            else:
                nivel_nome = nome
            niveis.append({"Nível": nivel_nome, "Valor": valor})

        niveis.append({"Nível": "💰 Preço Atual", "Valor": preco})

        # Remove itens incompletos ou nulos antes de criar o DataFrame
        niveis_filtrados = [n for n in niveis if n["Valor"] is not None and not pd.isna(n["Valor"])]
        df_niveis = pd.DataFrame(niveis_filtrados)
    
        df_niveis["DistânciaReal"] = (df_niveis["Valor"] - preco) / preco
        df_niveis["Distância"] = (df_niveis["DistânciaReal"] * 100).map("{:+.2f}%".format)
        df_niveis["Valor"] = df_niveis["Valor"].map("{:.2f}".format)
        df_niveis.sort_values(by="Valor", ascending=False, inplace=True)
        df_niveis.drop(columns=["DistânciaReal"], inplace=True)
        df_niveis.reset_index(drop=True, inplace=True)
        df_niveis = df_niveis[["Nível", "Valor", "Distância"]]
        df_niveis = df_niveis.replace(r"^\s*$", np.nan, regex=True)
        df_niveis = df_niveis.dropna(how="any")

        
        def highlight_niveis(row):
            nivel = row.name  # Agora usa o índice, não a coluna
            if "Preço Atual" in nivel:
                return ["background-color: #fff3b0; font-weight: bold;"] * len(row)
            elif "🔺" in nivel:
                return ["color: #1f77b4; font-weight: bold;"] * len(row)
            elif "🔻" in nivel:
                return ["color: #2ca02c; font-weight: bold;"] * len(row)
            elif any(tag in nivel for tag in ["🟣", "📏", "📈", "📉"]):
                return ["color: #9467bd; font-style: italic;"] * len(row)
            return [""] * len(row)




        # Exibe tabela
        df_niveis = df_niveis[["Nível", "Valor", "Distância"]]
        df_niveis.reset_index(drop=True, inplace=True)

        # Limpeza robusta de linhas vazias
        df_niveis = df_niveis.replace(r"^\s*$", np.nan, regex=True)
        df_niveis = df_niveis.dropna(how="any")
        df_niveis.reset_index(drop=True, inplace=True)

        df_niveis_styled = df_niveis.set_index("Nível").style.apply(highlight_niveis, axis=1)
        st.dataframe(df_niveis_styled, use_container_width=True, height=565)



        # 🔍 Crescimento Passado (Sales/EPS Q/Q)
        # Busca dados Finviz somente para o ticker manual
        df_resultado = get_quarterly_growth_table_yfinance(ticker_manual)
        if df_resultado is not None:
            st.markdown("📊 **Histórico Trimestral (YoY)**")
            st.table(df_resultado)
        else:
            st.warning("❌ Histórico de crescimento YoY não disponível.")