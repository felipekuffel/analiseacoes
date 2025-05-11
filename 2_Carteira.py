import streamlit as st
import pandas as pd
import numpy as np
from cryptography.hazmat.primitives import serialization
import re
import datetime
from firebase_admin import credentials, auth as admin_auth, db
import firebase_admin
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

try:
    key = st.secrets["firebase_admin"]["private_key"]
    serialization.load_pem_private_key(key.encode(), password=None)
except Exception as e:
    st.error(f"❌ Erro na chave privada: {e}")
    st.stop()

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(dict(st.secrets["firebase_admin"]))
        firebase_admin.initialize_app(cred, {
            "databaseURL": st.secrets["databaseURL"]
        })
    except Exception as e:
        st.error(f"Erro ao inicializar Firebase: {e}")
        st.stop()

if "user" not in st.session_state or "localId" not in st.session_state.user:
    st.error("Usuário não autenticado corretamente.")
    st.stop()

user_id = st.session_state.user["localId"]
ref = db.reference(f"carteiras/{user_id}/simulacoes")
finalizadas_ref = db.reference(f"carteiras/{user_id}/simulacoes_finalizadas")
simulacoes_salvas = ref.get()

if "simulacoes" not in st.session_state:
    st.session_state.simulacoes = simulacoes_salvas if simulacoes_salvas else []

def limpar_chaves_invalidas(obj, path="root"):
    if isinstance(obj, dict):
        novo = {}
        for k, v in obj.items():
            k_str = str(k)
            caminho_atual = f"{path}.{k_str}"
            if not k_str or re.search(r'[.$#[\]/]', k_str):
                continue
            novo[k_str] = limpar_chaves_invalidas(v, path=caminho_atual)
        return novo
    elif isinstance(obj, list):
        return [limpar_chaves_invalidas(item, path=f"{path}[{i}]") for i, item in enumerate(obj)]
    else:
        return obj

def registrar_venda(sim, preco_venda, qtd_vendida, data_venda):
    preco_venda = float(preco_venda)
    qtd_vendida = int(qtd_vendida)
    valor_total_venda = preco_venda * qtd_vendida

    # Calcula o custo com base nas compras reais, se houver
    if "compras_reais" in sim and sim["compras_reais"]:
        total_valor = sum([c["preco"] * c["qtd"] for c in sim["compras_reais"]])
        total_qtd = sum([c["qtd"] for c in sim["compras_reais"]])
        preco_medio = total_valor / total_qtd if total_qtd else 0
    else:
        preco_medio = sim["total_valor"] / sim["total_unidades"]

    valor_total_custo = preco_medio * qtd_vendida
    lucro = valor_total_venda - valor_total_custo
    lucro_pct = lucro / valor_total_custo * 100 if valor_total_custo else 0

    tipo = "🟡 Parcial" if qtd_vendida < sim.get("quantidade_real", sim["quantidade_restante"]) else "🔴 Total"

    venda = {
        "nome": sim["nome"],
        "data": data_venda,
        "preco_venda": preco_venda,
        "quantidade": qtd_vendida,
        "lucro": lucro,
        "lucro_pct": lucro_pct,
        "tipo": tipo
    }

    # Salva venda no Firebase
    vendas_anteriores = finalizadas_ref.get() or []
    vendas_anteriores.append(venda)
    finalizadas_ref.set(limpar_chaves_invalidas(vendas_anteriores))

    # Atualiza ou remove a simulação
    if tipo == "🔴 Total":
        st.session_state.simulacoes.remove(sim)
    else:
        sim["quantidade_real"] = max(0, sim.get("quantidade_real", 0) - qtd_vendida)
        sim["quantidade_restante"] = max(0, sim.get("quantidade_restante", 0) - qtd_vendida)

    # Atualiza Firebase
    ref.set(limpar_chaves_invalidas(st.session_state.simulacoes))

if 'edit_index' in st.session_state:
    sim = st.session_state.simulacoes[st.session_state.edit_index]
    nome_default = sim['nome']
    cotacao_default = sim['cotacao']
    venda_pct_default = sim['venda_pct']
    pl_total_default = sim['pl_total']
else:
    nome_default = "ACMR"
    cotacao_default = 28.87
    venda_pct_default = 17.0
    pl_total_default = 10000.0

with st.form("form_compras"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        nome_acao = st.text_input("🔹 Nome da Ação", nome_default)
    with col2:
        cotacao = st.number_input("💲 Cotação Inicial de Compra", value=cotacao_default, step=0.01, format="%.2f")
    with col3:
        venda_pct = st.number_input("🎯 % de Ganho para Venda", value=venda_pct_default, step=0.1, format="%.2f")
    with col4:
        pl_total = st.number_input("💼 Capital Total (PL)", value=pl_total_default, step=100.0)

    col_data = st.columns(1)[0]
    with col_data:
        data_simulacao = st.date_input("📅 Data da Simulação", value=datetime.date.today())


    st.markdown("---")
    st.subheader("📌 Configuração das Compras")
    compra_data = []
    for i, nome in enumerate(["COMPRA INICIAL", "COMPRA 2", "COMPRA 3"]):
        with st.expander(f"🛒 {nome}", expanded=(i == 0)):  # expandido por padrão só na inicial
            col1, col2, col3 = st.columns(3)
            with col1:
                subida = 0.0 if i == 0 else st.number_input("🔼 % de Subida", key=f"subida{i}", value=[0.0, 4.0, 10.0][i], step=0.1)
            with col2:
                pct_pl = st.number_input("📊 % do PL", key=f"pct_pl{i}", value=[8.0, 6.0, 6.0][i], step=0.1)
            with col3:
                stop = st.number_input("🛑 Stop (%)", key=f"stop{i}", value=[8.0, 8.0, 10.0][i], step=0.1)
            compra_data.append({"nome": nome, "subida_pct": subida, "pct_pl": pct_pl, "stop_pct": stop})


    enviado = st.form_submit_button("📥 Simular Compras")

if enviado:
    if 'edit_index' in st.session_state:
        del st.session_state.simulacoes[st.session_state.edit_index]
        del st.session_state.edit_index

    total_valor = 0
    total_unidades = 0
    linhas = []

    for i, compra in enumerate(compra_data):
        preco = cotacao * (1 + compra["subida_pct"] / 100)
        valor = pl_total * (compra["pct_pl"] / 100)
        unidades = valor / preco
        stop_preco = preco * (1 - compra["stop_pct"] / 100)
        risco_valor = (preco - stop_preco) * unidades
        risco_pct_pl = -risco_valor / pl_total * 100

        linhas.append([
            compra["nome"],
            f"${preco:.2f}",
            f"{compra['subida_pct']:.2f}%" if i > 0 else "Compra Inicial",
            f"${valor:,.2f}",
            f'{compra["pct_pl"]:.2f}%',
            f"{int(unidades)} UN",
            f'{compra["stop_pct"]:.2f}%',
            f"$ {stop_preco:.2f}",
            f"{risco_pct_pl:.2f}% PL",
            f"$ {-risco_valor:.2f}",
        ])

        total_valor += valor
        total_unidades += unidades

    preco_final = cotacao * (1 + venda_pct / 100)
    lucro = preco_final * total_unidades - total_valor
    lucro_pct = lucro / total_valor * 100
    lpl_pct = lucro / pl_total * 100

    df_tabela = pd.DataFrame(linhas, columns=[
        "Etapa", "ADD", "% PARA COMPRA", "COMPRA PL", "% PL COMPRA",
        "QTD", "STOP", "$ STOP", "RISCO", "$ RISCO"
    ])

    nova_simulacao = {
        "nome": nome_acao,
        "cotacao": cotacao,
        "venda_pct": venda_pct,
        "pl_total": pl_total,
        "data_simulacao": str(data_simulacao),  # <- campo novo
        "preco_final": preco_final,
        "lucro": lucro,
        "lucro_pct": lucro_pct,
        "lpl_pct": lpl_pct,
        "total_valor": total_valor,
        "total_unidades": total_unidades,
        "tabela": df_tabela.to_dict(),
        "quantidade_restante": int(total_unidades),
        "compras_reais": [
            {
                "etapa": "Inicial",
                "preco": cotacao,
                "qtd": int(pl_total * (compra_data[0]["pct_pl"] / 100) / cotacao),
                "data": str(data_simulacao)
            }
        ],
        "quantidade_real": int(pl_total * (compra_data[0]["pct_pl"] / 100) / cotacao),
        "preco_medio": cotacao
    }

    st.session_state.simulacoes.append(nova_simulacao)
    ref.set(limpar_chaves_invalidas(st.session_state.simulacoes))
    st.success("✅ Simulação salva com sucesso!")
    st.rerun()

st.markdown("---")
st.subheader("📈 Simulações em Aberto")

for idx, sim in enumerate(st.session_state.simulacoes):
    with st.expander(f"📈 {sim['nome']}  •  Alvo: +{sim['venda_pct']:.1f}%  •  Lucro: ${sim['lucro']:.2f}"):
        st.markdown(f"""
        <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 10px; font-size: 16px;'>
        <strong>Simulação para:</strong> {sim['nome']}  |  
        <strong>Meta de venda:</strong> +{sim['venda_pct']:.2f}% (alvo: $ {sim['preco_final']:.2f})  |  
        <strong>Qtd total:</strong> {int(sim['total_unidades'])} ações  |  
        <strong>Total investido:</strong> $ {sim['total_valor']:.2f}  |  
        <strong>Lucro estimado:</strong> $ {sim['lucro']:.2f} ({sim['lucro_pct']:.2f}%)  |  
        <strong>L/PL:</strong> {sim['lpl_pct']:.2f}%
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(pd.DataFrame(sim["tabela"], columns=sim["tabela"].keys()), use_container_width=True, hide_index=True)

        col_hist, col_botoes = st.columns([2, 3])

        with col_hist:
            if "compras_reais" in sim and sim["compras_reais"]:
                st.markdown("#### 📜 Histórico de Compras Reais")
                for i, c in enumerate(sim["compras_reais"]):
                    with st.container():
                        col1, col2, col3 = st.columns([4, 1, 1])
                        with col1:
                            st.markdown(
                                f"🛒 **Compra {i+1} ({c.get('etapa', '?')})** | "
                                f"📅 {c.get('data', '—')} • "
                                f"🔢 {c['qtd']} ações • "
                                f"💵 $ {c['preco']:.2f}"
                            )
                        with col2:
                            if st.button("✏️", key=f"editar_compra_{idx}_{i}"):
                                with st.form(f"form_editar_compra_{idx}_{i}"):
                                    nova_data = st.date_input("📅 Nova data", value=datetime.date.fromisoformat(c["data"]), key=f"edit_data_{idx}_{i}")
                                    novo_preco = st.number_input("💵 Novo preço", value=c["preco"], step=0.01, format="%.2f", key=f"edit_preco_{idx}_{i}")
                                    nova_qtd = st.number_input("🔢 Nova quantidade", value=c["qtd"], step=1, key=f"edit_qtd_{idx}_{i}")
                                    if st.form_submit_button("Salvar alteração"):
                                        sim["compras_reais"][i]["data"] = str(nova_data)
                                        sim["compras_reais"][i]["preco"] = novo_preco
                                        sim["compras_reais"][i]["qtd"] = nova_qtd
                                        total_qtd = sum([c["qtd"] for c in sim["compras_reais"]])
                                        total_valor = sum([c["preco"] * c["qtd"] for c in sim["compras_reais"]])
                                        sim["quantidade_real"] = total_qtd
                                        sim["preco_medio"] = total_valor / total_qtd if total_qtd else 0
                                        ref.set(limpar_chaves_invalidas(st.session_state.simulacoes))
                                        st.success("✅ Compra editada com sucesso!")
                                        st.rerun()
                        with col3:
                            if st.button("🗑", key=f"excluir_compra_{idx}_{i}"):
                                sim["compras_reais"].pop(i)
                                total_qtd = sum([c["qtd"] for c in sim["compras_reais"]])
                                total_valor = sum([c["preco"] * c["qtd"] for c in sim["compras_reais"]]) if sim["compras_reais"] else 0
                                sim["quantidade_real"] = total_qtd
                                sim["preco_medio"] = total_valor / total_qtd if total_qtd else 0
                                ref.set(limpar_chaves_invalidas(st.session_state.simulacoes))
                                st.success("🗑 Compra excluída com sucesso!")
                                st.rerun()

        with col_botoes:
            col_ed, col_del = st.columns([1, 1])
            with col_ed:
                if st.button(f"✏️ Editar {sim['nome']}", key=f"edit_{idx}"):
                    st.session_state.edit_index = idx
                    st.rerun()
            with col_del:
                if st.button(f"🗑 Excluir {sim['nome']}", key=f"del_{idx}"):
                    del st.session_state.simulacoes[idx]
                    ref.set(limpar_chaves_invalidas(st.session_state.simulacoes))
                    st.success("Simulação excluída com sucesso.")
                    st.rerun()

            col_venda, col_compra = st.columns(2)

            with col_venda:
                with st.form(f"venda_form_{idx}"):
                    st.markdown("### 💰 Registrar Venda")
                    disponivel = int(sim.get("quantidade_real", sim.get("quantidade_restante", 0)))
                    st.markdown(f"📦 Você possui **{disponivel} ações** disponíveis para venda.")
                    if 'preco_medio' in sim:
                        st.markdown(f"💰 Preço médio acumulado: **$ {sim['preco_medio']:.2f}**")
                    if 'compras_reais' in sim:
                        total_investido = sum([c['preco'] * c['qtd'] for c in sim['compras_reais']])
                        st.markdown(f"💸 Total investido nas compras reais: **$ {total_investido:.2f}**")
                    data_venda = st.date_input("📅 Data da venda", value=datetime.date.today(), key=f"data_venda_{idx}")
                    preco_venda = st.number_input("💲 Preço de venda", step=0.01, format="%.2f", key=f"preco_venda_{idx}")
                    qtd_vendida = st.number_input("🔢 Quantidade vendida", step=1, format="%d", key=f"qtd_vendida_{idx}", min_value=1, max_value=disponivel)
                    if st.form_submit_button("Confirmar Venda"):
                        registrar_venda(sim, preco_venda, qtd_vendida, str(data_venda))
                        st.success("✅ Venda registrada com sucesso!")
                        st.rerun()

            with col_compra:
                with st.form(f"form_add_compra_{idx}"):
                    st.markdown("### ➕ Registrar Compra Real")
                    etapa = st.selectbox("Etapa da compra", ["2", "3"], key=f"etapa_compra_{idx}")
                    etapa_idx = 1 if etapa == "2" else 2
                    pct_para_compra = float(sim["tabela"]["% PARA COMPRA"][etapa_idx].replace('%', ''))
                    preco_sugerido = sim["cotacao"] * (1 + pct_para_compra / 100)
                    preco_compra = st.number_input("💵 Preço da compra", step=0.01, format="%.2f", key=f"preco_compra_{idx}", value=float(preco_sugerido))
                    valor_sugerido = sim["pl_total"] * (float(sim["tabela"]["% PL COMPRA"][etapa_idx].replace('%','')) / 100)
                    qtd_sugerida = int(valor_sugerido / preco_sugerido)
                    qtd_compra = st.number_input("🔢 Quantidade comprada", step=1, min_value=1, key=f"qtd_compra_{idx}", value=qtd_sugerida)
                    data_compra = st.date_input("📅 Data da compra", value=datetime.date.today(), key=f"data_compra_{idx}")
                    if st.form_submit_button("Registrar Compra"):
                        nova_compra = {
                            "etapa": etapa,
                            "preco": preco_compra,
                            "qtd": qtd_compra,
                            "data": str(data_compra)
                        }
                        if "compras_reais" not in sim:
                            sim["compras_reais"] = []
                        sim["compras_reais"].append(nova_compra)
                        total_qtd = sum([c["qtd"] for c in sim["compras_reais"]])
                        total_valor = sum([c["preco"] * c["qtd"] for c in sim["compras_reais"]])
                        sim["quantidade_real"] = total_qtd
                        sim["preco_medio"] = total_valor / total_qtd if total_qtd else 0
                        ref.set(limpar_chaves_invalidas(st.session_state.simulacoes))
                        st.success("✅ Compra registrada com sucesso!")
                        st.rerun()






st.markdown("---")
st.subheader("📁 Simulações Finalizadas")
vendas_registradas = finalizadas_ref.get()
if vendas_registradas:
    for i, venda in enumerate(vendas_registradas[::-1]):
        with st.expander(f"{venda['tipo']} {venda['nome']} • {venda['data']} • {venda['quantidade']} ações vendidas a $ {venda['preco_venda']:.2f}"):
            st.markdown(f"""
            - 💰 **Lucro:** $ {venda['lucro']:.2f} ({venda['lucro_pct']:.2f}%)
            - 📆 **Data:** {venda['data']}
            - 📉 **Preço de venda:** $ {venda['preco_venda']:.2f}
            - 🔢 **Quantidade vendida:** {venda['quantidade']} ações
            """)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"✏️ Editar venda", key=f"editar_venda_{i}"):
                    with st.form(f"editar_venda_form_{i}"):
                        nova_data = st.date_input("📅 Nova data da venda", value=datetime.date.fromisoformat(venda["data"]), key=f"data_edit_{i}")
                        novo_preco = st.number_input("💲 Novo preço de venda", value=venda["preco_venda"], step=0.01, format="%.2f", key=f"preco_edit_{i}")
                        nova_qtd = st.number_input("🔢 Nova quantidade vendida", value=int(venda["quantidade"]), step=1, key=f"qtd_edit_{i}")
                        if st.form_submit_button("Salvar alterações"):
                            valor_total_venda = novo_preco * nova_qtd
                            valor_total_custo = (venda["lucro"] + venda["preco_venda"] * venda["quantidade"] - valor_total_venda)
                            lucro = valor_total_venda - valor_total_custo
                            lucro_pct = lucro / valor_total_custo * 100 if valor_total_custo != 0 else 0
                            venda_editada = {
                                **venda,
                                "data": str(nova_data),
                                "preco_venda": novo_preco,
                                "quantidade": nova_qtd,
                                "lucro": lucro,
                                "lucro_pct": lucro_pct
                            }
                            vendas_registradas[i] = venda_editada
                            finalizadas_ref.set(limpar_chaves_invalidas(vendas_registradas))
                            st.success("Venda editada com sucesso!")
                            st.rerun()
            with col2:
                if st.button(f"🗑 Excluir venda", key=f"excluir_venda_{i}"):
                    vendas_restantes = vendas_registradas.copy()
                    vendas_restantes.pop(len(vendas_registradas) - 1 - i)
                    finalizadas_ref.set(limpar_chaves_invalidas(vendas_restantes))
                    st.success("Venda removida com sucesso!")
                    st.rerun()
else:
    st.info("Nenhuma venda registrada ainda.")
