import streamlit as st
import pandas as pd
import datetime
import firebase_admin # Para db e admin_auth
from firebase_admin import db, auth as admin_auth_sdk # Renomeado para evitar conflito com auth_pyre de app.py se importado
import smtplib # Para envio de email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# --- Configurações da Página e Verificação de Login/Admin ---
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Por favor, faça login primeiro.")
    st.link_button("Ir para Login", "/")
    st.stop()

# ADMIN_EMAILS e DEFAULT_SMTP_EMAIL, DEFAULT_SMTP_SENHA deveriam vir de app.py ou config
# Para este exemplo, vamos redefinir ADMIN_EMAILS e carregar SMTP dos secrets.
ADMIN_EMAILS_LIST = ["felipekuffel@gmail.com"] # Lista de emails admin
DEFAULT_SMTP_EMAIL_CFG = "felipekuffel@gmail.com" # Seu email SMTP
try:
    DEFAULT_SMTP_SENHA_CFG = st.secrets["smtp_senha"]
except KeyError:
    st.error("❌ Senha SMTP ('smtp_senha') não configurada nos secrets. Funcionalidade de email desabilitada.")
    DEFAULT_SMTP_SENHA_CFG = None


if st.session_state.get('email') not in ADMIN_EMAILS_LIST:
    st.error("🚫 Acesso restrito à área de Administração.")
    st.stop()

st.title("🛠️ Painel de Administração")
st.info("Bem-vindo(a) à área de gerenciamento do sistema.")

# --- Funções do Painel Admin ---
def listar_usuarios_firebase_admin():
    users_list = []
    try:
        for user_record in admin_auth_sdk.list_users().iterate_all():
            uid = user_record.uid
            trial_data_node = db.reference(f"trials/{uid}").get() # Usando Admin SDK DB
            
            dias_restantes_trial = '-'
            trial_expira_em_fmt = '-'
            status_trial_descr = "N/A"

            if trial_data_node and "trial_expiration" in trial_data_node:
                exp_date_str = trial_data_node["trial_expiration"]
                try:
                    exp_date_obj = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d")
                    trial_expira_em_fmt = exp_date_obj.strftime("%d/%m/%Y")
                    dias_restantes_trial = (exp_date_obj - datetime.datetime.utcnow()).days
                    if dias_restantes_trial < 0:
                        status_trial_descr = f"❌ Expirado (dias: {dias_restantes_trial})"
                    elif dias_restantes_trial <= 3:
                        status_trial_descr = f"⚠️ Expira em {dias_restantes_trial+1} dias"
                    else:
                        status_trial_descr = f"✅ Ativo ({dias_restantes_trial+1} dias)"
                except ValueError:
                    trial_expira_em_fmt = "Data Inválida"
                    status_trial_descr = "⚠️ Erro Data"
            else:
                status_trial_descr = "ℹ️ Sem Trial"
            
            # Formatando timestamps de criação e último login
            creation_timestamp_ms = user_record.user_metadata.creation_timestamp
            last_signin_timestamp_ms = user_record.user_metadata.last_sign_in_timestamp

            criado_em_fmt = pd.to_datetime(creation_timestamp_ms, unit='ms').strftime('%d/%m/%Y %H:%M') if creation_timestamp_ms else "N/A"
            ultimo_login_fmt = pd.to_datetime(last_signin_timestamp_ms, unit='ms').strftime('%d/%m/%Y %H:%M') if last_signin_timestamp_ms else "N/A"


            users_list.append({
                "Email": user_record.email,
                "UID": uid,
                "Verificado": "Sim" if user_record.email_verified else "Não",
                "Criado em": criado_em_fmt,
                "Último login": ultimo_login_fmt,
                "Trial Expira em": trial_expira_em_fmt,
                "Dias Restantes Trial": dias_restantes_trial if isinstance(dias_restantes_trial, int) else '-',
                "Status Trial": status_trial_descr
            })
        return pd.DataFrame(users_list)
    except Exception as e_list_users:
        st.error(f"Erro ao listar usuários: {e_list_users}")
        return pd.DataFrame()

# --- Seção: Gerenciamento de Usuários ---
st.subheader("👥 Usuários Cadastrados no Firebase")
df_usuarios = listar_usuarios_firebase_admin()

if not df_usuarios.empty:
    filtro_email_admin = st.text_input("🔍 Filtrar por email:", key="admin_filter_email")
    if filtro_email_admin:
        df_usuarios_filtrados = df_usuarios[df_usuarios['Email'].str.contains(filtro_email_admin, case=False, na=False)]
    else:
        df_usuarios_filtrados = df_usuarios
    
    # Remover UID da exibição padrão, mas manter no DataFrame original para operações
    cols_to_display = [col for col in df_usuarios_filtrados.columns if col != "UID"]
    st.dataframe(df_usuarios_filtrados[cols_to_display], use_container_width=True, hide_index=True)
    
    @st.cache_data
    def to_csv_admin(df_to_convert_admin):
        return df_to_convert_admin.to_csv(index=False).encode('utf-8')
    
    csv_admin_data = to_csv_admin(df_usuarios_filtrados) # Usa o dataframe filtrado para exportação
    st.download_button("⬇️ Baixar CSV dos Usuários", csv_admin_data, file_name="usuarios_firebase.csv", mime="text/csv")
else:
    st.info("Nenhum usuário encontrado ou erro ao carregar.")

st.markdown("---")
st.subheader("✏️ Gerenciar Trial de Usuário")
col_trial1, col_trial2 = st.columns(2)
with col_trial1:
    email_usuario_trial = st.text_input("Email do Usuário para Gerenciar Trial:", key="admin_email_trial")
with col_trial2:
    dias_para_renovar = st.number_input("Dias para Adicionar/Definir no Trial:", min_value=-365, max_value=365, value=7, step=1, key="admin_dias_trial")

if st.button("🔁 Atualizar Trial do Usuário", key="admin_update_trial_btn"):
    if email_usuario_trial and df_usuarios is not None and not df_usuarios.empty:
        usuario_selecionado_df = df_usuarios[df_usuarios['Email'] == email_usuario_trial]
        if not usuario_selecionado_df.empty:
            uid_usuario_trial = usuario_selecionado_df.iloc[0]['UID']
            try:
                nova_data_expiracao = (datetime.datetime.utcnow() + datetime.timedelta(days=int(dias_para_renovar))).strftime("%Y-%m-%d")
                db.reference(f"trials/{uid_usuario_trial}").set({"trial_expiration": nova_data_expiracao})
                st.success(f"Trial de {email_usuario_trial} atualizado para expirar em {pd.to_datetime(nova_data_expiracao).strftime('%d/%m/%Y')}.")
                st.rerun() # Para atualizar a tabela de usuários
            except Exception as e_trial_update:
                st.error(f"Erro ao atualizar trial: {e_trial_update}")
        else:
            st.error(f"Usuário com email '{email_usuario_trial}' não encontrado.")
    else:
        st.warning("Digite um email válido e certifique-se que a lista de usuários foi carregada.")


st.markdown("---")
st.subheader("🗑️ Excluir Usuário")
email_para_excluir_admin = st.text_input("Email do Usuário para EXCLUIR (ação irreversível!):", key="admin_email_delete")
if st.button("❌ EXCLUIR USUÁRIO", type="primary", key="admin_delete_user_btn"):
    if email_para_excluir_admin and df_usuarios is not None and not df_usuarios.empty:
        usuario_a_excluir_df = df_usuarios[df_usuarios['Email'] == email_para_excluir_admin]
        if not usuario_a_excluir_df.empty:
            uid_a_excluir = usuario_a_excluir_df.iloc[0]['UID']
            confirm_delete = st.checkbox(f"🚨 Confirmo que desejo excluir permanentemente o usuário {email_para_excluir_admin} (UID: {uid_a_excluir}).", key="admin_confirm_delete")
            if confirm_delete:
                try:
                    admin_auth_sdk.delete_user(uid_a_excluir)
                    # Opcional: Remover dados do usuário do Realtime Database também (ex: trial, carteira)
                    db.reference(f"trials/{uid_a_excluir}").delete()
                    db.reference(f"carteiras_multipage/{uid_a_excluir}").delete() # Ajustar path se necessário
                    st.success(f"Usuário {email_para_excluir_admin} e seus dados associados foram excluídos.")
                    st.rerun()
                except Exception as e_delete_user:
                    st.error(f"Erro ao excluir usuário: {e_delete_user}")
            else:
                st.warning("Exclusão não confirmada.")
        else:
            st.error(f"Usuário com email '{email_para_excluir_admin}' não encontrado para exclusão.")
    else:
        st.warning("Digite um email válido para exclusão.")

# --- Seção: Envio de Notificações por Email (SMTP) ---
st.markdown("---")
st.subheader("✉️ Enviar Notificação por Email (via SMTP)")

if DEFAULT_SMTP_SENHA_CFG: # Apenas mostra se a senha SMTP está configurada
    with st.form("form_admin_send_email"):
        email_destinatario_admin = st.text_input("Email do Destinatário:", placeholder="usuario@exemplo.com")
        assunto_email_admin = st.text_input("Assunto do Email:", "Notificação do Administrador")
        mensagem_email_admin = st.text_area("Mensagem para o Usuário:", height=150)
        
        submit_send_email = st.form_submit_button("🚀 Enviar Email")

    if submit_send_email:
        if email_destinatario_admin and assunto_email_admin and mensagem_email_admin:
            try:
                msg = MIMEMultipart()
                msg['From'] = DEFAULT_SMTP_EMAIL_CFG
                msg['To'] = email_destinatario_admin
                msg['Subject'] = assunto_email_admin
                msg.attach(MIMEText(mensagem_email_admin, 'plain'))

                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server: # Ajuste para seu provedor SMTP
                    server.login(DEFAULT_SMTP_EMAIL_CFG, DEFAULT_SMTP_SENHA_CFG)
                    server.send_message(msg)
                st.success(f"Email enviado com sucesso para {email_destinatario_admin}!")
            except Exception as e_smtp:
                st.error(f"Falha ao enviar email: {e_smtp}")
        else:
            st.warning("Por favor, preencha todos os campos do email.")
else:
    st.warning("Configuração de SMTP não encontrada nos secrets. Envio de email desabilitado.")