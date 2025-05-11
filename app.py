import streamlit as st
import datetime
import pyrebase
import firebase_admin
from firebase_admin import credentials, auth as admin_auth, db
from cryptography.hazmat.primitives import serialization

# --- Configuração da Página ---
st.set_page_config(
    page_title="Painel de Análise",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes Globais (se necessário, ou importe de um config.py) ---
ADMIN_EMAILS = ["felipekuffel@gmail.com"] # Defina seus emails de administrador
DEFAULT_SMTP_EMAIL = "felipekuffel@gmail.com" # Necessário para funcionalidade de admin
# DEFAULT_SMTP_SENHA = st.secrets.get("smtp_senha", "SUA_SENHA_SMTP_PADRAO_SE_NAO_ESTIVER_NO_SECRETS") # Carregado depois dos secrets

# --- Verificação da Chave Privada Firebase Admin ---
try:
    firebase_admin_creds = st.secrets["firebase_admin"]
    key = firebase_admin_creds["private_key"]
    serialization.load_pem_private_key(key.encode(), password=None)
except KeyError:
    st.error("❌ Chave 'firebase_admin' não encontrada nos secrets do Streamlit.")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro na chave privada do Firebase Admin: {e}")
    st.stop()

# --- Inicialização do Firebase Admin SDK ---
if not firebase_admin._apps:
    try:
        cred_dict = dict(firebase_admin_creds) # Usa as credenciais já carregadas
        if "databaseURL" not in cred_dict and "databaseURL" in st.secrets: # Garante databaseURL
             cred_dict["databaseURL"] = st.secrets["databaseURL"]
        elif "databaseURL" not in cred_dict:
             st.error("❌ 'databaseURL' não encontrada nos secrets para Firebase Admin SDK.")
             st.stop()

        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": cred_dict["databaseURL"]})
    except Exception as e:
        st.error(f"❌ Erro ao inicializar Firebase Admin SDK: {e}")
        st.stop()

# --- Configuração do Pyrebase (para Autenticação) ---
try:
    firebase_config = {
        "apiKey": st.secrets["firebase_apiKey"],
        "authDomain": st.secrets["firebase_authDomain"],
        "projectId": st.secrets["firebase_projectId"],
        "storageBucket": st.secrets["firebase_storageBucket"],
        "messagingSenderId": st.secrets["firebase_messagingSenderId"],
        "appId": st.secrets["firebase_appId"],
        "measurementId": st.secrets.get("firebase_measurementId", None),
        "databaseURL": st.secrets["databaseURL"]
    }
    firebase_pyre_app = pyrebase.initialize_app(firebase_config)
    auth_pyre = firebase_pyre_app.auth()
except KeyError as e:
    st.error(f"❌ Chave de configuração do Firebase (Pyrebase) não encontrada nos secrets: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao inicializar Pyrebase (para autenticação): {e}")
    st.stop()

# --- Funções de Autenticação ---
def perform_login(email, password):
    try:
        user_creds = auth_pyre.sign_in_with_email_and_password(email, password)
        user_id = user_creds["localId"]
        
        # Armazena informações do usuário na sessão
        st.session_state.user = user_creds
        st.session_state.email = email # Email fornecido no login
        st.session_state.refresh_token = user_creds["refreshToken"]
        st.session_state.logged_in = True

        # Lógica de Trial para usuários não-admin
        if email not in ADMIN_EMAILS:
            trial_ref = db.reference(f"trials/{user_id}")
            trial_info = trial_ref.get()
            if trial_info is None: # Novo usuário ou sem trial registrado
                expiration_date = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                trial_ref.set({"trial_expiration": expiration_date})
                st.session_state.trial_expiration = expiration_date
                st.info("✅ Período de trial de 7 dias criado automaticamente.")
            else:
                trial_expiration_str = trial_info.get("trial_expiration", "")
                st.session_state.trial_expiration = trial_expiration_str
                try:
                    trial_expiration_date = datetime.datetime.strptime(trial_expiration_str, "%Y-%m-%d")
                    if trial_expiration_date < datetime.datetime.utcnow():
                        st.error("⛔️ Seu período de trial expirou. Contate o suporte.")
                        # Poderia deslogar aqui ou restringir acesso nas páginas
                        del st.session_state.logged_in # Força logout se trial expirado no login
                        return False # Indica falha no login devido a trial expirado
                except ValueError:
                     st.warning("⚠️ Data de expiração do trial em formato inválido. Contate o suporte.")
                     # Considerar criar um novo trial ou bloquear
        else:
            st.session_state.is_admin = True # Marca como admin

        st.rerun() # Importante para recarregar o app no estado logado
        return True # Indica sucesso no login
    except Exception as e:
        # Tenta extrair mensagens de erro mais amigáveis do Firebase
        error_message = str(e)
        if "EMAIL_NOT_FOUND" in error_message or "INVALID_PASSWORD" in error_message:
            st.error("Email ou senha incorretos.")
        elif "USER_DISABLED" in error_message:
            st.error("Esta conta de usuário foi desabilitada.")
        else:
            st.error(f"Erro ao fazer login: {e}")
        return False


def perform_registration(email, password):
    try:
        user_creds = auth_pyre.create_user_with_email_and_password(email, password)
        user_id = user_creds["localId"]
        
        # Criação automática de Trial para novos usuários
        expiration_date = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        db.reference(f"trials/{user_id}").set({"trial_expiration": expiration_date})
        
        st.success("Usuário criado com sucesso! Faça login para continuar.")
        st.balloons()
    except Exception as e:
        error_message = str(e)
        if "EMAIL_EXISTS" in error_message:
            st.error("⚠️ Este email já está registrado. Tente fazer login.")
        elif "WEAK_PASSWORD" in error_message:
            st.error("❌ A senha é muito fraca. Deve ter no mínimo 6 caracteres.")
        else:
            st.error(f"Erro ao registrar: {e}")

def page_login_registration():
    st.markdown("<h1 style='text-align: center;'>Painel de Análise Técnica</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([1,1.5,1])
    with col2:
        active_tab = st.radio("", ["Login", "Registrar Nova Conta"], horizontal=True, label_visibility="collapsed")
        email = st.text_input("📧 Email", key="auth_email")
        password = st.text_input("🔑 Senha", type="password", key="auth_password")

        if active_tab == "Login":
            if st.button("Entrar", use_container_width=True, type="primary"):
                if email and password:
                    perform_login(email, password)
                else:
                    st.warning("Por favor, preencha email e senha.")
        
        elif active_tab == "Registrar Nova Conta":
            confirm_password = st.text_input("🔑 Confirmar Senha", type="password", key="auth_confirm_password")
            if st.button("Criar Conta", use_container_width=True):
                if email and password and confirm_password:
                    if password == confirm_password:
                        perform_registration(email, password)
                    else:
                        st.error("As senhas não coincidem.")
                else:
                    st.warning("Por favor, preencha todos os campos.")
    st.markdown("---")

def restore_session():
    if "refresh_token" in st.session_state and "logged_in" not in st.session_state:
        try:
            user_refreshed = auth_pyre.refresh(st.session_state.refresh_token)
            st.session_state.user = user_refreshed
            st.session_state.logged_in = True
            
            # Tenta obter o email do Firebase Admin SDK (mais confiável)
            try:
                firebase_user_admin = admin_auth.get_user(user_refreshed["userId"])
                st.session_state.email = firebase_user_admin.email
            except Exception: # Fallback para o email do Pyrebase (pode não estar sempre presente)
                account_info = auth_pyre.get_account_info(user_refreshed["idToken"])
                if account_info and "users" in account_info and len(account_info["users"]) > 0:
                    st.session_state.email = account_info["users"][0].get("email", user_refreshed.get("email"))
                else: # Se não conseguir o email, pode ser um problema
                    st.session_state.email = user_refreshed.get("email", "Email não disponível")

            # Atualizar trial status na sessão
            if st.session_state.email not in ADMIN_EMAILS:
                user_id = user_refreshed["localId"]
                trial_ref = db.reference(f"trials/{user_id}")
                trial_info = trial_ref.get()
                if trial_info and "trial_expiration" in trial_info:
                    st.session_state.trial_expiration = trial_info["trial_expiration"]
                else: # Caso não tenha trial info, pode criar um novo ou marcar como expirado
                    expiration_date = (datetime.datetime.utcnow() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                    trial_ref.set({"trial_expiration": expiration_date})
                    st.session_state.trial_expiration = expiration_date

            return True # Sessão restaurada
        except Exception as e:
            # st.warning(f"Sessão expirada ou inválida: {e}. Por favor, faça login novamente.")
            for key_to_del in ['user', 'email', 'logged_in', 'refresh_token', 'trial_expiration', 'is_admin']:
                if key_to_del in st.session_state:
                    del st.session_state[key_to_del]
            return False # Falha ao restaurar
    return "logged_in" in st.session_state # Retorna true se já estava logado, false se não tinha refresh token

# --- Controle de Fluxo Principal ---
if not restore_session(): # Tenta restaurar sessão; se falhar (e não estava logado), mostra login
    page_login_registration()
    st.stop()

# --- Se chegou aqui, o usuário está logado ---
st.sidebar.success(f"Login como: {st.session_state.get('email', 'N/A')}")
st.sidebar.markdown("---")

# Mensagem de boas-vindas na página principal (app.py)
st.title(f"Bem-vindo(a) ao Painel de Análise, {st.session_state.get('email', '').split('@')[0]}!")
st.markdown("Utilize a navegação na barra lateral para acessar as funcionalidades do sistema.")
st.markdown("---")
st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Análise de Dados Financeiros")


# Botão de Logout na Sidebar (aparece em todas as páginas)
if st.sidebar.button("🚪 Sair", use_container_width=True):
    user_email_on_logout = st.session_state.get('email', 'Usuário')
    for key in list(st.session_state.keys()): # Limpa toda a session state
        del st.session_state[key]
    st.info(f"{user_email_on_logout} saiu com sucesso.")
    st.rerun() # Recarrega para voltar à tela de login