import os
import sys
from pathlib import Path
import uuid
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Localiza a pasta /src a partir de /web/app.py e adiciona ao sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spotify_agent_service import SpotifyCuratorGraphService

load_dotenv()

# 1. Configuração da Página
st.set_page_config(
    page_title="Spotify AI Music Curator",
    page_icon="🎧",
    layout="wide",
)

MAX_FREE_USES = int(os.getenv("MAX_FREE_USES", 3))

# 2. Inicialização de Recursos e Estado da Sessão (Session State)
@st.cache_resource
def get_agent_service() -> SpotifyCuratorGraphService:
    """
    Instancia o serviço do LangGraph uma única vez no servidor,
    compartilhando o ConnectionPool do PostgreSQL entre sessões.
    """
    return SpotifyCuratorGraphService()


def get_spotify_oauth() -> SpotifyOAuth:
    """Configura o gerenciador OAuth do Spotify sem cache em disco (seguro para multi-usuários)."""
    scope = (
        "playlist-read-private "
        "playlist-read-collaborative "
        "playlist-modify-public "
        "playlist-modify-private "
        "user-top-read "
        "user-read-recently-played "
    )
    return SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8501/callback"),
        scope=scope,
        open_browser=False,
        cache_handler=spotipy.MemoryCacheHandler(),
        cache_path=None,
    )


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "spotify_token" not in st.session_state:
        st.session_state.spotify_token = None
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "proposed_tracks" not in st.session_state:
        st.session_state.proposed_tracks = []
    if "playlist_url" not in st.session_state:
        st.session_state.playlist_url = None
    if "free_uses_count" not in st.session_state:
        st.session_state.free_uses_count = 0
    if "awaiting_approval" not in st.session_state:
        st.session_state.awaiting_approval = False
    if "editing_changes" not in st.session_state:
        st.session_state.editing_changes = False


init_session_state()
agent_service = get_agent_service()


# 3. Captura do Callback OAuth2 do Spotify via Query Params
if not st.session_state.spotify_token and "code" in st.query_params:
    auth_code = st.query_params["code"]
    try:
        oauth = get_spotify_oauth()
        access_token = oauth.get_access_token(auth_code, as_dict=False, check_cache=False)

        if not access_token:
            raise ValueError("Não foi possível obter o access_token a partir do código informado.")
            
        # Busca os dados do usuário logado no Spotify
        sp = spotipy.Spotify(auth=access_token)
        profile = sp.current_user()

        st.session_state.spotify_token = access_token
        st.session_state.user_profile = {
            "id": profile["id"],
            "name": profile.get("display_name", profile["id"]),
        }
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Falha ao autenticar com o Spotify: {e}")
        st.query_params.clear()


# 4. Barra Lateral (Autenticação, BYOK e Estado da Curadoria)
with st.sidebar:
    st.title("🎧 Configurações")

    # --- Seção 1: Conta Spotify ---
    st.subheader("1. Conta Spotify")
    if st.session_state.spotify_token and st.session_state.user_profile:
        st.success(f"Conectado como: **{st.session_state.user_profile['name']}**")
        if st.button("Sair / Trocar Conta", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    else:
        st.warning("Conecte sua conta para criar e editar playlists.")
        oauth = get_spotify_oauth()
        auth_url = oauth.get_authorize_url()
        st.markdown(
            f'<a href="{auth_url}" target="_self" style="'
            'display: inline-block; width: 100%; text-align: center; '
            'padding: 0.5rem 1rem; background-color: #1DB954; color: #FFFFFF; '
            'text-decoration: none; border-radius: 0.5rem; font-weight: 600;">'
            '🟢 Conectar com Spotify</a>',
            unsafe_allow_html=True,
        )

    st.divider()

    # --- Seção 2: Configuração da LLM (Híbrido: Free Tier + BYOK) ---
    st.subheader("2. Modelo de IA (Gemini)")
    remaining_free = max(0, MAX_FREE_USES - st.session_state.free_uses_count)
    st.caption(f"Interações gratuitas de demonstração restantes: **{remaining_free}/{MAX_FREE_USES}**")

    custom_gemini_key = st.text_input(
        "Sua Gemini API Key (Opcional):",
        type="password",
        placeholder="AIzaSy...",
        help="Insira sua chave do Google AI Studio para uso ilimitado.",
    )

    st.divider()

    # --- Seção 3: Prévia da Curadoria e Ações de Sessão ---
    if st.session_state.proposed_tracks:
        st.subheader("🎵 Faixas em Validação")
        for idx, track in enumerate(st.session_state.proposed_tracks, 1):
            st.markdown(f"**{idx}.** {track.get('title')} — *{track.get('artist')}*")

    if st.session_state.playlist_url:
        st.subheader("✅ Playlist Salva")
        st.link_button("Abrir Playlist no Spotify", st.session_state.playlist_url, use_container_width=True)

    if st.button("🔄 Nova Conversa / Playlist", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.proposed_tracks = []
        st.session_state.playlist_url = None
        st.session_state.awaiting_approval = False
        st.session_state.editing_changes = False
        st.rerun()


# 5. Área Principal de Chat
st.title("🎵 Spotify AI Music Curator")
st.markdown(
    "Peça uma nova playlist personalizada (ex: *'Crie uma playlist de Indie Rock anos 2000 com 8 músicas'*) "
    "ou envie o link de uma playlist existente para adicionar novas faixas."
)

# Bloqueia o chat enquanto o usuário não fizer login no Spotify
if not st.session_state.spotify_token:
    st.info("👈 Clique em **Conectar com Spotify** na barra lateral para iniciar sua curadoria.")
    st.stop()

# Renderiza o histórico de mensagens da sessão atual
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metrics"):
            m = msg["metrics"]
            st.caption(
                f"⚡ Total: {m['total_seconds']}s (LLM: {m['llm_seconds']}s | "
                f"APIs: {m['external_seconds']}s) | 🪙 Tokens: {m['total_tokens']}"
            )

# 6. Controles de Aprovação (Human-in-the-Loop) acima do campo de texto
pending_action_prompt = None

if st.session_state.awaiting_approval:
    st.markdown("---")
    if not st.session_state.editing_changes:
        st.info("🔔 **Aprovação pendente:** Escolha uma das opções abaixo para prosseguir.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirmar Playlist", type="primary", use_container_width=True):
                pending_action_prompt = "Sim, aprovo a playlist."
        with col2:
            if st.button("✏️ Solicitar Alterações", use_container_width=True):
                st.session_state.editing_changes = True
                st.rerun()
    else:
        st.warning("✏️ **Modo de edição ativo:** Digite abaixo quais alterações deseja fazer na seleção.")

# Bloqueia o campo de texto enquanto aguarda o clique em um dos botões de decisão
chat_disabled = st.session_state.awaiting_approval and not st.session_state.editing_changes
chat_placeholder = (
    "Selecione 'Confirmar Playlist' ou 'Solicitar Alterações' nos botões acima..."
    if chat_disabled
    else (
        "Descreva quais alterações deseja fazer na playlist..."
        if st.session_state.editing_changes
        else "Descreva a playlist que deseja criar ou envie o link de uma playlist existente..."
    )
)

# Captura a entrada do usuário no chat (ou a confirmação disparada pelo botão)
user_input = st.chat_input(chat_placeholder, disabled=chat_disabled)
active_prompt = pending_action_prompt or user_input

if active_prompt:
    clean_prompt = active_prompt.strip()
    if not clean_prompt:
        st.warning("Por favor, digite uma mensagem válida.")
        st.stop()

    # Verifica limite da cota de demonstração caso o usuário não tenha informado chave própria
    using_custom_key = bool(custom_gemini_key and custom_gemini_key.strip())
    if not using_custom_key and st.session_state.free_uses_count >= MAX_FREE_USES:
        st.error(
            "Você atingiu o limite de interações gratuitas de demonstração. "
            "Por favor, insira sua **Gemini API Key** na barra lateral para continuar."
        )
        st.stop()

    # 1. Exibe e salva a mensagem do usuário na interface
    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    # 2. Invoca o SpotifyCuratorGraphService
    with st.chat_message("assistant"):
        with st.spinner("O agente está processando sua solicitação..."):
            try:
                result = agent_service.execute_query(
                    user_id=st.session_state.user_profile["id"],
                    session_id=st.session_state.session_id,
                    spotify_access_token=st.session_state.spotify_token,
                    user_message=clean_prompt,
                    custom_gemini_key=custom_gemini_key.strip() if using_custom_key else None,
                )

                reply_text = result["reply"]
                metrics = result.get("metrics")

                # Atualiza o estado lateral com as músicas propostas e o link final
                st.session_state.proposed_tracks = result.get("proposed_tracks", [])
                st.session_state.playlist_url = result.get("playlist_url")

                # Verifica se a resposta exige aprovação do usuário (lista proposta e playlist ainda não salva)
                needs_approval = bool(
                    st.session_state.proposed_tracks and not st.session_state.playlist_url
                )
                st.session_state.awaiting_approval = needs_approval
                st.session_state.editing_changes = False

                # Contabiliza o uso gratuito caso não esteja usando chave própria
                if not using_custom_key:
                    st.session_state.free_uses_count += 1

                # Exibe a resposta do agente e a telemetria
                st.markdown(reply_text)
                if metrics:
                    st.caption(
                        f"⚡ Total: {metrics['total_seconds']}s (LLM: {metrics['llm_seconds']}s | "
                        f"APIs: {metrics['external_seconds']}s) | 🪙 Tokens: {metrics['total_tokens']}"
                    )

                # Salva a resposta no histórico do Streamlit
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply_text,
                    "metrics": metrics,
                })

                # Atualiza a interface para renderizar os botões de aprovação ou o link final
                st.rerun()

            except Exception as e:
                st.error(f"Erro ao processar sua solicitação: {e}")