import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from langchain_core.runnables import RunnableConfig

def get_spotify_client(config: RunnableConfig) -> spotipy.Spotify:
    """
    Obtém o cliente Spotipy autenticado.
    - Cenário 1 (Web Multi-Tenant): Usa o 'spotify_access_token' injetado no RunnableConfig.
    - Cenário 2 (Execução Local / Fallback): Usa as variáveis SPOTIPY_* do arquivo .env.
    """
    configurable = config.get("configurable", {})
    access_token = configurable.get("spotify_access_token")

    # 1. Se o token do usuário veio injetado na requisição (Arquitetura Web/API)
    if access_token:
        return spotipy.Spotify(auth=access_token)

    # 2. Fallback: Se não veio token externo, autentica usando as variáveis do .env
    scope = (
        "playlist-read-private "
        "playlist-read-collaborative "
        "playlist-modify-public "
        "playlist-modify-private "
        "user-top-read "
        "user-read-recently-played"
    )
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"),
        scope=scope,
        open_browser=False,
    )
    return spotipy.Spotify(auth_manager=auth_manager)
