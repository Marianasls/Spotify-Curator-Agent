import os
import logging
from typing import List, Dict, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from schemas import CurationOutput

logger = logging.getLogger("SpotifyAgentHelpers")

MAX_WEB_SEARCHES_PER_TURN = 2

_ddg_search = DuckDuckGoSearchRun()


@tool("web_search")
def web_search(query: str) -> str:
    """Pesquisa na web por músicas, artistas, repertórios ou lançamentos musicais."""
    try:
        res = _ddg_search.invoke(query)
        return res if res else "Nenhum resultado encontrado na busca. Use seu conhecimento prévio."
    except Exception as e:
        logger.warning(f"Erro ou timeout ao executar web_search para '{query}': {e}")
        return f"Aviso: A busca na web falhou ({e}). Prossiga utilizando seu conhecimento interno."


def get_llm(configurable: dict) -> ChatGoogleGenerativeAI:
    """Instancia o LLM dinamicamente usando a chave do usuário (BYOK) ou fallback do .env."""
    api_key = configurable.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
    model_name = configurable.get("gemini_model") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise ValueError("Nenhuma GEMINI_API_KEY fornecida na sessão ou no ambiente.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2,
    )


def format_curation_reply(structured_res: CurationOutput, playlist_id: Optional[str] = None) -> str:
    """
    Garante que a resposta enviada ao usuário sempre contenha texto válido,
    a lista de músicas numerada e o pedido explícito de aprovação (Human-in-the-Loop).
    """
    base_msg = (structured_res.message_to_user or "").strip()

    tracks_lines = [
        f"{idx}. **{t.title}** — *{t.artist}*"
        for idx, t in enumerate(structured_res.tracks, 1)
    ]
    tracks_formatted = "\n".join(tracks_lines)

    if not base_msg:
        action_desc = (
            f"adicionar à playlist existente (`{playlist_id}`)"
            if structured_res.operation_mode == "append" and playlist_id
            else f"criar a nova playlist **{structured_res.suggested_playlist_name}**"
        )
        base_msg = f"Preparei a seguinte seleção para {action_desc}:\n\n{tracks_formatted}"
    elif structured_res.tracks and structured_res.tracks[0].title.lower() not in base_msg.lower():
        base_msg = f"{base_msg}\n\n{tracks_formatted}"

    approval_prompt = (
        "\n\n⚠️ **Aprovação necessária:** Você aprova a criação/atualização da playlist "
        "no Spotify com essas músicas? Digite **'Sim'** para confirmar ou informe quais alterações deseja fazer."
    )

    lower_msg = base_msg.lower()
    if not any(kw in lower_msg for kw in ["aprova", "confirm", "sim para", "deseja criar"]):
        base_msg += approval_prompt

    return base_msg


def get_user_musical_profile(sp) -> dict:
    """Consulta os artistas e gêneros mais ouvidos pelo usuário no Spotify."""
    try:
        top_artists = sp.current_user_top_artists(limit=10, time_range="medium_term")
        items = top_artists.get("items", [])

        favorite_artists = [artist["name"] for artist in items]

        genres = []
        for artist in items:
            genres.extend(artist.get("genres", []))

        return {
            "top_artists": favorite_artists,
            "genres": list(dict.fromkeys(genres))[:9],
        }
    except Exception as e:
        logger.warning(f"Não foi possível obter o perfil musical do usuário: {e}")
        return {"top_artists": [], "genres": []}


def build_curator_prompt_context(sp, state: Dict[str, Any], base_system_prompt: str) -> str:
    """
    Constrói o System Prompt enriquecido com o perfil musical do usuário,
    faixas já existentes na playlist (modo append) e seleção atual em validação.
    """
    user_profile = get_user_musical_profile(sp)
    taste_context = ""
    if user_profile["top_artists"] or user_profile["genres"]:
        artists_str = ", ".join(user_profile["top_artists"]) if user_profile["top_artists"] else "Não informado"
        genres_str = ", ".join(user_profile["genres"]) if user_profile["genres"] else "Diversos"
        taste_context = (
            f"\n\n[PERFIL MUSICAL DO USUÁRIO]:\n"
            f"- Artistas mais ouvidos: {artists_str}\n"
            f"- Gêneros favoritos: {genres_str}\n"
            f"Diretriz: Use essas preferências como referência estética para selecionar "
            f"ou sugerir faixas que combinem com a identidade do usuário, a menos que ele peça algo totalmente diferente."
        )

    detected_id = state.get("playlist_id")
    mode = state.get("operation_mode", "create")
    existing_tracks_context = ""

    if mode == "append" and detected_id:
        try:
            res = sp.playlist_items(
                detected_id,
                fields="items(track(name,artists(name)))",
                limit=100,
            )
            track_names = [
                f"{item['track']['name']} - {item['track']['artists'][0]['name']}"
                for item in res.get("items", [])
                if item.get("track")
            ]
            if track_names:
                existing_tracks_context = (
                    f"\n\n[ATENÇÃO - FAIXAS JÁ EXISTENTES NA PLAYLIST ({detected_id})]:\n"
                    f"A playlist já contém: {', '.join(track_names)}.\n"
                    f"NÃO inclua nenhuma dessas músicas na sua sugestão. Escolha apenas músicas inéditas."
                )
        except Exception as e:
            logger.warning(f"Não foi possível ler faixas da playlist {detected_id}: {e}")

    current_proposed = state.get("proposed_tracks", [])
    proposed_context = ""
    if current_proposed:
        current_list_str = ", ".join([f"{t['title']} - {t['artist']}" for t in current_proposed])
        proposed_context = (
            f"\n\n[SELEÇÃO ATUAL EM VALIDAÇÃO]:\n{current_list_str}\n"
            f"Diretriz: Se o usuário pediu alterações na última mensagem, aplique os ajustes "
            f"sobre esta lista, apresente a lista atualizada e peça expressamente a aprovação do usuário."
        )

    return base_system_prompt + taste_context + existing_tracks_context + proposed_context


def sync_tracks_to_spotify_playlist(
    sp,
    mode: str,
    playlist_id: Optional[str],
    playlist_name: str,
    proposed_tracks: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Executa a criação ou atualização determinística da playlist no Spotify,
    deduplicando faixas e retornando o resumo da operação.
    """
    existing_uris = set()

    if mode == "create" or not playlist_id:
        payload = {
            "name": playlist_name,
            "description": "Playlist criada pelo Musical Curator",
            "public": False,
        }
        created = sp._post("me/playlists", payload=payload)
        playlist_id = created["id"]
    else:
        try:
            pl_meta = sp.playlist(playlist_id, fields="name")
            if pl_meta and pl_meta.get("name"):
                playlist_name = pl_meta["name"]
        except Exception:
            pass

        res = sp.playlist_items(playlist_id, fields="items(track(uri)),next", limit=100)
        while res:
            for item in res.get("items", []):
                if item.get("track") and item["track"].get("uri"):
                    existing_uris.add(item["track"]["uri"])
            res = sp.next(res) if res.get("next") else None

    valid_uris = []
    not_found = []
    already_in_playlist = []

    for track in proposed_tracks:
        query = f"track:{track['title']} artist:{track['artist']}"
        res = sp.search(q=query, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])

        if items:
            uri = items[0]["uri"]
            if uri in existing_uris or uri in valid_uris:
                already_in_playlist.append(f"{track['title']} - {track['artist']}")
            else:
                valid_uris.append(uri)
        else:
            not_found.append(f"{track['title']} - {track['artist']}")

    if valid_uris:
        sp.playlist_add_items(playlist_id=playlist_id, items=valid_uris[:100])

    playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
    summary_msg = (
        f"Sucesso! {len(valid_uris)} músicas adicionadas à playlist **{playlist_name}**.\n"
        f"Acesse aqui: {playlist_url}"
    )
    if already_in_playlist:
        summary_msg += f"\nFaixas ignoradas (já estavam na playlist): {', '.join(already_in_playlist)}"
    if not_found:
        summary_msg += f"\nFaixas não localizadas no catálogo: {', '.join(not_found)}"

    return {
        "playlist_id": playlist_id,
        "playlist_name": playlist_name,
        "playlist_url": playlist_url,
        "added_count": len(valid_uris),
        "summary_msg": summary_msg,
    }


def count_searches_since_last_human(messages: List[AnyMessage]) -> int:
    """Conta quantas buscas web (ToolMessage) ocorreram desde a última mensagem do usuário."""
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "web_search":
            count += 1
    return count


def sanitize_playlist_id(raw_id: Optional[str], fallback_id: Optional[str] = None) -> Optional[str]:
    """Limpa URLs completas do Spotify extraindo apenas o ID alfanumérico."""
    clean_id = raw_id
    if clean_id and "spotify.com/playlist/" in clean_id:
        clean_id = clean_id.split("playlist/")[-1].split("?")[0]
    return clean_id or fallback_id
