import os
import yaml
import logging
import time
from typing import TypedDict, Annotated, List, Dict, Literal, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from langfuse.langchain import CallbackHandler
from spotify_auth import get_spotify_client
from metricsCallback import ExecutionMetricsCallback

load_dotenv()

logger = logging.getLogger("SpotifyStateGraph")

# Inicializa Langfuse CallbackHandler para tracing em produção
langfuse_handler = CallbackHandler()


# 1. Definição de Schemas Estruturados (Pydantic) e Estado do Grafo (State)
class TrackItem(BaseModel):
    title: str = Field(description="Título exato da música")
    artist: str = Field(description="Nome do artista principal")


class CurationOutput(BaseModel):
    """Schema forçado no LLM para garantir saída estruturada e determinística."""
    operation_mode: Literal["create", "append"] = Field(
        description="'append' se o usuário informou link/ID de playlist, senão 'create'"
    )
    extracted_playlist_id: Optional[str] = Field(
        default=None, description="ID limpo da playlist caso informada (sem ?si=...)"
    )
    suggested_playlist_name: str = Field(
        description="Nome criativo sugerido para a playlist"
    )
    tracks: List[TrackItem] = Field(
        description="Lista de músicas selecionadas na curadoria"
    )
    message_to_user: str = Field(
        description="Mensagem apresentando a lista numerada e perguntando se o usuário aprova"
    )


class ApprovalDecision(BaseModel):
    """Schema para classificar a resposta do usuário na etapa de HITL."""
    is_approved: bool = Field(
        description="True se o usuário aprovou a lista sem pendências; False se pediu alterações."
    )
    feedback: Optional[str] = Field(
        default=None, description="Instruções de modificação caso is_approved seja False."
    )


class AgentState(TypedDict):
    """Estado tipado que trafega entre os nós do grafo."""
    messages: Annotated[List[AnyMessage], add_messages]
    operation_mode: Literal["create", "append"]
    playlist_id: Optional[str]
    playlist_name: str
    proposed_tracks: List[dict]
    is_approved: bool
    final_playlist_url: Optional[str]


# 2. Helpers e Ferramentas
web_search = DuckDuckGoSearchRun(name="web_search")


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

# 3. Construção dos Nós (Nodes) e Rotas (Edges)
class SpotifyCuratorGraphService:
    def __init__(self, yaml_path: Optional[str] = None):        
        # Se nenhum caminho for passado, busca o YAML na mesma pasta (/src) deste script
        if yaml_path is None:
            yaml_path = str(Path(__file__).resolve().parent / "spotify_curator_agent.yaml")

        with open(yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.pool: Optional[ConnectionPool] = None
        self.checkpointer = self._init_checkpointer()

        # Grafo compilado
        self.graph = self._build_graph()

    def _init_checkpointer(self):
        """
        Configura o PostgresSaver com ConnectionPool se DATABASE_URL estiver definida.
        Caso contrário, utiliza MemorySaver para desenvolvimento local rápido.
        """
        db_uri = os.getenv("DATABASE_URL")
        if db_uri:
            logger.info("Conectando ao PostgreSQL para persistência de estado (PostgresSaver)...")
            # autocommit=True e prepare_threshold=0 evitam erros em poolers como Supabase/Neon
            self.pool = ConnectionPool(
                conninfo=db_uri,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            checkpointer = PostgresSaver(self.pool)
            # Cria as tabelas de checkpoint automaticamente no banco caso não existam
            checkpointer.setup()
            return checkpointer

        logger.warning("DATABASE_URL não encontrada. Usando MemorySaver (em memória).")
        return MemorySaver()

    # --- NÓ 1: Pesquisa e Curadoria Musical ---
    def curator_node(self, state: AgentState, config: RunnableConfig):
        configurable = config.get("configurable", {})
        llm = get_llm(configurable)

        research_llm = llm.bind_tools([web_search])
        struct_curation_llm = llm.with_structured_output(CurationOutput)

        sp = get_spotify_client(config)

        # Extrai o perfil de gosto do usuário
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
        # Verifica se é uma playlist existente para avisar o LLM das músicas atuais
        detected_id = state.get("playlist_id")
        mode = state.get("operation_mode", "create")
        existing_tracks_context = ""

        if mode == "append" and detected_id:
            try:
                # Busca apenas nome e artista das faixas atuais
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

        # 4. Combina Prompt Base + Perfil Musical + Faixas Existentes
        full_system_prompt = self.config["system_prompt"] + taste_context + existing_tracks_context
        messages = [SystemMessage(content=full_system_prompt)] + state["messages"]

        # Primeiro verifica se o LLM quer fazer uma busca na web (DuckDuckGo)
        response = research_llm.invoke(messages)
        if response.tool_calls:
            return {"messages": [response]}

        # Quando já tem as informações, gera a curadoria estruturada
        structured_res: CurationOutput = struct_curation_llm.invoke(messages)

        # Sanitização determinística de URL caso o LLM tenha deixado passar
        clean_id = structured_res.extracted_playlist_id
        if clean_id and "spotify.com/playlist/" in clean_id:
            clean_id = clean_id.split("playlist/")[-1].split("?")[0]

        return {
            "messages": [AIMessage(content=structured_res.message_to_user)],
            "operation_mode": structured_res.operation_mode,
            "playlist_id": clean_id or state.get("playlist_id"),
            "playlist_name": structured_res.suggested_playlist_name,
            "proposed_tracks": [t.model_dump() for t in structured_res.tracks],
            "is_approved": False,
        }

    # --- NÓ 2: Avaliação do Feedback Humano (Pós-Interrupção) ---
    def evaluate_human_feedback_node(self, state: AgentState, config: RunnableConfig):
        configurable = config.get("configurable", {})
        llm = get_llm(configurable)
        approval_llm = llm.with_structured_output(ApprovalDecision)

        last_user_msg = state["messages"][-1]
        decision: ApprovalDecision = approval_llm.invoke([
            SystemMessage(content="Avalie se o usuário aprovou a lista de músicas proposta ou se pediu modificações."),
            last_user_msg
        ])
        return {"is_approved": decision.is_approved}

    # --- NÓ 3: Execução Determinística no Spotify ---
    def spotify_execution_node(self, state: AgentState, config: RunnableConfig):
        sp = get_spotify_client(config)
        playlist_id = state.get("playlist_id")
        mode = state.get("operation_mode", "create")

        existing_uris = set()

        # 1. Cria a playlist se necessário
        if mode == "create" or not playlist_id:
            payload = {
                "name": state["playlist_name"],
                "description": "Playlist criada pelo Musical Curator",
                "public": False,
            }
            created = sp._post("me/playlists", payload=payload)
            playlist_id = created["id"]
        else:
            # Modo append: coleta todos os URIs que já estão na playlist para trava exata
            res = sp.playlist_items(playlist_id, fields="items(track(uri)),next", limit=100)
            while res:
                for item in res.get("items", []):
                    if item.get("track") and item["track"].get("uri"):
                        existing_uris.add(item["track"]["uri"])
                res = sp.next(res) if res.get("next") else None
        
        # 2. Busca os URIs de cada faixa aprovada em Python puro
        valid_uris = []
        not_found = []
        already_in_playlist = []

        for track in state["proposed_tracks"]:
            query = f"track:{track['title']} artist:{track['artist']}"
            res = sp.search(q=query, type="track", limit=1)
            items = res.get("tracks", {}).get("items", [])

            if items:
                uri = items[0]["uri"]
                # Verifica se já está na playlist ou se já foi adicionada neste mesmo lote
                if uri in existing_uris or uri in valid_uris:
                    already_in_playlist.append(f"{track['title']} - {track['artist']}")
                else:
                    valid_uris.append(uri)
            else:
                not_found.append(f"{track['title']} - {track['artist']}")

        # 3. Adiciona na playlist em lote único
        if valid_uris:
            sp.playlist_add_items(playlist_id=playlist_id, items=valid_uris[:100])

        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
        summary_msg = (
            f"Sucesso! {len(valid_uris)} músicas adicionadas à playlist.\n"
            f"Acesse aqui: {playlist_url}"
        )
        if already_in_playlist:
            summary_msg += f"\nFaixas ignoradas (já estavam na playlist): {', '.join(already_in_playlist)}"
        if not_found:
            summary_msg += f"\nFaixas não localizadas no catálogo: {', '.join(not_found)}"

        return {
            "playlist_id": playlist_id,
            "final_playlist_url": playlist_url,
            "messages": [AIMessage(content=summary_msg)],
        }

    # --- LÓGICA DE ROTAS (Conditional Edges) ---
    def route_after_curator(self, state: AgentState) -> Literal["tools", "wait_human_approval"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "wait_human_approval"

    def route_after_human_feedback(self, state: AgentState) -> Literal["spotify_execution", "curator"]:
        if state.get("is_approved"):
            return "spotify_execution"
        return "curator"

    # --- MONTAGEM DO STATEGRAPH ---
    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Registro dos Nós
        workflow.add_node("curator", self.curator_node)
        workflow.add_node("tools", ToolNode([web_search]))
        workflow.add_node("evaluate_feedback", self.evaluate_human_feedback_node)
        workflow.add_node("spotify_execution", self.spotify_execution_node)

        # Fluxo Inicial
        workflow.add_edge(START, "curator")
        
        # Rota condicional: O curador precisa buscar na web ou já terminou a lista?
        workflow.add_conditional_edges(
            "curator",
            self.route_after_curator,
            {
                "tools": "tools",
                "wait_human_approval": "evaluate_feedback",
            },
        )
        workflow.add_edge("tools", "curator")

        # Rota condicional após resposta do usuário: Aprovou ou pediu mudança?
        workflow.add_conditional_edges(
            "evaluate_feedback",
            self.route_after_human_feedback,
            {
                "spotify_execution": "spotify_execution",
                "curator": "curator",
            },
        )
        workflow.add_edge("spotify_execution", END)

        # Compilação com INTERRUPÇÃO FÍSICA antes de avaliar o feedback humano
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["evaluate_feedback"],
        )

    def execute_query(
        self,
        user_id: str,
        session_id: str,
        spotify_access_token: str,
        user_message: str,
        custom_gemini_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executa um turno da conversa no StateGraph, medindo tempo total de
        processamento (LLM + Tools + API Spotify) e consumo exato de tokens.
        """
        metrics_callback = ExecutionMetricsCallback()
        start_time = time.perf_counter()

        thread_id = f"{user_id}:{session_id}"
        run_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "spotify_access_token": spotify_access_token,
                "gemini_api_key": custom_gemini_key,
            },
            "callbacks": [langfuse_handler, metrics_callback],
            "recursion_limit": self.config["model"].get("max_steps", 15),
        }

        # Verifica se o grafo está pausado no breakpoint (interrupt_before=["evaluate_feedback"])
        current_state = self.graph.get_state(run_config)

        # Sanitiza a entrada para verificar se está vazia ou só tem espaços
        clean_message = user_message.strip() if user_message else ""

        if current_state.next and "evaluate_feedback" in current_state.next:
            # Se user_message for vazio, mantém aguardando aprovação e solicita aprovação do usuário novamente.
            if not clean_message:
                state_values = current_state.values
                return {
                    "reply": "Por favor, informe se aprova a playlist ou quais ajustes deseja fazer.",
                    "proposed_tracks": state_values.get("proposed_tracks", []),
                    "playlist_url": state_values.get("final_playlist_url"),
                }

            # O grafo estava aguardando aprovação humana:
            # 1. Injeta a resposta do usuário no estado atual
            self.graph.update_state(
                run_config,
                {"messages": [HumanMessage(content=clean_message)]},
            )
            # 2. Retoma a execução passando None como input
            result_state = self.graph.invoke(None, config=run_config)
        else:
            # Se for o início do fluxo e o usuário der Enter vazio
            if not clean_message:
                return {
                    "reply": "Por favor, digite o que você gostaria de ouvir para começarmos.",
                    "proposed_tracks": [],
                    "playlist_url": None,
                }

            # Início normal de um novo fluxo de curadoria
            result_state = self.graph.invoke(
                {"messages": [HumanMessage(content=clean_message)]},
                config=run_config,
            )

        elapsed_seconds = time.perf_counter() - start_time
        assistant_reply = result_state["messages"][-1].content

        llm_time = round(metrics_callback.llm_latency_seconds, 3)
        tools_and_api_time = round(max(0.0, elapsed_seconds - llm_time), 3)

        return {
            "reply": assistant_reply,
            "proposed_tracks": result_state.get("proposed_tracks", []),
            "playlist_url": result_state.get("final_playlist_url"),
            "metrics": {
                "total_seconds": round(elapsed_seconds, 3),
                "llm_seconds": llm_time,
                "external_seconds": tools_and_api_time,
                "total_tokens": metrics_callback.total_tokens,
            }
        }