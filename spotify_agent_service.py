import os
import re
import yaml
import logging
import time
from typing import TypedDict, Annotated, List, Dict, Literal, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from spotify_auth import get_spotify_client
from metricsCallback import ExecutionMetricsCallback

load_dotenv()
logger = logging.getLogger("SpotifyStateGraph")


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

# 3. Construção dos Nós (Nodes) e Rotas (Edges)
class SpotifyCuratorGraphService:
    def __init__(self, yaml_path: str = "spotify_curator_agent.yaml"):
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

        # LLM com acesso apenas à busca web para pesquisar repertório
        self.research_llm = self.llm.bind_tools([web_search])
        # LLM forçado a devolver o objeto CurationOutput estruturado
        self.StructCurationLLM = self.llm.with_structured_output(CurationOutput)
        # LLM para interpretar a decisão humana (aprovou vs pediu ajuste)
        self.ApprovalLLM = self.llm.with_structured_output(ApprovalDecision)

        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # --- NÓ 1: Pesquisa e Curadoria Musical ---
    def curator_node(self, state: AgentState):
        system_msg = SystemMessage(content=self.config["system_prompt"])
        messages = [system_msg] + state["messages"]

        # Primeiro verifica se o LLM quer fazer uma busca na web (DuckDuckGo)
        response = self.research_llm.invoke(messages)
        if response.tool_calls:
            return {"messages": [response]}

        # Quando já tem as informações, gera a curadoria estruturada
        structured_res: CurationOutput = self.StructCurationLLM.invoke(messages)

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
    def evaluate_human_feedback_node(self, state: AgentState):
        last_user_msg = state["messages"][-1]
        decision: ApprovalDecision = self.ApprovalLLM.invoke([
            SystemMessage(content="Avalie se o usuário aprovou a lista de músicas proposta ou se pediu modificações."),
            last_user_msg
        ])
        return {"is_approved": decision.is_approved}

    # --- NÓ 3: Execução Determinística no Spotify ---
    def spotify_execution_node(self, state: AgentState, config: RunnableConfig):
        sp = get_spotify_client(config)
        playlist_id = state.get("playlist_id")
        mode = state.get("operation_mode", "create")

        # 1. Cria a playlist se necessário
        if mode == "create" or not playlist_id:
            payload = {
                "name": state["playlist_name"],
                "description": "Playlist criada pelo Musical Curator",
                "public": False,
            }
            created = sp._post("me/playlists", payload=payload)
            playlist_id = created["id"]

        # 2. Busca os URIs de cada faixa aprovada em Python puro
        valid_uris = []
        not_found = []
        for track in state["proposed_tracks"]:
            query = f"track:{track['title']} artist:{track['artist']}"
            res = sp.search(q=query, type="track", limit=1)
            items = res.get("tracks", {}).get("items", [])
            if items:
                valid_uris.append(items[0]["uri"])
            else:
                not_found.append(f"{track['title']} - {track['artist']}")

        # 3. Adiciona na playlist em lote único
        if valid_uris:
            sp.playlist_add_items(playlist_id=playlist_id, items=valid_uris[:100])

        playlist_url = f"https://open.spotify.com/playlist/{playlist_id}"
        summary_msg = (
            f"✅ Sucesso! {len(valid_uris)} músicas adicionadas à playlist.\n"
            f"🔗 Acesse aqui: {playlist_url}"
        )
        if not_found:
            summary_msg += f"\n⚠️ Faixas não localizadas no catálogo: {', '.join(not_found)}"

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

        # Rota condicional após resposta do usuário: Aprovou ou pediu mudanças?
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
            },
            "callbacks": [metrics_callback],
            "recursion_limit": self.config["model"].get("max_steps", 15),
        }

        # Verifica se o grafo está pausado no breakpoint (interrupt_before=["evaluate_feedback"])
        current_state = self.graph.get_state(run_config)

        if current_state.next and "evaluate_feedback" in current_state.next:
            # O grafo estava aguardando aprovação humana:
            # 1. Injeta a resposta do usuário no estado atual
            self.graph.update_state(
                run_config,
                {"messages": [HumanMessage(content=user_message)]},
            )
            # 2. Retoma a execução passando None como input
            result_state = self.graph.invoke(None, config=run_config)
        else:
            # Início normal de um novo fluxo de curadoria
            result_state = self.graph.invoke(
                {"messages": [HumanMessage(content=user_message)]},
                config=run_config,
            )

        elapsed_seconds = time.perf_counter() - start_time
        assistant_reply = result_state["messages"][-1].content

        llm_time = round(metrics_callback.llm_latency_seconds, 3)
        tools_and_api_time = round(max(0.0, elapsed_seconds - llm_time), 3)
        
        # Estrutura de telemetria da requisição
        telemetry = {
            "total_agent_seconds": round(elapsed_seconds, 3), # Tempo total do turno (sem humano)
            "llm_inference_seconds": llm_time,                # Tempo exclusivo do Gemini
            "external_apis_seconds": tools_and_api_time,      # Tempo de DuckDuckGo + Spotipy
            "llm_calls": metrics_callback.llm_calls,
            "input_tokens": metrics_callback.input_tokens,
            "output_tokens": metrics_callback.output_tokens,
            "total_tokens": metrics_callback.total_tokens,
        }

        # Registra no app.log para auditoria de custos e performance
        logger.info(
            f"[TELEMETRY] thread={thread_id} | "
            f"latency={telemetry['latency_seconds']}s | "
            f"llm_calls={telemetry['llm_calls']} | "
            f"tokens_in={telemetry['input_tokens']} | "
            f"tokens_out={telemetry['output_tokens']} | "
            f"tokens_total={telemetry['total_tokens']}"
        )

        return {
            "reply": assistant_reply,
            "proposed_tracks": result_state.get("proposed_tracks", []),
            "playlist_url": result_state.get("final_playlist_url"),
            "metrics": telemetry,
        }