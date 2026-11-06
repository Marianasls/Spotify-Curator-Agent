import yaml
import logging
import time
from typing import List, Dict, Literal, Optional, Any
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError
from langfuse.langchain import CallbackHandler

from schemas import CurationOutput, ApprovalDecision, AgentState
from database import ConversationRepository
from agent_helpers import (
    MAX_WEB_SEARCHES_PER_TURN,
    web_search,
    get_llm,
    format_curation_reply,
    build_curator_prompt_context,
    sync_tracks_to_spotify_playlist,
    count_searches_since_last_human,
    sanitize_playlist_id,
)
from spotify_auth import get_spotify_client
from metricsCallback import ExecutionMetricsCallback

load_dotenv()

logger = logging.getLogger("SpotifyStateGraph")
langfuse_handler = CallbackHandler()


class SpotifyCuratorGraphService:
    """
    Serviço principal de orquestração do agente no LangGraph.
    Delega a persistência de dados para ConversationRepository e a lógica
    auxiliar/Spotify para agent_helpers.
    """

    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            yaml_path = str(Path(__file__).resolve().parent / "spotify_curator_agent.yaml")

        with open(yaml_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Inicializa camada de banco de dados e checkpointer
        self.db = ConversationRepository()
        self.checkpointer = self.db.checkpointer

        # Compila o StateGraph
        self.graph = self._build_graph()

    # --- Fachada de Consulta ao Banco (consumida pelo Frontend/Streamlit) ---
    def list_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        return self.db.list_user_conversations(user_id=user_id, limit=limit)

    def get_conversation_details(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        return self.db.get_conversation_details(user_id=user_id, session_id=session_id)

    def list_user_playlists(self, user_id: str) -> List[Dict[str, Any]]:
        return self.db.list_user_playlists(user_id=user_id)

    # --- NÓ 1: Pesquisa e Curadoria Musical ---
    def curator_node(self, state: AgentState, config: RunnableConfig):
        configurable = config.get("configurable", {})
        llm = get_llm(configurable)
        research_llm = llm.bind_tools([web_search])
        struct_curation_llm = llm.with_structured_output(CurationOutput)

        sp = get_spotify_client(config)
        full_system_prompt = build_curator_prompt_context(
            sp=sp,
            state=state,
            base_system_prompt=self.config["system_prompt"],
        )
        messages = [SystemMessage(content=full_system_prompt)] + state["messages"]

        max_searches = self.config.get("model", {}).get("max_web_searches", MAX_WEB_SEARCHES_PER_TURN)
        if count_searches_since_last_human(state["messages"]) < max_searches:
            response = research_llm.invoke(messages)
            if response.tool_calls:
                return {"messages": [response]}
        else:
            logger.info(f"Limite de buscas ({max_searches}) atingido. Forçando síntese estruturada.")

        structured_res: CurationOutput = struct_curation_llm.invoke(messages)
        final_playlist_id = sanitize_playlist_id(
            structured_res.extracted_playlist_id, state.get("playlist_id")
        )
        formatted_reply = format_curation_reply(structured_res, final_playlist_id)

        return {
            "messages": [AIMessage(content=formatted_reply)],
            "operation_mode": structured_res.operation_mode,
            "playlist_id": final_playlist_id,
            "playlist_name": structured_res.suggested_playlist_name,
            "proposed_tracks": [t.model_dump() for t in structured_res.tracks],
            "is_approved": False,
        }

    # --- NÓ 2: Avaliação do Feedback Humano (Pós-Interrupção) ---
    def evaluate_human_feedback_node(self, state: AgentState, config: RunnableConfig):
        configurable = config.get("configurable", {})
        approval_llm = get_llm(configurable).with_structured_output(ApprovalDecision)

        decision: ApprovalDecision = approval_llm.invoke([
            SystemMessage(
                content=(
                    "Avalie a resposta do usuário sobre a playlist proposta. "
                    "Retorne is_approved=True APENAS se o usuário confirmar/aprovar explicitamente "
                    "(ex: 'sim', 'aprovo', 'pode criar', 'perfeito'). "
                    "Se o usuário solicitar qualquer alteração, troca, adição ou remoção de músicas, "
                    "retorne obrigatoriamente is_approved=False."
                )
            ),
            state["messages"][-1],
        ])
        return {"is_approved": decision.is_approved}

    # --- NÓ 3: Execução Determinística no Spotify ---
    def spotify_execution_node(self, state: AgentState, config: RunnableConfig):
        sp = get_spotify_client(config)
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id", "unknown_user")
        session_id = configurable.get("session_id", "unknown_session")

        sync_result = sync_tracks_to_spotify_playlist(
            sp=sp,
            mode=state.get("operation_mode", "create"),
            playlist_id=state.get("playlist_id"),
            playlist_name=state.get("playlist_name") or "Playlist AI Curator",
            proposed_tracks=state.get("proposed_tracks", []),
        )

        self.db.save_playlist(
            user_id=user_id,
            session_id=session_id,
            playlist_id=sync_result["playlist_id"],
            playlist_name=sync_result["playlist_name"],
            playlist_url=sync_result["playlist_url"],
            track_count=sync_result["added_count"],
        )

        return {
            "playlist_id": sync_result["playlist_id"],
            "playlist_name": sync_result["playlist_name"],
            "final_playlist_url": sync_result["playlist_url"],
            "messages": [AIMessage(content=sync_result["summary_msg"])],
        }

    # --- LÓGICA DE ROTAS E MONTAGEM DO GRAFO ---
    def route_after_curator(self, state: AgentState) -> Literal["tools", "wait_human_approval"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "wait_human_approval"

    def route_after_human_feedback(self, state: AgentState) -> Literal["spotify_execution", "curator"]:
        return "spotify_execution" if state.get("is_approved") else "curator"

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("curator", self.curator_node)
        workflow.add_node("tools", ToolNode([web_search]))
        workflow.add_node("evaluate_feedback", self.evaluate_human_feedback_node)
        workflow.add_node("spotify_execution", self.spotify_execution_node)

        workflow.add_edge(START, "curator")
        workflow.add_conditional_edges(
            "curator",
            self.route_after_curator,
            {"tools": "tools", "wait_human_approval": "evaluate_feedback"},
        )
        workflow.add_edge("tools", "curator")
        workflow.add_conditional_edges(
            "evaluate_feedback",
            self.route_after_human_feedback,
            {"spotify_execution": "spotify_execution", "curator": "curator"},
        )
        workflow.add_edge("spotify_execution", END)

        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["evaluate_feedback"],
        )

    def _fallback_synthesis_on_recursion_error(self, run_config: RunnableConfig) -> Dict[str, Any]:
        """Sintetiza a proposta estruturada caso GraphRecursionError ocorra."""
        state_values = self.graph.get_state(run_config).values or {}
        struct_curation_llm = get_llm(run_config.get("configurable", {})).with_structured_output(CurationOutput)

        clean_history = [
            m for m in state_values.get("messages", [])
            if not (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
        ]
        messages = [
            SystemMessage(
                content=(
                    self.config["system_prompt"]
                    + "\n\n[FALLBACK DE LIMITE DE BUSCA]: Gere imediatamente a lista "
                    "estruturada de músicas usando as informações já coletadas e seu conhecimento interno."
                )
            )
        ] + clean_history

        structured_res: CurationOutput = struct_curation_llm.invoke(messages, config=run_config)
        final_playlist_id = sanitize_playlist_id(
            structured_res.extracted_playlist_id, state_values.get("playlist_id")
        )
        updated_values = {
            "messages": [AIMessage(content=format_curation_reply(structured_res, final_playlist_id))],
            "operation_mode": structured_res.operation_mode,
            "playlist_id": final_playlist_id,
            "playlist_name": structured_res.suggested_playlist_name,
            "proposed_tracks": [t.model_dump() for t in structured_res.tracks],
            "is_approved": False,
        }

        self.graph.update_state(run_config, updated_values, as_node="curator")
        return self.graph.get_state(run_config).values

    def execute_query(
        self,
        user_id: str,
        session_id: str,
        spotify_access_token: str,
        user_message: str,
        custom_gemini_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Executa um turno da conversa no StateGraph, mede telemetria e persiste no banco."""
        metrics_callback = ExecutionMetricsCallback()
        start_time = time.perf_counter()

        thread_id = f"{user_id}:{session_id}"
        run_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id,
                "session_id": session_id,
                "spotify_access_token": spotify_access_token,
                "gemini_api_key": custom_gemini_key,
            },
            "callbacks": [langfuse_handler, metrics_callback],
            "recursion_limit": self.config["model"].get("max_steps", 30),
        }

        current_state = self.graph.get_state(run_config)
        clean_message = user_message.strip() if user_message else ""

        try:
            if current_state.next and "evaluate_feedback" in current_state.next:
                if not clean_message:
                    state_values = current_state.values
                    return {
                        "reply": "Por favor, informe se aprova a playlist (digite 'Sim') ou quais ajustes deseja fazer.",
                        "proposed_tracks": state_values.get("proposed_tracks", []),
                        "playlist_url": state_values.get("final_playlist_url"),
                    }

                self.graph.update_state(
                    run_config,
                    {"messages": [HumanMessage(content=clean_message)]},
                    as_node="curator",
                )
                result_state = self.graph.invoke(None, config=run_config)
            else:
                if not clean_message:
                    return {
                        "reply": "Por favor, digite o que você gostaria de ouvir para começarmos.",
                        "proposed_tracks": [],
                        "playlist_url": None,
                    }

                result_state = self.graph.invoke(
                    {"messages": [HumanMessage(content=clean_message)]},
                    config=run_config,
                )
        except GraphRecursionError as e:
            logger.warning(f"GraphRecursionError na thread {thread_id}: {e}. Executando fallback.")
            result_state = self._fallback_synthesis_on_recursion_error(run_config)

        elapsed_seconds = time.perf_counter() - start_time
        assistant_reply = next(
            (
                msg.content if isinstance(msg.content, str) else str(msg.content)
                for msg in reversed(result_state.get("messages", []))
                if isinstance(msg, AIMessage) and msg.content
            ),
            "",
        )

        if not assistant_reply.strip():
            proposed = result_state.get("proposed_tracks", [])
            if proposed:
                tracks_str = "\n".join(
                    [f"{i}. **{t['title']}** — *{t['artist']}*" for i, t in enumerate(proposed, 1)]
                )
                assistant_reply = (
                    f"Aqui está a seleção atualizada:\n\n{tracks_str}\n\n"
                    "⚠️ **Aprovação necessária:** Você aprova a criação/atualização da playlist com essas músicas? "
                    "Digite **'Sim'** para confirmar ou informe novos ajustes."
                )
            else:
                assistant_reply = "Não foi possível gerar a resposta textual neste turno. Por favor, tente novamente."

        llm_time = round(metrics_callback.llm_latency_seconds, 3)
        metrics_payload = {
            "total_seconds": round(elapsed_seconds, 3),
            "llm_seconds": llm_time,
            "external_seconds": round(max(0.0, elapsed_seconds - llm_time), 3),
            "total_tokens": metrics_callback.total_tokens,
        }

        self.db.save_turn(
            user_id=user_id,
            session_id=session_id,
            user_message=clean_message,
            assistant_reply=assistant_reply,
            result_state=result_state,
            metrics=metrics_payload,
        )

        return {
            "reply": assistant_reply,
            "proposed_tracks": result_state.get("proposed_tracks", []),
            "playlist_url": result_state.get("final_playlist_url"),
            "metrics": metrics_payload,
        }
