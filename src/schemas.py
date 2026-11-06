from typing import TypedDict, Annotated, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


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
        description="Mensagem apresentando a lista numerada e perguntando explicitamente se o usuário aprova"
    )


class ApprovalDecision(BaseModel):
    """Schema para classificar a resposta do usuário na etapa de HITL."""
    is_approved: bool = Field(
        description=(
            "True SOMENTE se o usuário aprovou explicitamente a lista atual sem pedir "
            "nenhuma alteração adicional. False se pediu trocas, adições, remoções ou ajustes."
        )
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
