import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

logger = logging.getLogger("SpotifyDatabase")


class ConversationRepository:
    """
    Gerencia a conexão com o PostgreSQL (ConnectionPool), inicialização do
    checkpointer do LangGraph e operações de leitura/escrita de conversas e playlists.
    Possui fallback transparente em memória quando DATABASE_URL não está definida.
    """

    def __init__(self, db_uri: Optional[str] = None):
        self.db_uri = db_uri or os.getenv("DATABASE_URL")
        self.pool: Optional[ConnectionPool] = None

        # Estruturas de fallback em memória para execução local sem banco
        self._memory_conversations: Dict[str, Dict[str, Any]] = {}
        self._memory_messages: Dict[str, List[Dict[str, Any]]] = {}
        self._memory_playlists: Dict[str, List[Dict[str, Any]]] = {}

        self.checkpointer = self._init_checkpointer()
        self.init_app_tables()

    def _init_checkpointer(self):
        """
        Configura o PostgresSaver com ConnectionPool se DATABASE_URL estiver definida.
        Caso contrário, utiliza MemorySaver para desenvolvimento local.
        """
        if self.db_uri:
            logger.info("Conectando ao PostgreSQL para persistência de estado (PostgresSaver)...")
            self.pool = ConnectionPool(
                conninfo=self.db_uri,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": None},
            )
            checkpointer = PostgresSaver(self.pool)
            checkpointer.setup()
            return checkpointer

        logger.warning("DATABASE_URL não encontrada. Usando MemorySaver (em memória).")
        return MemorySaver()

    def init_app_tables(self) -> None:
        """Cria as 3 tabelas de histórico de conversas e playlists no PostgreSQL."""
        if not self.pool:
            return

        ddl_statements = [
            """
            CREATE TABLE IF NOT EXISTS user_conversations (
                session_id VARCHAR(128) PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                title VARCHAR(255) NOT NULL,
                summary TEXT,
                operation_mode VARCHAR(20) DEFAULT 'create',
                proposed_tracks JSONB DEFAULT '[]'::jsonb,
                awaiting_approval BOOLEAN DEFAULT FALSE,
                playlist_url TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_user_conversations_user_updated
                ON user_conversations (user_id, updated_at DESC)
            """,
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id BIGSERIAL PRIMARY KEY,
                session_id VARCHAR(128) NOT NULL REFERENCES user_conversations(session_id) ON DELETE CASCADE,
                user_id VARCHAR(128) NOT NULL,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                metrics JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
                ON conversation_messages (session_id, id ASC)
            """,
            """
            CREATE TABLE IF NOT EXISTS user_playlists (
                id BIGSERIAL PRIMARY KEY,
                user_id VARCHAR(128) NOT NULL,
                session_id VARCHAR(128) REFERENCES user_conversations(session_id) ON DELETE SET NULL,
                playlist_id VARCHAR(128) NOT NULL,
                playlist_name VARCHAR(255) NOT NULL,
                playlist_url TEXT NOT NULL,
                track_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_user_playlists_user_created
                ON user_playlists (user_id, created_at DESC)
            """,
        ]

        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    for stmt in ddl_statements:
                        cur.execute(stmt, prepare=False)
        except Exception as e:
            logger.error(f"Erro ao inicializar tabelas de histórico no PostgreSQL: {e}")


    def save_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_reply: str,
        result_state: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Salva/atualiza o resumo da conversa e registra as mensagens do turno."""
        playlist_name = (result_state.get("playlist_name") or "").strip()
        default_title = user_message[:60] + ("..." if len(user_message) > 60 else "")
        title = playlist_name if playlist_name else default_title

        proposed_tracks = result_state.get("proposed_tracks", [])
        playlist_url = result_state.get("final_playlist_url")
        operation_mode = result_state.get("operation_mode", "create")
        awaiting_approval = bool(proposed_tracks and not playlist_url)
        summary = assistant_reply[:140].replace("\n", " ") + ("..." if len(assistant_reply) > 140 else "")
        now_iso = datetime.now(timezone.utc).isoformat()

        if not self.pool:
            conv = self._memory_conversations.get(session_id)
            created_at = conv["created_at"] if conv else now_iso
            final_title = playlist_name or (conv["title"] if conv else default_title)
            self._memory_conversations[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "title": final_title,
                "summary": summary,
                "operation_mode": operation_mode,
                "proposed_tracks": proposed_tracks,
                "awaiting_approval": awaiting_approval,
                "playlist_url": playlist_url,
                "created_at": created_at,
                "updated_at": now_iso,
            }
            msgs = self._memory_messages.setdefault(session_id, [])
            msgs.append({"role": "user", "content": user_message, "metrics": None, "created_at": now_iso})
            msgs.append({"role": "assistant", "content": assistant_reply, "metrics": metrics, "created_at": now_iso})
            return

        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_conversations (
                            session_id, user_id, title, summary, operation_mode,
                            proposed_tracks, awaiting_approval, playlist_url, created_at, updated_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW(), NOW())
                        ON CONFLICT (session_id) DO UPDATE SET
                            title = CASE
                                WHEN %s <> '' THEN %s
                                ELSE user_conversations.title
                            END,
                            summary = EXCLUDED.summary,
                            operation_mode = EXCLUDED.operation_mode,
                            proposed_tracks = EXCLUDED.proposed_tracks,
                            awaiting_approval = EXCLUDED.awaiting_approval,
                            playlist_url = COALESCE(EXCLUDED.playlist_url, user_conversations.playlist_url),
                            updated_at = NOW();
                        """,
                        (
                            session_id,
                            user_id,
                            title,
                            summary,
                            operation_mode,
                            json.dumps(proposed_tracks, ensure_ascii=False),
                            awaiting_approval,
                            playlist_url,
                            playlist_name,
                            playlist_name,
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO conversation_messages (session_id, user_id, role, content, metrics)
                        VALUES (%s, %s, 'user', %s, NULL),
                               (%s, %s, 'assistant', %s, %s::jsonb);
                        """,
                        (
                            session_id,
                            user_id,
                            user_message,
                            session_id,
                            user_id,
                            assistant_reply,
                            json.dumps(metrics, ensure_ascii=False),
                        ),
                    )
        except Exception as e:
            logger.error(f"Erro ao salvar turno da conversa {session_id}: {e}")

    def save_playlist(
        self,
        user_id: str,
        session_id: str,
        playlist_id: str,
        playlist_name: str,
        playlist_url: str,
        track_count: int,
    ) -> None:
        """Registra a playlist criada pelo usuário na tabela user_playlists."""
        now_iso = datetime.now(timezone.utc).isoformat()
        if not self.pool:
            user_list = self._memory_playlists.setdefault(user_id, [])
            user_list.insert(
                0,
                {
                    "playlist_id": playlist_id,
                    "playlist_name": playlist_name,
                    "playlist_url": playlist_url,
                    "track_count": track_count,
                    "created_at": now_iso,
                },
            )
            return

        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_conversations (session_id, user_id, title)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO NOTHING;
                        """,
                        (session_id, user_id, playlist_name),
                    )
                    cur.execute(
                        """
                        INSERT INTO user_playlists (
                            user_id, session_id, playlist_id, playlist_name, playlist_url, track_count
                        )
                        VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                        (user_id, session_id, playlist_id, playlist_name, playlist_url, track_count),
                    )
        except Exception as e:
            logger.error(f"Erro ao salvar playlist {playlist_id} no banco: {e}")

    def list_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna a lista de conversas do usuário com informações resumidas."""
        if not self.pool:
            convs = [
                c for c in self._memory_conversations.values() if c["user_id"] == user_id
            ]
            convs.sort(key=lambda x: x["updated_at"], reverse=True)
            return convs[:limit]

        try:
            with self.pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT
                            session_id,
                            title,
                            summary,
                            operation_mode,
                            awaiting_approval,
                            playlist_url,
                            created_at,
                            updated_at
                        FROM user_conversations
                        WHERE user_id = %s
                        ORDER BY updated_at DESC
                        LIMIT %s;
                        """,
                        (user_id, limit),
                    )
                    return list(cur.fetchall())
        except Exception as e:
            logger.error(f"Erro ao listar conversas do usuário {user_id}: {e}")
            return []

    def get_conversation_details(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Retorna todo o conteúdo de uma conversa específica para exibição na interface."""
        if not self.pool:
            conv = self._memory_conversations.get(session_id)
            if not conv or conv["user_id"] != user_id:
                return None
            return {
                **conv,
                "messages": self._memory_messages.get(session_id, []),
            }

        try:
            with self.pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT
                            session_id,
                            user_id,
                            title,
                            summary,
                            operation_mode,
                            proposed_tracks,
                            awaiting_approval,
                            playlist_url,
                            created_at,
                            updated_at
                        FROM user_conversations
                        WHERE session_id = %s AND user_id = %s;
                        """,
                        (session_id, user_id),
                    )
                    conv_row = cur.fetchone()
                    if not conv_row:
                        return None

                    cur.execute(
                        """
                        SELECT role, content, metrics, created_at
                        FROM conversation_messages
                        WHERE session_id = %s AND user_id = %s
                        ORDER BY id ASC;
                        """,
                        (session_id, user_id),
                    )
                    messages_rows = list(cur.fetchall())

                    return {
                        **conv_row,
                        "messages": messages_rows,
                    }
        except Exception as e:
            logger.error(f"Erro ao buscar detalhes da conversa {session_id}: {e}")
            return None

    def list_user_playlists(self, user_id: str) -> List[Dict[str, Any]]:
        """Retorna nome, URL e data de criação das playlists geradas pelo usuário."""
        if not self.pool:
            return [
                {
                    "playlist_name": p["playlist_name"],
                    "playlist_url": p["playlist_url"],
                    "created_at": p["created_at"],
                }
                for p in self._memory_playlists.get(user_id, [])
            ]

        try:
            with self.pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT
                            playlist_name,
                            playlist_url,
                            created_at
                        FROM user_playlists
                        WHERE user_id = %s
                        ORDER BY created_at DESC;
                        """,
                        (user_id,),
                    )
                    return list(cur.fetchall())
        except Exception as e:
            logger.error(f"Erro ao listar playlists do usuário {user_id}: {e}")
            return []
