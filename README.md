# 🎧 Spotify AI Music Curator

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agent-1C3C3C?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/LLM-Google_Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)

Agente autônomo com arquitetura **Human-in-the-Loop (HITL)** para curadoria musical e automação de playlists no Spotify. Construído com **LangGraph** e **Google Gemini**, o sistema combina o perfil de consumo do usuário, pesquisa web controlada e validação interativa antes de executar alterações na conta do Spotify.

---

## ✨ Principais Funcionalidades

- **Curadoria Contextualizada**: Extrai automaticamente os artistas e gêneros mais ouvidos pelo usuário na API do Spotify para calibrar as recomendações.
- **Human-in-the-Loop com Trava de UI**: O grafo interrompe a execução (`interrupt_before`) antes de criar ou modificar playlists. A interface bloqueia o campo de chat e exibe botões de ação explícitos (**Confirmar Playlist** ou **Solicitar Alterações**).
- **Resiliência e Controle de Tool Calling**: Limite determinístico de buscas web por turno (`max_web_searches`) e fallback automático para `GraphRecursionError`, garantindo síntese de resposta mesmo em consultas complexas.
- **Histórico Completo de Conversas e Playlists**: Persistência relacional de sessões, mensagens e catálogo de playlists geradas por usuário, com visualização dedicada em modal na interface.
- **Deduplicação Determinística**: Verificação de URIs pré-existentes no modo de atualização (`append`) para evitar músicas duplicadas.
- **Modelo Híbrido de Uso (Free Tier + BYOK)**: Suporte a cota gratuita de demonstração com opção *Bring Your Own Key* (chave da API Gemini do próprio usuário).
- **Observabilidade e Métricas**: Rastreamento de execuções via **Langfuse** e telemetria por mensagem (latência de LLM vs. APIs externas e contagem de tokens).

---

## 🏗️ Arquitetura e Fluxo do Grafo

```text
[Início] ──► (curator_node) ◄──► (tools: DuckDuckGo Search [máx. 2x/turno])
                   │
                   ▼ [interrupt_before: aguarda decisão na UI]
         (evaluate_feedback_node)
            ├──► [Aprovado] ──────► (spotify_execution_node) ──► [Fim]
            └──► [Pediu Ajustes] ─► Retorna ao (curator_node)
```

### Persistência de Dados (PostgreSQL + Fallback em Memória)
Além das tabelas internas de checkpoint do LangGraph (`PostgresSaver`), a aplicação gerencia três tabelas relacionais (com fallback automático em memória quando `DATABASE_URL` não está definida):
1. `user_conversations`: Metadados da sessão, status de aprovação e resumo para listagem na barra lateral.
2. `conversation_messages`: Histórico integral de mensagens (`user` / `assistant`) e métricas de execução (`JSONB`).
3. `user_playlists`: Registro consolidado das playlists criadas pelo usuário (nome, URL e data de criação) exibidas em modal.

---

## 📁 Estrutura do Repositório

```text
├── src/
│   ├── spotify_agent_service.py     # Orquestração do StateGraph, HITL e persistência PostgreSQL
│   ├── spotify_auth.py              # Gerenciamento de autenticação OAuth2 Spotipy
│   ├── spotify_curator_agent.yaml   # Prompt de sistema, hiperparâmetros e guardrails
│   ├── metricsCallback.py           # Callback de aferição de tokens e latência
│   └── index.py                     # Interface de linha de comando (CLI)
├── web/
│   └── app.py                       # Interface web interativa (Streamlit)
├── requirements.txt                 # Dependências Python
├── Dockerfile                       # Configuração de imagem de container
└── .env.example                     # Modelo de variáveis de ambiente
```

---

## ⚙️ Configuração do Ambiente

### Pré-requisitos
- **Python 3.11+**
- Credenciais de aplicativo no [Spotify Developer Dashboard](https://developer.spotify.com/dashboard) (configurar a Redirect URI como `http://localhost:8501/callback` para Web ou `http://localhost:8888/callback` para CLI).
- Chave de API do [Google AI Studio (Gemini)](https://aistudio.google.com/).
- *(Opcional)* Instância PostgreSQL (local, Supabase ou Neon) e conta no [Langfuse](https://langfuse.com/).

### Variáveis de Ambiente (`.env`)

Crie um arquivo `.env` na raiz do projeto com base no modelo abaixo:

```ini
# Google Gemini
GEMINI_API_KEY="sua_chave_gemini"
GEMINI_MODEL="gemini-2.5-flash"

# Spotify OAuth2
SPOTIPY_CLIENT_ID="seu_client_id_spotify"
SPOTIPY_CLIENT_SECRET="seu_client_secret_spotify"
SPOTIPY_REDIRECT_URI="http://localhost:8501/callback"

# Banco de Dados (Opcional - usa memória caso omitido)
DATABASE_URL="postgresql://usuario:senha@host:5432/dbname"

# Limite de uso gratuito na interface Web
MAX_FREE_USES=3

# Observabilidade Langfuse (Opcional)
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_HOST="https://cloud.langfuse.com"
```

---

## 🚀 Instalação e Execução

### Execução Local

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Iniciar a aplicação Web (Streamlit)
streamlit run web/app.py
```

A aplicação web ficará disponível em `http://localhost:8501`. Para testar via terminal, execute `python src/index.py`.

### Execução via Docker

```bash
# Construir a imagem
docker build -t spotify-music-curator .

# Iniciar o container
docker run -p 8501:8501 --env-file .env spotify-music-curator
```