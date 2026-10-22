# Spotify Music Curator Agent 🎵🤖

Agente inteligente com suporte a **Human-in-the-Loop (HITL)** para curadoria musical automatizada no Spotify. Utiliza **LangGraph**, modelos **Google Gemini**, busca dinâmica via DuckDuckGo e integração com a Web API do Spotify para criar novas playlists ou enriquecer playlists existentes com base no perfil e gosto do usuário.

---

## 🚀 Funcionalidades

- **Curadoria Inteligente**: Recomendações orientadas por LLM (Google Gemini) com pesquisa em tempo real via DuckDuckGo.
- **Contexto de Preferências**: Lê automaticamente os artistas e gêneros mais ouvidos pelo usuário no Spotify.
- **Human-in-the-Loop (HITL)**: O fluxo pausa via interrupção de grafo (`interrupt_before`) para aprovação do usuário antes de realizar alterações na conta Spotify.
- **Deduplicação Determinística**: Verifica se as faixas sugeridas já constam na playlist para evitar repetições.
- **Persistência de Sessão**: Suporte a PostgreSQL (`PostgresSaver`) com fallback para memória (`MemorySaver`).
- **Observabilidade**: Tracing com Langfuse.

---

## 📁 Estrutura do Projeto

```text
├── src/
│   ├── spotify_agent_service.py     # Definição do StateGraph (LangGraph) e lógica HITL
│   ├── spotify_auth.py              # Autenticação Spotipy (OAuth / BYOK token)
│   ├── spotify_curator_agent.yaml   # Configurações do agente e system prompt
│   ├── metricsCallback.py           # Agregação de tokens e métricas de latência
│   └── index.py                     # CLI para testes locais via terminal
├── web/
│   └── app.py                       # Interface web interativa em Streamlit
├── requirements.txt
├── Dockerfile
└── .env.example
```

## 🛠️ Instalação e Execução Local
### Pré-requisitos
- Python 3.11+
- Conta de desenvolvedor no Spotify Developer Dashboard

### 1. Clonar e Instalar Dependências (Linux)
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Executar via Interface Web (Streamlit)
```
streamlit run web/app.py
```

### 3. Executar via Linha de Comando (CLI)
```
python src/index.py
```

## 🐳 Executando com Docker
```
# Construir a imagem:
docker build -t spotify-music-curator .

# Rodar o container
docker run -p 8501:8501 --env-file .env spotify-music-curator
```

A interface web estará disponível pelo link http://localhost:8501/