from spotify_agent_service import SpotifyCuratorGraphService

if __name__ == "__main__":
    user_access_token = ""
    service = SpotifyCuratorGraphService()

    while True:
        query = input("\nO que você quer ouvir: ")
        if query.lower() in ["sair", "exit"]:
            break

        response = service.execute_query(
            user_id="user_marisls",
            session_id="session_01",
            spotify_access_token=user_access_token,
            user_message=query,
        )

        print(f"\nAgente:\n{response['reply']}")
        
        m = response["metrics"]
        print(
            f"\n📊 [Métricas da Query] "
            f"⏱️ Tempo: {m['latency_seconds']}s ({m['latency_ms']} ms) | "
            f"🤖 Chamadas LLM: {m['llm_calls']} | "
            f"🪙 Tokens: {m['total_tokens']} (Prompt: {m['input_tokens']} / Resposta: {m['output_tokens']})"
        )