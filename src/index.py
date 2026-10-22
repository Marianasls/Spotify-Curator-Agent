from spotify_agent_service import SpotifyCuratorGraphService
import os
import sys

if __name__ == "__main__":
    user_access_token = ""
    service = SpotifyCuratorGraphService()

    response = {}
    query = ""
    while True:
        if not response:
            query = input("\nDescreva a playlist que você deseja (se a quantidade de músicas não for especificado, será escolhido 5 músicas): ")

        if query.lower() in ["sair", "exit", "quit", "q"]:
            break

        if query.strip():
            response = service.execute_query(
                user_id="user_marisls",
                session_id="session_01",
                spotify_access_token=user_access_token,
                user_message=query,
            )

            print(f"\nAgente:\n{response['reply']}")
        else:
            print("Por favor, digite uma resposta válida para continuar.")

        query = input("\nUsuário: ")
        
