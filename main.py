from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import date # Importação para obter a data atual

import yfinance as yf
import pandas as pd
import numpy as np

# Inicialização do FastAPI
app = FastAPI()

# ----------------------------------------------------
# 1. Configurações e Rota Raiz
# ----------------------------------------------------

# Configuração CORS (Permite acesso de qualquer frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Rota de Boas-Vindas (GET /) - Resolve o erro "Not Found" na raiz
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de Diversificação de Portfólios está online! Use /docs para ver a documentação."}

# Rota de Verificação de Saúde (GET /health) - Para o Render Health Check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Modelo de dados para o POST
class Diversificacao(BaseModel):
    names: list[str]

# ----------------------------------------------------
# 2. Lógica de Processamento
# ----------------------------------------------------

def processaDiversificacao(tickers):
    # Data final é a data atual (ou último dia de negociação)
    end_date = date.today()

    # 1️⃣ Baixar preços históricos
    # Removido 'end="2025-08-08"' para evitar problemas de data no futuro/fuso horário
    dados = yf.download(tickers, start="2020-01-01", end=end_date.strftime("%Y-%m-%d"), auto_adjust=True)

    # 🛑 VERIFICAÇÃO CRÍTICA: Verifica se dados foram baixados
    if dados.empty:
        raise ValueError("Falha ao baixar dados. Verifique se os tickers estão corretos ou se há dados históricos disponíveis.")
        
    # Ajustar se multi-index
    if isinstance(dados.columns, pd.MultiIndex):
        if "Adj Close" in dados.columns.levels[0]:
            dados = dados.xs('Adj Close', axis=1, level=0)
        elif "Close" in dados.columns.levels[0]:
            dados = dados.xs('Close', axis=1, level=0)
        else:
            raise ValueError("Nenhuma coluna de preços Adj Close ou Close encontrada no DataFrame.")
    else:
        if "Adj Close" in dados.columns:
            dados = dados["Adj Close"]
        elif "Close" in dados.columns:
            dados = dados["Close"]
        else:
            raise ValueError("Nenhuma coluna de preços Adj Close ou Close encontrada no DataFrame")

    # 🛑 VERIFICAÇÃO CRÍTICA: Filtra tickers que não retornaram dados (colunas NaN)
    dados = dados.dropna(axis=1, how='all')

    if dados.empty:
         raise ValueError("Nenhum ticker fornecido ou todos os tickers falharam ao retornar dados válidos.")
         
    # 2️⃣ Retornos (Se o código chegou até aqui, os dados estão válidos)
    retornos = dados.pct_change().dropna()
    n_ativos = len(retornos.columns)
    n_carteiras = 20000

    # Lógica de Otimização de Portfólio (mantida a original)
    retornos_anuais = retornos.mean() * 252
    cov_matrix = retornos.cov() * 252

    retornos_carteira = []
    riscos_carteira = []
    pesos_carteira = []

    np.random.seed(42)
    for _ in range(n_carteiras):
        pesos = np.random.random(n_ativos)
        pesos /= np.sum(pesos)
        retorno = np.dot(pesos, retornos_anuais)
        risco = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        retornos_carteira.append(retorno)
        riscos_carteira.append(risco)
        pesos_carteira.append(pesos)

    retornos_carteira = np.array(retornos_carteira)
    riscos_carteira = np.array(riscos_carteira)
    pesos_carteira = np.array(pesos_carteira)

    # Perfis
    # (A lógica de perfis falharia se o array estivesse vazio,
    #  mas a verificação de dados vazios acima já protege contra isso)
    
    # Índice Conservador: Mínimo risco
    if not riscos_carteira.size:
        raise ValueError("Não foi possível gerar carteiras suficientes para análise.")

    idx_conservador = np.argmin(riscos_carteira)
    idx_arriscado = np.argmax(retornos_carteira)
    idx_moderado = np.argmin(
        np.abs((riscos_carteira - riscos_carteira.mean()) +
               (retornos_carteira - retornos_carteira.mean()))
    )

    # Monta resposta JSON (mantida a original)
    resposta = {
        "fronteira": [
            {
                "risco": float(risco),
                "retorno": float(retorno),
                "pesos": {
                    ativo: round(float(peso * 100), 2)
                    for ativo, peso in zip(retornos.columns, pesos)
                }
            }
            for risco, retorno, pesos in zip(riscos_carteira, retornos_carteira, pesos_carteira)
        ],
        "perfis": {
            "conservador": {
                "risco": float(riscos_carteira[idx_conservador]),
                "retorno": float(retornos_carteira[idx_conservador])
            },
            "moderado": {
                "risco": float(riscos_carteira[idx_moderado]),
                "retorno": float(retornos_carteira[idx_moderado])
            },
            "arriscado": {
                "risco": float(riscos_carteira[idx_arriscado]),
                "retorno": float(retornos_carteira[idx_arriscado])
            }
        }
    }

    return resposta

# ----------------------------------------------------
# 3. Endpoint Principal
# ----------------------------------------------------

@app.post("/diversificacao")
def create_item(item: Diversificacao):
    try:
        # Tenta executar a função de processamento
        resultado = processaDiversificacao(item.names)
        return resultado
    except ValueError as ve:
        # Captura erros de lógica de dados e retorna 400 Bad Request (Dados Inválidos)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Captura qualquer outro erro inesperado e retorna 500
        # O str(e) aqui deve ser a mensagem de erro do log
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento: {str(e)}")
