from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import yfinance as yf
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Modelo de dados para o POST
class Diversificacao(BaseModel):
    names: list[str]


def processaDiversificacao(tickers):
    # 1️⃣ Baixar preços históricos
    dados = yf.download(tickers, start="2020-01-01", end="2025-08-08", auto_adjust=True)

    # Ajustar se multi-index
    if isinstance(dados.columns, pd.MultiIndex):
        if "Adj Close" in dados.columns.levels[0]:
            dados = dados.xs('Adj Close', axis=1, level=0)
        elif "Close" in dados.columns.levels[0]:
            dados = dados.xs('Close', axis=1, level=0)
        else:
            raise ValueError("Nenhuma coluna de preços encontrada no DataFrame")
    else:
        if "Adj Close" in dados.columns:
            dados = dados["Adj Close"]
        elif "Close" in dados.columns:
            dados = dados["Close"]
        else:
            raise ValueError("Nenhuma coluna de preços encontrada no DataFrame")

    # 2️⃣ Retornos
    retornos = dados.pct_change().dropna()
    n_ativos = len(retornos.columns)
    n_carteiras = 20000

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
    idx_conservador = np.argmin(riscos_carteira)
    idx_arriscado = np.argmax(retornos_carteira)
    idx_moderado = np.argmin(
        np.abs((riscos_carteira - riscos_carteira.mean()) +
               (retornos_carteira - retornos_carteira.mean()))
    )

    # Monta resposta JSON
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


@app.post("/diversificacao")
def create_item(item: Diversificacao):
    try:
        resultado = processaDiversificacao(item.names)
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
