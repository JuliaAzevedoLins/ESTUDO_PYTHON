"""
API REST para análise de carteiras de ações e geração da fronteira eficiente.
Endpoints:
  - GET /analise?tickers=PETR4.SA,VALE3.SA,ITSA4.SA
      Retorna os perfis Conservador, Moderado e Arriscado com pesos, retorno e risco.
  - GET /grafico
      Retorna o HTML do gráfico da fronteira eficiente.
"""

# Dependências: fastapi, uvicorn, yfinance, numpy, pandas, plotly, scikit-learn
from fastapi import FastAPI, Query, Response
from fastapi.responses import JSONResponse, HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import os

app = FastAPI()

# Função principal de análise das carteiras
# Recebe lista de tickers, simula carteiras, calcula perfis e gera gráfico
# Retorna dicionário com os perfis

def analisar_carteiras(tickers):
    # Baixar preços históricos
    dados = yf.download(tickers, start="2020-01-01", end="2025-08-08", auto_adjust=True)

    # Selecionar colunas de preços
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

    # Calcular retornos diários
    retornos = dados.pct_change().dropna()
    n_ativos = len(retornos.columns)
    n_carteiras = 5000
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

    # Identificar carteiras para perfis de investidor
    idx_conservador = np.argmin(riscos_carteira)
    idx_arriscado = np.argmax(retornos_carteira)
    idx_moderado = np.argmin(np.abs((riscos_carteira - riscos_carteira.mean()) +
                                    (retornos_carteira - retornos_carteira.mean())))

    # Criar DataFrame para Plotly
    df = pd.DataFrame({
        'Risco': riscos_carteira,
        'Retorno': retornos_carteira,
    })
    for i, ativo in enumerate(retornos.columns):
        df[ativo] = (pesos_carteira[:, i] * 100).round(2)
    df['Perfil'] = ''
    df.loc[idx_conservador, 'Perfil'] = 'Conservador'
    df.loc[idx_moderado, 'Perfil'] = 'Moderado'
    df.loc[idx_arriscado, 'Perfil'] = 'Arriscado'

    # Gerar gráfico Plotly com degradê azul, responsivo, sem colorbar e sem labels dos eixos
    fig = px.scatter(
        df,
        x='Risco',
        y='Retorno',
        color='Risco',
        color_continuous_scale=['#1976d2', '#3d5872', '#162c3f'],
        hover_data=[*retornos.columns],
    )
    fig.update_traces(marker=dict(size=14, opacity=0.7))
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        margin=dict(l=12, r=12, t=32, b=32),
        font=dict(size=18, family="Arial, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=None,
        yaxis_title=None
    )
    fig.update_coloraxes(showscale=False)
    # Salvar gráfico como HTML
    fig.write_html(
        "fronteira_eficiente.html",
        full_html=True,
        include_plotlyjs='cdn',
        config={
            "responsive": True,
            "displayModeBar": False,
            "scrollZoom": True,
        }
    )
    # Adiciona meta tag viewport para responsividade mobile
    with open("fronteira_eficiente.html", "r", encoding="utf-8") as f:
        html = f.read()
    if '<head>' in html:
        html = html.replace('<head>', '<head>\n    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">')
    with open("fronteira_eficiente.html", "w", encoding="utf-8") as f:
        f.write(html)

    # Montar resposta dos perfis
    def perfil_dict(idx):
        return {
            "Retorno": float(round(retornos_carteira[idx], 4)),
            "Risco": float(round(riscos_carteira[idx], 4)),
            "Pesos": {ativo: float(round(pesos_carteira[idx][i], 4)) for i, ativo in enumerate(retornos.columns)}
        }

    return {
        "Conservador": perfil_dict(idx_conservador),
        "Moderado": perfil_dict(idx_moderado),
        "Arriscado": perfil_dict(idx_arriscado)
    }

# Endpoint principal: análise dos perfis
@app.get("/analise", response_class=JSONResponse)
async def analise_endpoint(tickers: str = Query(..., description="Lista de tickers separada por vírgula")):
    """
    Endpoint que recebe tickers e retorna os perfis de investidor.
    Exemplo: /analise?tickers=PETR4.SA,VALE3.SA,ITSA4.SA
    """
    tickers_list = [t.strip() for t in tickers.split(",") if t.strip()]
    try:
        resultado = analisar_carteiras(tickers_list)
        return JSONResponse(content=resultado)
    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=400)

# Endpoint para servir o gráfico HTML
@app.get("/grafico", response_class=HTMLResponse)
async def grafico_endpoint():
    """
    Endpoint que retorna o HTML do gráfico da fronteira eficiente.
    """
    if not os.path.exists("fronteira_eficiente.html"):
        return Response(content="Gráfico não gerado ainda. Execute /analise primeiro.", media_type="text/plain", status_code=404)
    with open("fronteira_eficiente.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
