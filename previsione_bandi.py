import pandas as pd 
from prophet import Prophet
import plotly.graph_objects as go
from tabulate import tabulate
import os
import argparse

OUTPUT_HTML = "grafico_bandi_interattivo.html"
PREVISIONI_ANNI = 10


def carica_dati(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"⚠️ File non trovato: {path}")
    df = pd.read_excel(path)
    df['Data di pubblicazione'] = pd.to_datetime(df['Data di pubblicazione'], errors='coerce')
    df = df.dropna(subset=['Data di pubblicazione'])
    df['Anno'] = df['Data di pubblicazione'].dt.year
    return df


def prepara_dati_prophet(df):
    bandi_per_anno = df.groupby('Anno').size().reset_index(name='count')
    df_prophet = bandi_per_anno.rename(columns={'Anno': 'ds', 'count': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%Y')
    return df_prophet


def addestra_modello(df_prophet):
    model = Prophet(yearly_seasonality=True)
    model.fit(df_prophet)
    return model


def crea_previsioni(model, anni, ultimo_anno):
    future = model.make_future_dataframe(periods=anni, freq='Y')
    forecast = model.predict(future)
    previsioni = forecast[forecast['ds'].dt.year > ultimo_anno][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    previsioni['Anno'] = previsioni['ds'].dt.year
    return previsioni[['Anno', 'yhat', 'yhat_lower', 'yhat_upper']], forecast


def stampa_tabella(previsioni):
    print("\n📊 Previsioni future:")
    print(tabulate(previsioni.round(2), headers='keys', tablefmt='fancy_grid', showindex=False))


def crea_grafico(df_prophet, forecast, output_file):
    storico = go.Scatter(
        x=df_prophet['ds'], y=df_prophet['y'],
        mode='markers+lines', name='Dati storici',
        line=dict(color='royalblue'), marker=dict(size=6)
    )

    previsione = go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='Previsione',
        line=dict(color='darkblue', width=2)
    )

    confidenza = go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(135, 206, 250, 0.3)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
        name='Intervallo di confidenza'
    )

    fig = go.Figure(data=[confidenza, storico, previsione])

    fig.update_layout(
        title={
            'text': '📈 Previsione Annuale del Numero di Bandi Autobus',
            'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=22)
        },
        xaxis=dict(
            title='Anno', showspikes=True, spikecolor="grey",
            spikethickness=1, spikemode="across", showline=True,
            showgrid=True, ticks="outside", tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Numero Bandi', showspikes=True, spikecolor="grey",
            spikethickness=1, spikemode="across", tickfont=dict(size=12)
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=100, b=50),
        width=900, height=600,
        template="plotly_white"
    )

    fig.write_html(output_file)
    fig.show()
    print(f"\n✅ Grafico interattivo salvato come: {output_file}")


def main(file_path):
    df = carica_dati(file_path)
    df_prophet = prepara_dati_prophet(df)
    model = addestra_modello(df_prophet)
    ultimo_anno = df_prophet['ds'].max().year
    previsioni, forecast = crea_previsioni(model, PREVISIONI_ANNI, ultimo_anno)
    stampa_tabella(previsioni)
    crea_grafico(df_prophet, forecast, OUTPUT_HTML)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Previsione bandi autobus")
    parser.add_argument("--file", required=True, help="Percorso file Excel con dati")
    args = parser.parse_args()

    try:
        main(args.file)
    except Exception as e:
        print(f"❌ Errore: {e}")
