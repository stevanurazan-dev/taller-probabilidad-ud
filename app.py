import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import binom, poisson

# Configuración de la página
st.set_page_config(page_title="Desafío de Ingeniería: Binomial vs Poisson", layout="wide")

st.title("📊 Laboratorio Interactivo: Aproximación de Poisson a la Binomial")
st.markdown("""
*Este tablero interactivo resuelve el Punto 2 del Taller de Distribuciones, demostrando la convergencia 
de una red de sensores de alta fidelidad mediante simulación estocástica y visualización algorítmica.*
""")

# Parámetros fijos según el enunciado para evitar renderizados infinitos en la carga inicial
st.sidebar.header("⚙️ Parámetros del Sistema")
n = st.sidebar.slider("Número de nodos (n)", min_value=10, max_value=3000, value=1500, step=10)
p = st.sidebar.slider("Probabilidad de fallo (p)", min_value=0.0005, max_value=0.05, value=0.002, step=0.0005, format="%.4f")
k_target = st.sidebar.number_input("Exactamente 'k' fallos a calcular", min_value=0, max_value=n, value=5)

lambda_param = n * p

# --- SECCIÓN 1: CÁLCULOS MATEMÁTICOS ---
st.header("1. Confrontación Analítica")

prob_binom = binom.pmf(k_target, n, p)
prob_poisson = poisson.pmf(k_target, lambda_param)
error_absoluto = abs(prob_binom - prob_poisson)

# Usamos contenedores fijos en lugar de columnas dinámicas para evitar el error de 'removeChild'
with st.container():
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Probabilidad Exacta (Binomial)", value=f"{prob_binom:.6f}")
    col2.metric(label="Aproximación (Poisson, λ = {0:.2f})".format(lambda_param), value=f"{prob_poisson:.6f}")
    col3.metric(label="Error Absoluto", value=f"{error_absoluto:.6e}")

# --- SECCIÓN 2: GRÁFICA DE CONVERGENCIA ---
st.header("2. Visualización Algorítmica de las Distribuciones")
st.markdown("Mueve los parámetros en la barra lateral para observar cómo divergen o convergen ambas distribuciones.")

# Rango de fallos optimizado
k_values = np.arange(max(0, int(lambda_param - 4*np.sqrt(lambda_param))), int(lambda_param + 4*np.sqrt(lambda_param)) + 5)
binom_pmf_vals = binom.pmf(k_values, n, p)
poisson_pmf_vals = poisson.pmf(k_values, lambda_param)

fig = go.Figure()
fig.add_trace(go.Bar(x=k_values, y=binom_pmf_vals, name='Binomial Exacta', opacity=0.7, marker_color='#1f77b4'))
fig.add_trace(go.Scatter(x=k_values, y=poisson_pmf_vals, name='Poisson (Límite)', mode='lines+markers', line=dict(color='#ff7f0e', width=3)))

fig.update_layout(
    xaxis_title="Número de nodos caídos (k)",
    yaxis_title="Probabilidad P(X = k)",
    barmode='overlay',
    legend=dict(x=0.8, y=0.9),
    margin=dict(l=20, r=20, t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN 3: SIMULACIÓN ESTOCÁSTICA ---
st.header("3. Simulación Estocástica en Vivo (Montecarlo)")
st.markdown("Vamos a simular el comportamiento real de los sensores ejecutando experimentos aleatorios masivos.")

num_simulaciones = st.selectbox("Número de iteraciones de la simulación", [10000, 50000, 100000], index=1)

# Inicializamos el estado para que la interfaz no intente redibujar elementos inexistentes antes de tiempo
if 'simulado' not in st.session_state:
    st.session_state.simulado = False

if st.button("🚀 Ejecutar Simulación de Montecarlo"):
    with st.spinner("Simulando fallos en la red..."):
        simulaciones = np.random.binomial(n, p, num_simulaciones)
        st.session_state.exitos_simulados = int(np.sum(simulaciones == k_target))
        st.session_state.prob_simulada = float(st.session_state.exitos_simulados / num_simulaciones)
        st.session_state.simulado = True

if st.session_state.simulado:
    st.success("¡Simulación completada con éxito!")
    with st.container():
        col_sim1, col_sim2 = st.columns(2)
        col_sim1.markdown(f"**Resultados del experimento:**\n* En **{num_simulaciones:}** redes simuladas de {n} nodos...\n* En **{st.session_state.exitos_simulados:}** ocasiones fallaron exactamente {k_target} nodos.")
        col_sim2.metric(label="Probabilidad Empírica (Simulada)", value=f"{st.session_state.prob_simulada:.6f}", 
                        delta=f"Dif. Teórica: {abs(st.session_state.prob_simulada - prob_binom):.6f}", delta_color="inverse")