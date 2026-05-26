import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import binom, poisson

# 1. Configuración inicial de la página web
st.set_page_config(page_title="Taller Probabilidad", layout="centered")

st.title("📊 Laboratorio Interactivo: Binomial vs Poisson")
st.markdown("### Universidad Distrital Francisco José de Caldas")
st.markdown("Solución interactiva y simulación estocástica para el Punto 2.")

# 2. Entrada de datos en la barra lateral
st.sidebar.header("⚙️ Parámetros del Sistema")
n = st.sidebar.slider("Número de nodos (n)", 10, 3000, 1500, 10)
p = st.sidebar.slider("Probabilidad de fallo (p)", 0.0005, 0.05, 0.002, 0.0005, format="%.4f")
k_target = st.sidebar.number_input("Exactamente 'k' fallos a calcular", value=5, min_value=0, max_value=n)

# 3. Cálculos matemáticos
lambda_param = n * p
prob_binom = binom.pmf(k_target, n, p)
prob_poisson = poisson.pmf(k_target, lambda_param)
error_absoluto = abs(prob_binom - prob_poisson)

# 4. Mostrar resultados analíticos
st.header("1. Confrontación Analítica")
st.write(f"**Probabilidad Exacta (Binomial):** {prob_binom:.6f}")
st.write(f"**Aproximación de Poisson (λ = {lambda_param:.2f}):** {prob_poisson:.6f}")
st.write(f"**Error Absoluto de la Aproximación:** {error_absoluto:.6e}")

# 5. Gráfico de Convergencia con Plotly
st.header("2. Visualización Algorítmica")
st.write("A continuación se muestra cómo se superponen ambas distribuciones:")

# Generar rango de datos para el gráfico
k_min = max(0, int(lambda_param - 4 * np.sqrt(lambda_param)))
k_max = int(lambda_param + 4 * np.sqrt(lambda_param)) + 5
k_values = np.arange(k_min, k_max)

binom_vals = binom.pmf(k_values, n, p)
poisson_vals = poisson.pmf(k_values, lambda_param)

fig = go.Figure()
fig.add_trace(go.Bar(x=k_values, y=binom_vals, name='Binomial (Real)'))
fig.add_trace(go.Scatter(x=k_values, y=poisson_vals, name='Poisson (Límite)', mode='lines+markers', line=dict(color='orange')))

fig.update_layout(xaxis_title="Fallos (k)", yaxis_title="Probabilidad")
st.plotly_chart(fig, use_container_width=True)

# 6. Simulación de Montecarlo
st.header("3. Simulación Estocástica (Montecarlo)")
st.write("Ejecuta experimentos aleatorios masivos en la red de sensores:")

num_sim = st.selectbox("Iteraciones de simulación", [10000, 50000, 100000], index=0)

if st.button("🚀 Ejecutar Simulación"):
    # Simulación directa usando generación binomial aleatoria masiva
    simulaciones = np.random.binomial(n, p, num_sim)
    exitos = np.sum(simulaciones == k_target)
    prob_sim = exitos / num_sim
    
    st.success("¡Simulación completada!")
    st.write(f"- En **{num_sim:,}** redes simuladas de {n} nodos...")
    st.write(f"- En **{exitos:,}** ocasiones fallaron exactamente {k_target} nodos.")
    st.write(f"- **Probabilidad Empírica Simulada:** {prob_sim:.6f}")