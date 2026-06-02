import streamlit as st
import pandas as pd
from io import BytesIO
from ortools.sat.python import cp_model

# =========================================================
# SAYFA AYARI
# =========================================================
st.set_page_config(
    page_title="Montaj Hattı Dengeleme",
    page_icon="🏭",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #2f3142;
}
.subtitle {
    font-size: 18px;
    color: #444;
    margin-bottom: 25px;
}
.metric-card {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #e6e6e6;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
}
.success-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #e9f8ef;
    border-left: 6px solid #20a35b;
}
.warning-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #fff4e5;
    border-left: 6px solid #ff9800;
}
.error-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #fdecea;
    border-left: 6px solid #e53935;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# VERİ
# =========================================================
I = range(1, 64)
J = range(1, 37)
W = range(1, 37)

t_raw = {
    1: 2.43,  2: 9.79,  3: 2.12,  4: 9.92,  5: 4.66,  6: 11.58,
    7: 1.01,  8: 1.44,  9: 9.66, 10: 10.30, 11: 0.49, 12: 7.13,
    13: 7.18, 14: 2.44, 15: 3.58, 16: 4.90, 17: 3.21, 18: 7.78,
    19: 11.27, 20: 11.35, 21: 0.80, 22: 3.31, 23: 9.83, 24: 0.80,
    25: 4.61, 26: 5.20, 27: 11.89, 28: 6.30, 29: 13.32, 30: 0.98,
    31: 14.20, 32: 6.13, 33: 0.98, 34: 14.49, 35: 3.14, 36: 12.12,
    37: 1.07, 38: 5.14, 39: 5.63, 40: 0.57, 41: 10.13, 42: 0.90,
    43: 1.39, 44: 1.43, 45: 0.51, 46: 10.74, 47: 5.65, 48: 7.38,
    49: 1.71, 50: 15.09, 51: 7.31, 52: 6.93, 53: 10.72, 54: 1.31,
    55: 6.45, 56: 2.39, 57: 0.89, 58: 11.06, 59: 8.02, 60: 6.48,
    61: 3.13, 62: 0.53, 63: 7.74
}

SCALE = 100
t = {i: int(round(t_raw[i] * SCALE)) for i in t_raw}
P = [(i, i + 1) for i in range(1, 63)]


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Ayarlar")

with st.sidebar.expander("🏗️ Hat Parametreleri", expanded=True):
    L = st.number_input("Maksimum Yürüme Mesafesi (L)", min_value=0, max_value=100, value=4, step=1)
    D = st.number_input("Hedef Üretim Miktarı (D)", min_value=1, max_value=500, value=32, step=1)
    T = st.number_input("Vardiya Süresi (T - dk)", min_value=1, max_value=1440, value=510, step=10)

with st.sidebar.expander("⚖️ Optimizasyon Kısıtları", expanded=True):
    U_MAX = st.slider("Maks. Operatör Doluluğu (U_MAX)", 0.10, 1.00, 1.00, 0.01)

with st.sidebar.expander("⏱️ Solver Ayarları", expanded=True):
    time_limit = st.slider("Senaryo Başına Süre Limiti (sn)", 5, 120, 30, 5)
    max_workers_to_solve = st.slider("Maksimum Denenecek Operatör Sayısı", 1, 36, 36, 1)

st.sidebar.divider()
epsilon_choice = st.sidebar.slider("Detaylı Rapor İçin Operatör Seç", 1, 36, 29, 1)


# =========================================================
# MODEL
# =========================================================
def solve_model(exact_workers=None, time_limit=30, L=4, D=32, T=510, U_MAX=1.0):
    BIG_M = sum(t.values())

    d = {
        j: {k: 2 * abs(j - k) for k in J}
        for j in J
    }

    model = cp_model.CpModel()

    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in I for j in J}
    y = {(w, j): model.NewBoolVar(f"y_{w}_{j}") for w in W for j in J}
    z = {w: model.NewBoolVar(f"z_{w}") for w in W}

    l = {j: model.NewIntVar(0, BIG_M, f"l_{j}") for j in J}
    q = {(w, j): model.NewIntVar(0, BIG_M, f"q_{w}_{j}") for w in W for j in J}

    C = model.NewIntVar(0, BIG_M, "C")

    for i in I:
        model.Add(sum(x[i, j] for j in J) == 1)

    for i, h in P:
        model.Add(
            sum(j * x[i, j] for j in J)
            <=
            sum(j * x[h, j] for j in J)
        )

    for j in J:
        model.Add(l[j] == sum(t[i] * x[i, j] for i in I))

    for j in J:
        model.Add(sum(y[w, j] for w in W) == 1)

    for w in W:
        for j in J:
            model.Add(y[w, j] <= z[w])

    for w in W:
        for j in J:
            model.Add(q[w, j] <= l[j])
            model.Add(q[w, j] <= BIG_M * y[w, j])
            model.Add(q[w, j] >= l[j] - BIG_M * (1 - y[w, j]))

    for w in W:
        model.Add(sum(q[w, j] for j in J) <= C)

    for j in J:
        model.Add(l[j] <= C)

    for w in W:
        for j in J:
            for k in J:
                if j < k and d[j][k] > L:
                    model.Add(y[w, j] + y[w, k] <= 1)

    if exact_workers is not None:
        model.Add(sum(z[w] for w in W) == exact_workers)

    # Operatör doluluğu kısıtı
    # U = D * iş yükü / T <= U_MAX
    for w in W:
        model.Add(D * sum(q[w, j] for j in J) <= int(U_MAX * T * SCALE))

    model.Minimize(C)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return None

    C_value = solver.Value(C) / SCALE

    solution = {
        "status": "Optimal" if status == cp_model.OPTIMAL else "Feasible",
        "C": C_value,
        "used_workers": sum(solver.Value(z[w]) for w in W),
        "stations_of_worker": {w: [] for w in W},
        "ops_of_station": {j: [] for j in J},
        "station_loads": {j: solver.Value(l[j]) / SCALE for j in J},
        "worker_load_per_product": {
            w: sum(solver.Value(q[w, j]) for j in J) / SCALE for w in W
        },
        "worker_load_per_shift": {
            w: D * sum(solver.Value(q[w, j]) for j in J) / SCALE for w in W
        },
        "worker_U": {
            w: 100 * ((D / T) * (sum(solver.Value(q[w, j]) for j in J) / SCALE))
            for w in W
        },
        "reachable_output": T / C_value if C_value > 0 else float("inf"),
        "meets_target": (T / C_value >= D - 1e-6) if C_value > 0 else True,
        "solver_wall_time": solver.WallTime(),
        "objective": solver.ObjectiveValue() / SCALE
    }

    for i in I:
        for j in J:
            if solver.Value(x[i, j]) == 1:
                solution["ops_of_station"][j].append(i)

    for w in W:
        for j in J:
            if solver.Value(y[w, j]) == 1:
                solution["stations_of_worker"][w].append(j)

    return solution


# =========================================================
# DATAFRAME FONKSİYONLARI
# =========================================================
def create_summary_df(results):
    rows = []

    for eps, res in results.items():
        if res is None:
            rows.append({
                "Operatör Sayısı": eps,
                "Çevrim Süresi": None,
                "Kullanılan Operatör": None,
                "Ulaşılabilir Üretim": None,
                "Hedef Durumu": "Uygun Değil",
                "Solver Durumu": "Infeasible"
            })
        else:
            rows.append({
                "Operatör Sayısı": eps,
                "Çevrim Süresi": round(res["C"], 2),
                "Kullanılan Operatör": res["used_workers"],
                "Ulaşılabilir Üretim": round(res["reachable_output"], 2),
                "Hedef Durumu": "Sağlanıyor" if res["meets_target"] else "Sağlanmıyor",
                "Solver Durumu": res["status"]
            })

    return pd.DataFrame(rows)


def create_station_df(res):
    rows = []

    for j in J:
        rows.append({
            "İstasyon": j,
            "Operasyonlar": ", ".join(map(str, res["ops_of_station"][j])),
            "İstasyon Yükü (dk)": round(res["station_loads"][j], 2),
            "Operasyon Sayısı": len(res["ops_of_station"][j])
        })

    return pd.DataFrame(rows)


def create_worker_df(res):
    rows = []

    for w in W:
        if len(res["stations_of_worker"][w]) > 0:
            rows.append({
                "Operatör": w,
                "Sorumlu İstasyonlar": ", ".join(map(str, res["stations_of_worker"][w])),
                "Ürün Başı Yük (dk)": round(res["worker_load_per_product"][w], 2),
                "Vardiya Yükü (dk)": round(res["worker_load_per_shift"][w], 2),
                "Verimlilik (%)": round(res["worker_U"][w], 2)
            })

    return pd.DataFrame(rows)


def create_distance_df(res, L):
    d = {
        j: {k: 2 * abs(j - k) for k in J}
        for j in J
    }

    rows = []

    for w in W:
        stations = res["stations_of_worker"][w]

        if len(stations) >= 2:
            for a in range(len(stations)):
                for b in range(a + 1, len(stations)):
                    j = stations[a]
                    k = stations[b]
                    distance = d[j][k]

                    rows.append({
                        "Operatör": w,
                        "İstasyon Çifti": f"{j}-{k}",
                        "Mesafe": distance,
                        "Limit": L,
                        "Durum": "Uygun" if distance <= L else "İhlal"
                    })

    return pd.DataFrame(rows)


def create_operations_df():
    rows = []

    for i, duration in t_raw.items():
        rows.append({
            "Operasyon": i,
            "Süre (dk)": duration
        })

    return pd.DataFrame(rows)


def export_excel(summary_df, station_df, worker_df, distance_df, operations_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Senaryo Özeti")
        station_df.to_excel(writer, index=False, sheet_name="İstasyon Atamaları")
        worker_df.to_excel(writer, index=False, sheet_name="Operatör Atamaları")
        distance_df.to_excel(writer, index=False, sheet_name="Mesafe Kontrolü")
        operations_df.to_excel(writer, index=False, sheet_name="Operasyon Süreleri")

    return output.getvalue()


# =========================================================
# ANA EKRAN
# =========================================================
st.markdown('<div class="main-title">🏭 Montaj Hattı Dengeleme & Operatör Atama Sistemi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Google OR-Tools CP-SAT Solver tabanlı gelişmiş optimizasyon arayüzü.</div>',
    unsafe_allow_html=True
)

run_button = st.button("🚀 Tüm Senaryoları Hesapla ve Analiz Et", type="primary")

if "results" not in st.session_state:
    st.session_state.results = None

if run_button:
    results = {}
    progress = st.progress(0)
    status_text = st.empty()

    for eps in range(1, max_workers_to_solve + 1):
        status_text.write(f"🔄 {eps} operatörlü senaryo çözülüyor...")
        results[eps] = solve_model(
            exact_workers=eps,
            time_limit=time_limit,
            L=L,
            D=D,
            T=T,
            U_MAX=U_MAX
        )
        progress.progress(eps / max_workers_to_solve)

    status_text.write("✅ Tüm senaryolar tamamlandı.")
    st.session_state.results = results


# =========================================================
# SONUÇLAR
# =========================================================
if st.session_state.results is not None:
    results = st.session_state.results
    summary_df = create_summary_df(results)

    feasible_results = {
        eps: res for eps, res in results.items()
        if res is not None
    }

    st.divider()
    st.subheader("📊 Genel Senaryo Özeti")

    if len(feasible_results) == 0:
        st.markdown(
            '<div class="error-box">Hiçbir operatör sayısı için uygun çözüm bulunamadı.</div>',
            unsafe_allow_html=True
        )
        st.dataframe(summary_df, use_container_width=True)
        st.stop()

    ideal_C = min(res["C"] for res in feasible_results.values())
    ideal_Z = min(res["used_workers"] for res in feasible_results.values())
    nadir_C = max(res["C"] for res in feasible_results.values())
    nadir_Z = max(res["used_workers"] for res in feasible_results.values())

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("İdeal Çevrim Süresi", f"{ideal_C:.2f} dk")

    with c2:
        st.metric("Minimum Operatör", f"{ideal_Z}")

    with c3:
        st.metric("Nadir Çevrim Süresi", f"{nadir_C:.2f} dk")

    with c4:
        st.metric("Maksimum Operatör", f"{nadir_Z}")

    st.dataframe(summary_df, use_container_width=True)

    chart_df = summary_df.dropna(subset=["Çevrim Süresi"])

    st.subheader("📈 Çevrim Süresi - Operatör Sayısı Grafiği")
    st.line_chart(
        chart_df,
        x="Operatör Sayısı",
        y="Çevrim Süresi"
    )

    st.subheader("📈 Ulaşılabilir Üretim Grafiği")
    st.bar_chart(
        chart_df,
        x="Operatör Sayısı",
        y="Ulaşılabilir Üretim"
    )

    # =====================================================
    # DETAYLI RAPOR
    # =====================================================
    st.divider()
    st.subheader(f"📌 Detaylı Senaryo Raporu | Operatör Sayısı = {epsilon_choice}")

    if epsilon_choice not in results or results[epsilon_choice] is None:
        st.markdown(
            f'<div class="error-box">{epsilon_choice} operatör için uygun çözüm bulunamadı.</div>',
            unsafe_allow_html=True
        )
    else:
        res = results[epsilon_choice]

        if res["meets_target"]:
            st.markdown(
                '<div class="success-box">Bu senaryo hedef üretim miktarını sağlamaktadır.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="warning-box">Bu senaryo hedef üretim miktarını sağlamamaktadır.</div>',
                unsafe_allow_html=True
            )

        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric("Çevrim Süresi", f"{res['C']:.2f} dk")

        with m2:
            st.metric("Kullanılan Operatör", f"{res['used_workers']}")

        with m3:
            st.metric("Üretim Kapasitesi", f"{res['reachable_output']:.2f}")

        with m4:
            st.metric("Hedef Üretim", f"{D}")

        with m5:
            st.metric("Solver Süresi", f"{res['solver_wall_time']:.2f} sn")

        station_df = create_station_df(res)
        worker_df = create_worker_df(res)
        distance_df = create_distance_df(res, L)
        operations_df = create_operations_df()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏭 İstasyon Atamaları",
            "👷 Operatör Atamaları",
            "📏 Mesafe Kontrolü",
            "⏱️ Operasyon Süreleri",
            "📥 Excel Rapor"
        ])

        with tab1:
            st.dataframe(station_df, use_container_width=True)

            st.subheader("İstasyon Yükleri Grafiği")
            st.bar_chart(
                station_df,
                x="İstasyon",
                y="İstasyon Yükü (dk)"
            )

        with tab2:
            st.dataframe(worker_df, use_container_width=True)

            st.subheader("Operatör Verimlilik Grafiği")
            st.bar_chart(
                worker_df,
                x="Operatör",
                y="Verimlilik (%)"
            )

        with tab3:
            if len(distance_df) > 0:
                st.dataframe(distance_df, use_container_width=True)

                violation_count = len(distance_df[distance_df["Durum"] == "İhlal"])

                if violation_count == 0:
                    st.success("Tüm mesafe kısıtları sağlanıyor.")
                else:
                    st.error(f"{violation_count} adet mesafe ihlali var.")
            else:
                st.info("Birden fazla istasyona atanmış operatör yok.")

        with tab4:
            st.dataframe(operations_df, use_container_width=True)

            st.subheader("Operasyon Süreleri")
            st.bar_chart(
                operations_df,
                x="Operasyon",
                y="Süre (dk)"
            )

        with tab5:
            excel_file = export_excel(
                summary_df,
                station_df,
                worker_df,
                distance_df,
                operations_df
            )

            st.download_button(
                label="📥 Excel Raporunu İndir",
                data=excel_file,
                file_name=f"montaj_hatti_raporu_{epsilon_choice}_operator.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    st.info("Başlamak için yukarıdaki butona basarak tüm senaryoları hesapla.")
