"""
dashboard.py — Streamlit dashboard điểm danh
Chạy: streamlit run dashboard.py --server.port 8502
"""

import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta, timezone

# ── Config ──
ATTENDANCE_DB = os.environ.get("ATTENDANCE_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/attendance.db"))
PHOTOS_DIR    = os.environ.get("PHOTOS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/photos"))
VN_TZ = timezone(timedelta(hours=7))

st.set_page_config(
    page_title  = "Điểm danh BKTJSC",
    page_icon   = "🏢",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS tùy chỉnh ──
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #3b82f6;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e8eaf0; }
    .metric-label { font-size: 0.85rem; color: #888; margin-top: 4px; }
    .checkin-badge  { background:#166534; color:#86efac; padding:2px 10px; border-radius:99px; font-size:12px; }
    .checkout-badge { background:#7f1d1d; color:#fca5a5; padding:2px 10px; border-radius:99px; font-size:12px; }
    .absent-badge   { background:#374151; color:#9ca3af; padding:2px 10px; border-radius:99px; font-size:12px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  DB helpers
# ════════════════════════════════════════════
def get_conn():
    if not os.path.exists(ATTENDANCE_DB):
        return None
    conn = sqlite3.connect(ATTENDANCE_DB)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    if conn is None:
        return pd.DataFrame()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception as e:
        st.error(f"DB error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def get_all_employees() -> list[str]:
    df = query("SELECT name FROM employees ORDER BY name")
    return df["name"].tolist() if not df.empty else []


def get_attendance_today() -> pd.DataFrame:
    today = date.today().isoformat()
    return query("""
        SELECT e.name, e.title,
               a.check_in, a.check_out,
               a.checkin_score, a.checkout_score,
               a.checkin_photo, a.checkout_photo,
               a.employee_id
        FROM attendance_log a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.date = ?
        ORDER BY a.check_in ASC
    """, (today,))


def get_attendance_by_date(target_date: str) -> pd.DataFrame:
    return query("""
        SELECT e.name, e.title,
               a.check_in, a.check_out,
               a.checkin_score, a.checkout_score,
               a.employee_id, a.date
        FROM attendance_log a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.date = ?
        ORDER BY a.check_in ASC
    """, (target_date,))


def get_attendance_range(start: str, end: str) -> pd.DataFrame:
    return query("""
        SELECT e.name, e.title,
               a.date, a.check_in, a.check_out,
               a.checkin_score
        FROM attendance_log a
        JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.date BETWEEN ? AND ?
        ORDER BY a.date DESC, a.check_in ASC
    """, (start, end))


def get_checkin_hours_this_week() -> pd.DataFrame:
    """Lấy giờ vào của tất cả nhân viên trong 7 ngày qua."""
    start = (date.today() - timedelta(days=6)).isoformat()
    end   = date.today().isoformat()
    df    = get_attendance_range(start, end)
    if df.empty:
        return df
    df["check_in_dt"] = pd.to_datetime(df["check_in"])
    df["hour"]        = df["check_in_dt"].dt.hour + df["check_in_dt"].dt.minute / 60
    df["day"]         = pd.to_datetime(df["date"]).dt.strftime("%a %d/%m")
    return df


# ════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.shields.io/badge/BKTJSC-Attendance-blue", use_column_width=True)
    st.title("🏢 Điểm danh")

    page = st.radio("Trang", [
        "📋 Hôm nay",
        "📅 Theo ngày",
        "📊 Biểu đồ tuần",
    ])

    st.divider()
    st.caption(f"🕐 {datetime.now(VN_TZ).strftime('%H:%M:%S  %d/%m/%Y')}")
    st.caption(f"DB: `{ATTENDANCE_DB}`")

    if st.button("🔄 Làm mới"):
        st.rerun()


# ════════════════════════════════════════════
#  Trang 1: Hôm nay
# ════════════════════════════════════════════
if page == "📋 Hôm nay":
    st.title(f"📋 Điểm danh hôm nay — {date.today().strftime('%d/%m/%Y')}")

    df = get_attendance_today()
    all_emps = get_all_employees()
    total    = len(all_emps)
    checked  = len(df)
    absent   = total - checked
    checked_out = df["check_out"].notna().sum() if not df.empty else 0

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Tổng nhân viên", total)
    with c2:
        st.metric("✅ Đã check-in", checked,
                  delta=f"{checked/total*100:.0f}%" if total > 0 else "0%")
    with c3:
        st.metric("🚪 Đã check-out", int(checked_out))
    with c4:
        st.metric("❌ Vắng mặt", absent,
                  delta=f"-{absent}" if absent > 0 else None,
                  delta_color="inverse")

    st.divider()

    if df.empty:
        st.info("Chưa có dữ liệu điểm danh hôm nay.")
    else:
        # ── Bảng điểm danh ──
        st.subheader("Danh sách điểm danh")

        for _, row in df.iterrows():
            with st.container():
                col_photo, col_info, col_time = st.columns([1, 3, 3])

                with col_photo:
                    ci_photo = row.get("checkin_photo")
                    co_photo = row.get("checkout_photo")
                    if ci_photo and os.path.exists(ci_photo):
                        st.image(ci_photo, caption="Vào", width=70)
                    if co_photo and os.path.exists(co_photo):
                        st.image(co_photo, caption="Ra", width=70)
                    if not ci_photo and not co_photo:
                        st.markdown("👤")

                with col_info:
                    st.markdown(f"**{row['name']}**")
                    st.caption(row.get("title") or "—")
                    score = row.get("checkin_score")
                    if score:
                        st.caption(f"Score: {score:.2%}")

                with col_time:
                    if row.get("check_in"):
                        ci = pd.to_datetime(row["check_in"]).strftime("%H:%M:%S")
                        st.markdown(f'🟢 Check-in: **{ci}**')
                    if row.get("check_out"):
                        co = pd.to_datetime(row["check_out"]).strftime("%H:%M:%S")
                        st.markdown(f'🔴 Check-out: **{co}**')
                        # Tính giờ công
                        ci_dt = pd.to_datetime(row["check_in"])
                        co_dt = pd.to_datetime(row["check_out"])
                        hours = (co_dt - ci_dt).total_seconds() / 3600
                        st.caption(f"⏱ {hours:.1f} giờ")
                    else:
                        st.markdown("🔴 Chưa check-out")

                st.divider()

    # ── Nhân viên vắng mặt ──
    if absent > 0:
        checked_names = set(df["name"].tolist()) if not df.empty else set()
        absent_names  = [n for n in all_emps if n not in checked_names]
        st.subheader(f"❌ Vắng mặt ({absent} người)")
        cols = st.columns(min(absent, 4))
        for i, name in enumerate(absent_names):
            with cols[i % 4]:
                st.markdown(f'<span class="absent-badge">👤 {name}</span>',
                            unsafe_allow_html=True)


# ════════════════════════════════════════════
#  Trang 2: Theo ngày
# ════════════════════════════════════════════
elif page == "📅 Theo ngày":
    st.title("📅 Lịch sử điểm danh theo ngày")

    col_date, col_btn = st.columns([3, 1])
    with col_date:
        selected_date = st.date_input("Chọn ngày",
            value=date.today(),
            max_value=date.today())
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        export = st.button("📥 Xuất CSV")

    df = get_attendance_by_date(selected_date.isoformat())

    if df.empty:
        st.info(f"Không có dữ liệu ngày {selected_date.strftime('%d/%m/%Y')}")
    else:
        # Xử lý columns hiển thị
        display = df.copy()
        display["check_in"]  = pd.to_datetime(display["check_in"]).dt.strftime("%H:%M:%S")
        display["check_out"] = pd.to_datetime(display["check_out"]).dt.strftime("%H:%M:%S").fillna("—")

        # Tính giờ công
        ci = pd.to_datetime(df["check_in"], errors="coerce")
        co = pd.to_datetime(df["check_out"], errors="coerce")
        display["giờ công"] = ((co - ci).dt.total_seconds() / 3600).round(1).astype(str)
        display.loc[display["giờ công"] == "nan", "giờ công"] = "—"

        display["score"] = df["checkin_score"].apply(
            lambda x: f"{x:.2%}" if pd.notna(x) else "—")

        st.dataframe(
            display[["name", "title", "check_in", "check_out", "giờ công", "score"]].rename(columns={
                "name"     : "Họ tên",
                "title"    : "Chức vụ",
                "check_in" : "Giờ vào",
                "check_out": "Giờ ra",
                "giờ công" : "Giờ công",
                "score"    : "Score",
            }),
            use_container_width=True,
            hide_index=True,
        )

        if export:
            csv = display.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label     = "Tải xuống CSV",
                data      = csv,
                file_name = f"diemdanh_{selected_date.isoformat()}.csv",
                mime      = "text/csv",
            )

        # Summary
        total = len(get_all_employees())
        c1, c2, c3 = st.columns(3)
        c1.metric("Có mặt", len(df))
        c2.metric("Vắng", total - len(df))
        gi_cong = ((co - ci).dt.total_seconds() / 3600).dropna()
        c3.metric("Giờ công TB", f"{gi_cong.mean():.1f}h" if len(gi_cong) > 0 else "—")


# ════════════════════════════════════════════
#  Trang 3: Biểu đồ tuần
# ════════════════════════════════════════════
elif page == "📊 Biểu đồ tuần":
    st.title("📊 Biểu đồ giờ vào theo tuần")

    # Chọn khoảng thời gian
    col1, col2 = st.columns(2)
    with col1:
        end_date   = st.date_input("Đến ngày", value=date.today())
    with col2:
        start_date = st.date_input("Từ ngày",
            value=date.today() - timedelta(days=6),
            max_value=end_date)

    df = get_checkin_hours_this_week()

    if df.empty:
        st.info("Chưa có dữ liệu trong khoảng thời gian này.")
    else:
        # ── Biểu đồ scatter: giờ vào theo ngày ──
        st.subheader("Giờ check-in từng ngày")
        fig1 = px.scatter(
            df, x="day", y="hour",
            color="name",
            hover_data={"name": True, "hour": ":.2f", "day": True},
            labels={"day": "Ngày", "hour": "Giờ vào", "name": "Nhân viên"},
            height=400,
        )
        fig1.add_hline(y=8.5, line_dash="dash", line_color="red",
                       annotation_text="8:30 (muộn giờ)", annotation_position="top right")
        fig1.update_yaxes(range=[5, 13], tickvals=list(range(6, 13)),
                          ticktext=[f"{h:02d}:00" for h in range(6, 13)])
        fig1.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e8eaf0",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Biểu đồ bar: số người có mặt mỗi ngày ──
        st.subheader("Số người có mặt mỗi ngày")
        daily_count = df.groupby("day")["name"].nunique().reset_index()
        daily_count.columns = ["Ngày", "Số người"]
        total_emp = len(get_all_employees())

        fig2 = px.bar(
            daily_count, x="Ngày", y="Số người",
            text="Số người",
            color="Số người",
            color_continuous_scale=["#ef4444", "#f97316", "#22c55e"],
            range_color=[0, total_emp],
            height=300,
        )
        fig2.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e8eaf0",
            coloraxis_showscale=False,
        )
        fig2.update_traces(textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

        # ── Biểu đồ histogram: phân phối giờ vào ──
        st.subheader("Phân phối giờ vào")
        fig3 = px.histogram(
            df, x="hour",
            nbins=20,
            labels={"hour": "Giờ vào", "count": "Số lần"},
            height=280,
            color_discrete_sequence=["#3b82f6"],
        )
        fig3.add_vline(x=8.5, line_dash="dash", line_color="red",
                       annotation_text="8:30")
        fig3.update_xaxes(range=[6, 13],
                          tickvals=list(range(6, 13)),
                          ticktext=[f"{h:02d}:00" for h in range(6, 13)])
        fig3.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="#e8eaf0",
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Bảng tóm tắt đi muộn ──
        st.subheader("Tóm tắt đi muộn (sau 8:30)")
        late_df = df[df["hour"] > 8.5].copy()
        if late_df.empty:
            st.success("Không có ai đi muộn trong khoảng thời gian này! 🎉")
        else:
            late_summary = late_df.groupby("name").agg(
                so_lan_di_muon=("hour", "count"),
                gio_muon_tb=("hour", lambda x: f"{(x - 8.5).mean() * 60:.0f} phút"),
            ).reset_index().rename(columns={
                "name"           : "Nhân viên",
                "so_lan_di_muon" : "Số lần đi muộn",
                "gio_muon_tb"    : "Muộn TB",
            })
            st.dataframe(late_summary, use_container_width=True, hide_index=True)


# ── Auto refresh ──
st.markdown("---")
col_refresh, col_info = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 Làm mới dữ liệu"):
        st.rerun()
with col_info:
    st.caption("Dữ liệu từ attendance.db · Cập nhật thủ công hoặc F5")