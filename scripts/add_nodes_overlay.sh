#!/usr/bin/env bash
set -euo pipefail
APP="app/streamlit_app.py"
BK="app/streamlit_app.nodes_backup_$(date +%Y%m%d-%H%M%S).py"
cp "$APP" "$BK"

# Insert our overlay RIGHT BEFORE the first st.plotly_chart(fig, ...)
awk '
BEGIN{done=0}
{
  if (!done && $0 ~ /st\.plotly_chart\(fig.*\)/) {
    print "";
    print "# ====== Nodes overlay (auto-inserted) ======";
    print "with st.sidebar.expander(\"Nodes overlay\", expanded=False):";
    print "    show_core  = st.checkbox(\"Show core nodes\", value=True)";
    print "    show_outer = st.checkbox(\"Show outer nodes\", value=False)";
    print "    node_stride = st.slider(\"Node stride\", 1, 8, 3)";
    print "    node_size   = st.slider(\"Marker size\", 1, 6, 2)";
    print "";
    print "def _mask_points(mask: np.ndarray, stride: int):";
    print "    # returns x,y,z indices subsampled by stride";
    print "    z, y, x = np.where(mask)";
    print "    if z.size == 0:";
    print "        return np.array([]), np.array([]), np.array([])";
    print "    sel = (np.arange(z.size) % stride) == 0";
    print "    return x[sel], y[sel], z[sel]";
    print "";
    print "try:";
    print "    if show_core:";
    print "        _x,_y,_z = _mask_points(core_mask, node_stride)";
    print "        if _x.size:";
    print "            fig.add_trace(go.Scatter3d(x=_x, y=_y, z=_z,";
    print "                mode=\"markers\", name=\"core\",";
    print "                marker=dict(size=node_size), opacity=0.9))";
    print "    if show_outer:";
    print "        _x,_y,_z = _mask_points(outer_mask, node_stride)";
    print "        if _x.size:";
    print "            fig.add_trace(go.Scatter3d(x=_x, y=_y, z=_z,";
    print "                mode=\"markers\", name=\"outer\",";
    print "                marker=dict(size=node_size), opacity=0.6))";
    print "except NameError:";
    print "    # variables not in scope (e.g. file didn\\x27t load) — ignore";
    print "    pass";
    print "# ====== /Nodes overlay ======";
    print $0;  # original st.plotly_chart(...) call
    done=1;
  } else {
    print $0;
  }
}
END{
  if (!done) {
    print \"# [nodes overlay patch warning] st.plotly_chart(fig, ...) not found\" > \"/dev/stderr\";
    exit 1;
  }
}
' "$APP" > "$APP.tmp"

mv "$APP.tmp" "$APP"
echo "[patch] Nodes overlay inserted. Backup at: $BK"

# restart streamlit (best-effort)
pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null || true
nohup bash -c 'source .venv/bin/activate && \
  streamlit run app/streamlit_app.py --server.headless true \
  --server.address 0.0.0.0 --server.port 8501 >> logs/streamlit.log 2>&1' >/dev/null 2>&1 & disown
echo "[patch] Streamlit restarting. Tail logs with: tail -f logs/streamlit.log"
