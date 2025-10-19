#!/bin/bash
set -euo pipefail

APP_FILE="app/streamlit_app.py"
BACKUP="app/streamlit_app_backup_$(date +%Y%m%d-%H%M%S).py"

echo "[INFO] Backing up current $APP_FILE -> $BACKUP"
cp "$APP_FILE" "$BACKUP"

# Insert overlay code after the isosurface plotly_chart
awk '
/st\.plotly_chart\(fig, use_container_width=True\)/ {
    print $0;
    print "";
    print "# ====== Edge overlay (auto-inserted) ======";
    print "import pandas as pd, numpy as np";
    print "from pathlib import Path";
    print "from plotly import graph_objects as go";
    print "_run_dir = Path('/home/busbar/fuka-runs/runs')/run_id";
    print "_shards_dir = _run_dir/'shards'";
    print "@st.cache_data(show_spinner=True, ttl=30)";
    print "def _load_edges_epoch(ep, max_rows=8000):";
    print "    import pyarrow.parquet as pq";
    print "    dfs=[];";
    print "    import glob";
    print "    for f in glob.glob(str(_shards_dir/'edges_*.parquet')):";
    print "        try:";
    print "            df=pd.read_parquet(f, engine='pyarrow')";
    print "        except Exception:";
    print "            continue";
    print "        dfe=df[df['epoch']==int(ep)]";
    print "        if not dfe.empty: dfs.append(dfe)";
    print "    if not dfs: return pd.DataFrame()";
    print "    out=pd.concat(dfs).head(max_rows)";
    print "    return out";
    print "with st.sidebar.expander('Edges overlay', expanded=False):";
    print "    show_edges=st.checkbox('Show edges', value=True)";
    print "    color_key=st.selectbox('Color by', ['C','B','A','E'], index=0)";
    print "    max_edges=st.slider('Max edges',1000,20000,5000,step=1000)";
    print "    edge_opacity=st.slider('Edge opacity (%)',10,100,50)/100";
    print "if show_edges:";
    print "    edges=_load_edges_epoch(epoch, max_rows=max_edges)";
    print "    if not edges.empty:";
    print "        X=np.vstack([edges.x_u,edges.x_v,np.full_like(edges.x_u,np.nan)]).T.reshape(-1)";
    print "        Y=np.vstack([edges.y_u,edges.y_v,np.full_like(edges.y_u,np.nan)]).T.reshape(-1)";
    print "        Z=np.vstack([edges.z_u,edges.z_v,np.full_like(edges.z_u,np.nan)]).T.reshape(-1)";
    print "        C=edges[color_key].to_numpy(dtype=float)";
    print "        cmin,cmax=np.percentile(C,5),np.percentile(C,95)";
    print "        if cmax<=cmin:cmax=cmin+1e-9";
    print "        Cn=(np.clip(C,cmin,cmax)-cmin)/(cmax-cmin)";
    print "        Cline=np.repeat(Cn,3);Cline[2::3]=np.nan";
    print "        fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='lines',line=dict(width=2,color=Cline,colorscale='Viridis'),opacity=edge_opacity,name=f'Edges ({len(edges)})'))";
    print "        st.plotly_chart(fig,use_container_width=True)";
    print "# ====== End overlay ======";
    next
}
{print}
' "$BACKUP" > "$APP_FILE"

echo "[INFO] Restarting Streamlit..."
pkill -f "streamlit run .*app/streamlit_app.py" 2>/dev/null || true
nohup streamlit run app/streamlit_app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 \
  >> logs/streamlit.log 2>&1 & disown

sleep 3
tail -n 10 logs/streamlit.log
