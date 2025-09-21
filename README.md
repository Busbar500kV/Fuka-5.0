# Fuka-5.0
First Universal Kommon Ancestor by යසස් පොන්වීර 


# Fuka 5.0 — Space–Time Capacitor Substrate (Cloud + UI)

Fuka 5.0 is a headless physics sandbox that evolves a **space–time capacitor network** inside a larger environment defined *only* by **charge density** ρ(x,t). The substrate learns **locally**:

- Adaptive capacitances **C** per edge (with floors/ceilings)
- **Bandwise attention gates** (split 1 vs 1′, and **mix/fuse**)
- **Edge batteries** (harvest→store→spend), **maturity/aging**, **self-rehearsal**
- **Thermal** node states (χ(T) consolidation, over-temp leak)
- LMS **decoders** (s1, s1′, fused)
- **Morphogenesis**: ρ grows where power harvest is high and trims dead mass

Runs execute on a GCE VM, artifacts stream to **Google Cloud Storage (GCS)**, and a **Streamlit** app visualizes 3D ρ, edges, gates, batteries, temps, and metrics over time.

---

## Quickstart (GCE VM)

```bash
# 1) Clone repo and enter
git clone https://github.com/youruser/Fuka-5.0.git
cd Fuka-5.0

# 2) Configure environment
cp .env.example scripts/env.sh
# edit scripts/env.sh with your PROJECT, BUCKET, etc.

# 3) One-time VM setup (venv, gcloud, bucket, deps)
bash scripts/setup_vm.sh

# 4) Run a simulation
bash scripts/run_sim.sh

# 5) Launch Streamlit UI (on VM or locally with port forward)
$F5_PY -m streamlit run app/streamlit_app.py --server.port $F5_STREAMLIT_PORT