# ── Base image 
FROM python:3.10-slim

# System dependencies for OpenCV + libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory 
WORKDIR /app

# ── Python dependencies (cached layer) 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code + assets 
COPY app.py .
COPY labels.json .
COPY models/ models/

# ── Streamlit config 
RUN mkdir -p /root/.streamlit
RUN echo '\
[server]\n\
headless = true\n\
port = 8501\n\
address = "0.0.0.0"\n\
enableCORS = false\n\
' > /root/.streamlit/config.toml

# ── Expose & run 
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
