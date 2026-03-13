#!/bin/bash
# ============================================================
#  SkinGuard AI — Deploy Script (H100 Server)
#  Runs both frontend + backend on a SINGLE port (8000)
# ============================================================

set -e

echo "=================================================="
echo "  🚀 SkinGuard AI — Deploying"
echo "=================================================="

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 1. Pull latest code
echo ""
echo "📥 Pulling latest code..."
git pull || true

# 2. Install backend deps
echo ""
echo "📦 Installing backend dependencies..."
pip install -q fastapi uvicorn[standard] python-multipart 2>/dev/null || pip install fastapi uvicorn python-multipart

# 3. Build frontend
echo ""
echo "🔨 Building frontend..."
if command -v node &> /dev/null; then
    cd "$PROJECT_DIR/frontend"
    npm install --silent
    npm run build
    cd "$PROJECT_DIR"
    echo "  ✅ Frontend built → frontend/dist/"
else
    echo "  ⚠️  Node.js not found. Install with:"
    echo "     curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y nodejs"
    echo "  Skipping frontend build..."
fi

# 4. Start server
echo ""
echo "=================================================="
echo "  🌐 Starting SkinGuard AI on port 8000"
echo "  📍 Frontend: http://0.0.0.0:8000"
echo "  📍 API Docs: http://0.0.0.0:8000/api/docs"
echo "=================================================="
echo ""

cd "$PROJECT_DIR"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level info
