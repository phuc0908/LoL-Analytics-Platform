# 🎮 LoL Esports Analytics - Killer Project

## Overview
AI-powered League of Legends Esports Analytics Platform với Win Prediction Model.

## Features
- **AI Win Prediction**: Dự đoán kết quả trận đấu dựa trên draft và team stats
- **Match Analytics**: Phân tích chi tiết các trận đấu
- **Player Stats**: Thống kê và đánh giá player
- **Team Rankings**: Xếp hạng các đội theo performance

## Tech Stack
- **ML**: Python, Scikit-learn, XGBoost, Pandas
- **Backend**: FastAPI (Python)
- **Frontend**: Next.js 15, TypeScript, Tailwind CSS
- **Database**: SQLite (dev) / PostgreSQL (prod)

## Project Structure
```
KillerProject/
├── ml/                    # Machine Learning
│   ├── data/             # Data files
│   ├── models/           # Trained models
│   ├── notebooks/        # Jupyter notebooks
│   └── src/              # ML source code
├── web/                   # Next.js Frontend
│   ├── app/              # App router
│   ├── components/       # React components
│   └── lib/              # Utilities
├── api/                   # FastAPI Backend
│   └── main.py           # API server
└── README.md
```

## Getting Started

### 1. Train ML Model
```bash
cd ml
pip install -r requirements.txt
python src/train.py
```

### 2. Start API Server
```bash
cd api
pip install -r requirements.txt
python main.py
```

### 3. Start Web App
```bash
cd web
npm install
npm run dev
```

## Dataset
- Source: Oracle's Elixir
- Games: 10,053 professional matches
- Leagues: 45 (LPL, LCK, LEC, etc.)
- Year: 2025

