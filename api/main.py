"""
FastAPI Backend for LoL Esports Analytics
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json

app = FastAPI(
    title="LoL Esports Analytics API",
    description="AI-powered League of Legends Esports Analytics",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).parent.parent
ML_DIR = BASE_DIR / "ml"
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"

# Global variables for loaded data
model = None
scaler = None
metadata = None
champion_stats = None
team_stats = None
player_stats = None
league_stats = None

# Advanced stats
league_details = None
region_stats = None
weighted_team_stats = None
weighted_player_stats = None

# Raw data for dynamic queries
raw_df = None
team_df = None
player_df = None


def load_resources():
    """Load ML model and statistics"""
    global model, scaler, metadata, champion_stats, team_stats, player_stats, league_stats
    global league_details, region_stats, weighted_team_stats, weighted_player_stats
    global raw_df, team_df, player_df
    
    try:
        # Load model
        if (MODELS_DIR / "win_predictor.joblib").exists():
            model = joblib.load(MODELS_DIR / "win_predictor.joblib")
            scaler = joblib.load(MODELS_DIR / "scaler.joblib")
            with open(MODELS_DIR / "metadata.json", 'r') as f:
                metadata = json.load(f)
            print("✅ Model loaded")
        else:
            print("⚠️ Model not found - run training first")
        
        # Load basic stats
        if (DATA_DIR / "champion_stats.csv").exists():
            champion_stats = pd.read_csv(DATA_DIR / "champion_stats.csv")
            team_stats = pd.read_csv(DATA_DIR / "team_stats.csv")
            player_stats = pd.read_csv(DATA_DIR / "player_stats.csv")
            league_stats = pd.read_csv(DATA_DIR / "league_stats.csv")
            print("✅ Basic statistics loaded")
        else:
            print("⚠️ Basic statistics not found - run training first")
        
        # Load advanced stats
        if (DATA_DIR / "league_details.csv").exists():
            league_details = pd.read_csv(DATA_DIR / "league_details.csv")
            region_stats = pd.read_csv(DATA_DIR / "region_stats.csv")
            print("✅ Advanced statistics loaded")
        else:
            print("⚠️ Advanced statistics not found - run advanced_stats.py")
        
        # Load raw data for dynamic queries
        raw_data_path = BASE_DIR / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
        if raw_data_path.exists():
            raw_df = pd.read_csv(raw_data_path, low_memory=False)
            team_df = raw_df[raw_df['position'] == 'team'].copy()
            player_df = raw_df[raw_df['position'] != 'team'].copy()
            print(f"✅ Raw data loaded ({len(raw_df):,} rows)")
            
    except Exception as e:
        print(f"❌ Error loading resources: {e}")


@app.on_event("startup")
async def startup_event():
    load_resources()


# ==================== SCHEMAS ====================

class PredictionRequest(BaseModel):
    blue_dragons: float = 0
    red_dragons: float = 0
    blue_heralds: float = 0
    red_heralds: float = 0
    blue_barons: float = 0
    red_barons: float = 0
    blue_towers: float = 0
    red_towers: float = 0
    blue_teamkills: float = 0
    red_teamkills: float = 0
    blue_teamdeaths: float = 0
    red_teamdeaths: float = 0
    golddiff_at10: float = 0
    xpdiff_at10: float = 0
    csdiff_at10: float = 0
    golddiff_at15: float = 0
    xpdiff_at15: float = 0
    csdiff_at15: float = 0
    blue_firstblood: float = 0
    blue_firstdragon: float = 0
    blue_firstherald: float = 0
    blue_firsttower: float = 0


class PredictionResponse(BaseModel):
    blue_win_probability: float
    red_win_probability: float
    predicted_winner: str
    confidence: float


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "🎮 LoL Esports Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "stats": "/api/stats",
            "champions": "/api/champions",
            "teams": "/api/teams",
            "players": "/api/players",
            "leagues": "/api/leagues",
            "predict": "/api/predict",
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    if champion_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    return {
        "total_champions": len(champion_stats),
        "total_teams": len(team_stats),
        "total_players": len(player_stats),
        "total_leagues": len(league_stats),
        "model_accuracy": metadata['metrics']['accuracy'] if metadata else None,
        "model_auc": metadata['metrics']['auc'] if metadata else None,
    }


@app.get("/api/champions")
async def get_champions(
    limit: int = 50,
    min_games: int = 10,
    sort_by: str = "games",
    league: Optional[str] = None,
    tier: Optional[str] = None
):
    """Get champion statistics, optionally filtered by league or tier"""
    global player_df, league_details
    
    # If league or tier filter is specified, calculate stats from raw data
    if league or tier:
        if player_df is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        df = player_df.copy()
        
        # Filter by specific league
        if league:
            df = df[df['league'].str.upper() == league.upper()]
        
        # Filter by tier
        if tier and league_details is not None:
            tier_leagues = league_details[league_details['tier'] == tier.upper()]['league'].tolist()
            df = df[df['league'].isin(tier_leagues)]
        
        if len(df) == 0:
            return {"count": 0, "champions": [], "filter": {"league": league, "tier": tier}}
        
        # Calculate champion stats from filtered data
        stats = df.groupby('champion').agg({
            'gameid': 'count',
            'result': ['sum', 'mean'],
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
        }).reset_index()
        
        stats.columns = ['champion', 'games', 'wins', 'winrate', 'avg_kills', 'avg_deaths', 'avg_assists']
        stats['kda'] = ((stats['avg_kills'] + stats['avg_assists']) / stats['avg_deaths'].replace(0, 1)).round(2)
        stats = stats[stats['games'] >= min_games]
        
        if len(stats) == 0:
            return {"count": 0, "champions": [], "filter": {"league": league, "tier": tier}}
        
        # Round values
        stats['winrate'] = (stats['winrate'] * 100).round(1)
        stats['avg_kills'] = stats['avg_kills'].round(1)
        stats['avg_deaths'] = stats['avg_deaths'].round(1)
        stats['avg_assists'] = stats['avg_assists'].round(1)
        
        if sort_by in stats.columns:
            stats = stats.sort_values(sort_by, ascending=False)
        
        stats = stats.head(limit)
        
        return {
            "count": len(stats),
            "champions": stats.to_dict(orient='records'),
            "filter": {"league": league, "tier": tier}
        }
    
    # Default: use pre-calculated stats
    if champion_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    df = champion_stats[champion_stats['games'] >= min_games].copy()
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    df = df.head(limit)
    
    # Round values
    df['winrate'] = (df['winrate'] * 100).round(1)
    df['kda'] = df['kda'].round(2)
    df['avg_kills'] = df['avg_kills'].round(1)
    df['avg_deaths'] = df['avg_deaths'].round(1)
    df['avg_assists'] = df['avg_assists'].round(1)
    
    return {
        "count": len(df),
        "champions": df.to_dict(orient='records'),
        "filter": {"league": None, "tier": None}
    }


@app.get("/api/champions/{champion_name}")
async def get_champion(champion_name: str):
    """Get specific champion statistics"""
    if champion_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    champ = champion_stats[champion_stats['champion'].str.lower() == champion_name.lower()]
    
    if len(champ) == 0:
        raise HTTPException(status_code=404, detail="Champion not found")
    
    champ = champ.iloc[0].to_dict()
    champ['winrate'] = round(champ['winrate'] * 100, 1)
    champ['kda'] = round(champ['kda'], 2)
    
    return champ


@app.get("/api/teams")
async def get_teams(
    limit: int = 50,
    min_games: int = 10,
    sort_by: str = "winrate",
    league: Optional[str] = None,
    tier: Optional[str] = None
):
    """Get team statistics, optionally filtered by league or tier"""
    global team_df, league_details
    
    # If league or tier filter is specified, calculate stats from raw data
    if league or tier:
        if team_df is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        df = team_df.copy()
        
        # Filter by specific league
        if league:
            df = df[df['league'].str.upper() == league.upper()]
        
        # Filter by tier
        if tier and league_details is not None:
            tier_leagues = league_details[league_details['tier'] == tier.upper()]['league'].tolist()
            df = df[df['league'].isin(tier_leagues)]
        
        if len(df) == 0:
            return {"count": 0, "teams": [], "filter": {"league": league, "tier": tier}}
        
        # Calculate team stats from filtered data
        stats = df.groupby('teamname').agg({
            'gameid': 'count',
            'result': ['sum', 'mean'],
            'teamkills': 'mean',
            'teamdeaths': 'mean',
            'dragons': 'mean',
            'barons': 'mean',
            'towers': 'mean',
            'gamelength': 'mean',
        }).reset_index()
        
        stats.columns = ['team', 'games', 'wins', 'winrate', 'avg_kills', 'avg_deaths', 
                        'avg_dragons', 'avg_barons', 'avg_towers', 'avg_gamelength']
        
        stats = stats[stats['games'] >= min_games]
        
        if len(stats) == 0:
            return {"count": 0, "teams": [], "filter": {"league": league, "tier": tier}}
        
        # Round values
        stats['winrate'] = (stats['winrate'] * 100).round(1)
        for col in ['avg_kills', 'avg_deaths', 'avg_dragons', 'avg_barons', 'avg_towers']:
            stats[col] = stats[col].round(1)
        stats['avg_gamelength'] = (stats['avg_gamelength'] / 60).round(1)
        
        if sort_by in stats.columns:
            ascending = sort_by == 'avg_deaths'
            stats = stats.sort_values(sort_by, ascending=ascending)
        
        stats = stats.head(limit)
        
        return {
            "count": len(stats),
            "teams": stats.to_dict(orient='records'),
            "filter": {"league": league, "tier": tier}
        }
    
    # Default: use pre-calculated stats
    if team_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    df = team_stats[team_stats['games'] >= min_games].copy()
    
    if sort_by in df.columns:
        ascending = sort_by == 'avg_deaths'
        df = df.sort_values(sort_by, ascending=ascending)
    
    df = df.head(limit)
    
    # Round values
    df['winrate'] = (df['winrate'] * 100).round(1)
    for col in ['avg_kills', 'avg_deaths', 'avg_dragons', 'avg_barons', 'avg_towers']:
        if col in df.columns:
            df[col] = df[col].round(1)
    df['avg_gamelength'] = (df['avg_gamelength'] / 60).round(1)  # Convert to minutes
    
    return {
        "count": len(df),
        "teams": df.to_dict(orient='records'),
        "filter": {"league": None, "tier": None}
    }


@app.get("/api/teams/{team_name}")
async def get_team(team_name: str):
    """Get specific team statistics"""
    if team_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    team = team_stats[team_stats['team'].str.lower() == team_name.lower()]
    
    if len(team) == 0:
        raise HTTPException(status_code=404, detail="Team not found")
    
    team = team.iloc[0].to_dict()
    team['winrate'] = round(team['winrate'] * 100, 1)
    team['avg_gamelength'] = round(team['avg_gamelength'] / 60, 1)
    
    return team


@app.get("/api/players")
async def get_players(
    limit: int = 50,
    min_games: int = 10,
    position: Optional[str] = None,
    sort_by: str = "kda",
    league: Optional[str] = None,
    tier: Optional[str] = None
):
    """Get player statistics, optionally filtered by league or tier"""
    global player_df, league_details
    
    # If league or tier filter is specified, calculate stats from raw data
    if league or tier:
        if player_df is None:
            raise HTTPException(status_code=503, detail="Data not loaded")
        
        df = player_df.copy()
        
        # Filter by specific league
        if league:
            df = df[df['league'].str.upper() == league.upper()]
        
        # Filter by tier
        if tier and league_details is not None:
            tier_leagues = league_details[league_details['tier'] == tier.upper()]['league'].tolist()
            df = df[df['league'].isin(tier_leagues)]
        
        if position:
            df = df[df['position'].str.lower() == position.lower()]
        
        if len(df) == 0:
            return {"count": 0, "players": [], "filter": {"league": league, "tier": tier, "position": position}}
        
        # Calculate player stats from filtered data
        stats = df.groupby(['playername', 'position', 'teamname']).agg({
            'gameid': 'count',
            'result': ['sum', 'mean'],
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
            'dpm': 'mean',
            'cspm': 'mean',
            'visionscore': 'mean',
        }).reset_index()
        
        stats.columns = ['player', 'position', 'team', 'games', 'wins', 'winrate', 
                        'avg_kills', 'avg_deaths', 'avg_assists', 'avg_dpm', 'avg_cspm', 'avg_vision']
        
        stats['kda'] = ((stats['avg_kills'] + stats['avg_assists']) / stats['avg_deaths'].replace(0, 1)).round(2)
        stats = stats[stats['games'] >= min_games]
        
        if len(stats) == 0:
            return {"count": 0, "players": [], "filter": {"league": league, "tier": tier, "position": position}}
        
        # Round values
        stats['winrate'] = (stats['winrate'] * 100).round(1)
        for col in ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_dpm', 'avg_cspm', 'avg_vision']:
            stats[col] = stats[col].round(1)
        
        if sort_by in stats.columns:
            stats = stats.sort_values(sort_by, ascending=False)
        
        stats = stats.head(limit)
        
        return {
            "count": len(stats),
            "players": stats.to_dict(orient='records'),
            "filter": {"league": league, "tier": tier, "position": position}
        }
    
    # Default: use pre-calculated stats
    if player_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    df = player_stats[player_stats['games'] >= min_games].copy()
    
    if position:
        df = df[df['position'].str.lower() == position.lower()]
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    df = df.head(limit)
    
    # Round values
    df['winrate'] = (df['winrate'] * 100).round(1)
    df['kda'] = df['kda'].round(2)
    for col in ['avg_kills', 'avg_deaths', 'avg_assists', 'avg_dpm', 'avg_cspm', 'avg_vision']:
        if col in df.columns:
            df[col] = df[col].round(1)
    
    return {
        "count": len(df),
        "players": df.to_dict(orient='records'),
        "filter": {"league": None, "tier": None, "position": position}
    }


@app.get("/api/leagues/list")
async def get_leagues_list():
    """Get list of all leagues for filter dropdowns"""
    global league_details, team_df
    
    if league_details is not None:
        leagues = league_details[['league', 'tier', 'region', 'games']].copy()
        leagues = leagues.sort_values(['tier', 'games'], ascending=[True, False])
        return {
            "leagues": leagues.to_dict(orient='records'),
            "tiers": ["S", "A", "B", "C", "D"]
        }
    
    # Fallback: get unique leagues from raw data
    if team_df is not None:
        leagues = team_df['league'].unique().tolist()
        return {
            "leagues": [{"league": l, "tier": "C", "region": "Unknown"} for l in sorted(leagues)],
            "tiers": ["S", "A", "B", "C", "D"]
        }
    
    return {"leagues": [], "tiers": []}


@app.get("/api/leagues")
async def get_leagues(sort_by: str = "games"):
    """Get league statistics"""
    if league_stats is None:
        raise HTTPException(status_code=503, detail="Statistics not loaded")
    
    df = league_stats.copy()
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    # Round values
    df['avg_gamelength'] = (df['avg_gamelength'] / 60).round(1)
    df['avg_kills'] = df['avg_kills'].round(1)
    
    return {
        "count": len(df),
        "leagues": df.to_dict(orient='records')
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_winner(request: PredictionRequest):
    """Predict match winner based on game state"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Create feature array
    features = pd.DataFrame([{
        'blue_dragons': request.blue_dragons,
        'red_dragons': request.red_dragons,
        'blue_heralds': request.blue_heralds,
        'red_heralds': request.red_heralds,
        'blue_barons': request.blue_barons,
        'red_barons': request.red_barons,
        'blue_towers': request.blue_towers,
        'red_towers': request.red_towers,
        'blue_teamkills': request.blue_teamkills,
        'red_teamkills': request.red_teamkills,
        'blue_teamdeaths': request.blue_teamdeaths,
        'red_teamdeaths': request.red_teamdeaths,
        'golddiff_at10': request.golddiff_at10,
        'xpdiff_at10': request.xpdiff_at10,
        'csdiff_at10': request.csdiff_at10,
        'golddiff_at15': request.golddiff_at15,
        'xpdiff_at15': request.xpdiff_at15,
        'csdiff_at15': request.csdiff_at15,
        'blue_firstblood': request.blue_firstblood,
        'blue_firstdragon': request.blue_firstdragon,
        'blue_firstherald': request.blue_firstherald,
        'blue_firsttower': request.blue_firsttower,
    }])
    
    # Ensure column order matches training
    features = features[metadata['feature_names']]
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0]
    
    return PredictionResponse(
        blue_win_probability=round(float(prob[1]) * 100, 1),
        red_win_probability=round(float(prob[0]) * 100, 1),
        predicted_winner='Blue' if prob[1] > 0.5 else 'Red',
        confidence=round(float(max(prob)) * 100, 1)
    )


@app.get("/api/model/info")
async def get_model_info():
    """Get model information"""
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "features": len(metadata['feature_names']),
        "feature_names": metadata['feature_names'],
        "metrics": {
            "accuracy": round(metadata['metrics']['accuracy'] * 100, 2),
            "auc": round(metadata['metrics']['auc'], 4),
            "cv_accuracy": round(metadata['metrics']['cv_mean'] * 100, 2),
        },
        "training_samples": metadata['metrics']['train_size'],
        "trained_at": metadata['metrics']['trained_at'],
    }


# ==================== ADVANCED STATS ENDPOINTS ====================

@app.get("/api/leagues/details")
async def get_league_details(sort_by: str = "games"):
    """Get detailed league statistics with tiers and weights"""
    if league_details is None:
        raise HTTPException(status_code=503, detail="Advanced statistics not loaded")
    
    df = league_details.copy()
    
    if sort_by in df.columns:
        ascending = sort_by in ['avg_gamelength']
        df = df.sort_values(sort_by, ascending=ascending)
    
    return {
        "count": len(df),
        "leagues": df.to_dict(orient='records')
    }


@app.get("/api/leagues/{league_name}/details")
async def get_single_league_details(league_name: str):
    """Get details for a specific league"""
    if league_details is None:
        raise HTTPException(status_code=503, detail="Advanced statistics not loaded")
    
    league = league_details[league_details['league'].str.upper() == league_name.upper()]
    
    if len(league) == 0:
        raise HTTPException(status_code=404, detail="League not found")
    
    return league.iloc[0].to_dict()


@app.get("/api/regions")
async def get_regions():
    """Get statistics by region"""
    if region_stats is None:
        raise HTTPException(status_code=503, detail="Advanced statistics not loaded")
    
    return {
        "count": len(region_stats),
        "regions": region_stats.to_dict(orient='records')
    }


@app.get("/api/teams/weighted")
async def get_weighted_teams(
    limit: int = 50,
    region: Optional[str] = None,
    sort_by: str = "weighted_winrate"
):
    """Get team statistics with weighted winrate by league importance"""
    if weighted_team_stats is None:
        raise HTTPException(status_code=503, detail="Advanced statistics not loaded")
    
    df = weighted_team_stats.copy()
    
    if region:
        df = df[df['region'].str.lower() == region.lower()]
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    df = df.head(limit)
    
    return {
        "count": len(df),
        "teams": df.to_dict(orient='records')
    }


@app.get("/api/players/weighted")
async def get_weighted_players(
    limit: int = 50,
    position: Optional[str] = None,
    region: Optional[str] = None,
    sort_by: str = "weighted_kda"
):
    """Get player statistics with weighted stats by league importance"""
    if weighted_player_stats is None:
        raise HTTPException(status_code=503, detail="Advanced statistics not loaded")
    
    df = weighted_player_stats.copy()
    
    if position:
        df = df[df['position'].str.lower() == position.lower()]
    
    if region:
        df = df[df['region'].str.lower() == region.lower()]
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    
    df = df.head(limit)
    
    return {
        "count": len(df),
        "players": df.to_dict(orient='records')
    }


@app.get("/api/tiers")
async def get_league_tiers():
    """Get league tier definitions"""
    return {
        "tiers": {
            "S": {"name": "International", "weight": 3.0, "leagues": ["WLDs", "MSI", "Worlds"]},
            "A": {"name": "Major Regions", "weight": 2.0, "leagues": ["LCK", "LPL", "LEC", "LCS"]},
            "B": {"name": "Minor Regions", "weight": 1.5, "leagues": ["PCS", "VCS", "LJL", "CBLOL"]},
            "C": {"name": "Regional Leagues", "weight": 1.0, "leagues": ["LFL", "PRM", "NLC", "EM"]},
            "D": {"name": "Academy/Challenger", "weight": 0.8, "leagues": ["LCKC", "NACL", "LDL"]},
        }
    }


# ==================== LEAGUE DETAIL ENDPOINTS ====================

@app.get("/api/leagues/{league_name}/matches")
async def get_league_matches(league_name: str, limit: int = 50):
    """Get recent matches for a specific league, grouped by series"""
    global team_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = team_df[team_df['league'].str.upper() == league_name.upper()].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="League not found")
    
    # Extract date only for grouping
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    
    # Get unique games first
    games = df.groupby('gameid').agg({
        'date': 'first',
        'date_only': 'first',
        'teamname': lambda x: list(x),
        'result': lambda x: list(x),
        'teamkills': lambda x: list(x),
        'gamelength': 'first',
        'playoffs': 'first',
        'game': 'first',
    }).reset_index()
    
    # Process each game
    game_records = []
    for _, row in games.iterrows():
        teams = row['teamname']
        results = list(row['result'])
        
        if len(teams) == 2:
            # Sort teams alphabetically for consistent grouping
            sorted_teams = tuple(sorted(teams))
            winner_idx = results.index(1) if 1 in results else 0
            winner = teams[winner_idx]
            
            game_records.append({
                'date_only': row['date_only'],
                'date': row['date'],
                'team_pair': sorted_teams,
                'team1': teams[0],
                'team2': teams[1],
                'winner': winner,
                'gamelength': row['gamelength'],
                'playoffs': row['playoffs'],
                'game_number': row['game'],
            })
    
    if not game_records:
        return {"count": 0, "matches": []}
    
    games_df = pd.DataFrame(game_records)
    
    # Group by date and team pair to form series
    series_list = []
    for (date_only, team_pair), group in games_df.groupby(['date_only', 'team_pair']):
        team1, team2 = team_pair
        
        # Count wins for each team
        team1_wins = sum(1 for _, r in group.iterrows() if r['winner'] == team1)
        team2_wins = sum(1 for _, r in group.iterrows() if r['winner'] == team2)
        
        # Determine series winner
        series_winner = team1 if team1_wins > team2_wins else team2
        
        # Calculate total game length
        total_gamelength = group['gamelength'].sum()
        avg_gamelength = group['gamelength'].mean()
        
        # Determine series type
        total_games = len(group)
        if total_games == 1:
            series_type = 'BO1'
        elif total_games <= 3:
            series_type = 'BO3'
        else:
            series_type = 'BO5'
        
        series_list.append({
            'date': str(group['date'].iloc[0]),
            'team1': team1,
            'team2': team2,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'series_score': f"{team1_wins}-{team2_wins}",
            'winner': series_winner,
            'total_games': total_games,
            'series_type': series_type,
            'avg_gamelength': round(float(avg_gamelength) / 60, 1),
            'playoffs': int(group['playoffs'].iloc[0]),
        })
    
    # Sort by date descending
    series_list = sorted(series_list, key=lambda x: x['date'], reverse=True)[:limit]
    
    return {"count": len(series_list), "matches": series_list}


@app.get("/api/leagues/{league_name}/teams")
async def get_league_teams(league_name: str):
    """Get team statistics for a specific league"""
    global team_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = team_df[team_df['league'].str.upper() == league_name.upper()].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="League not found")
    
    team_stats = df.groupby('teamname').agg({
        'gameid': 'count',
        'result': ['sum', 'mean'],
        'teamkills': 'mean',
        'teamdeaths': 'mean',
        'dragons': 'mean',
        'barons': 'mean',
        'towers': 'mean',
        'gamelength': 'mean',
    }).reset_index()
    
    team_stats.columns = [
        'team', 'games', 'wins', 'winrate', 'avg_kills', 'avg_deaths',
        'avg_dragons', 'avg_barons', 'avg_towers', 'avg_gamelength'
    ]
    
    team_stats['winrate'] = (team_stats['winrate'] * 100).round(1)
    team_stats['avg_gamelength'] = (team_stats['avg_gamelength'] / 60).round(1)
    team_stats['avg_kills'] = team_stats['avg_kills'].round(1)
    team_stats['avg_deaths'] = team_stats['avg_deaths'].round(1)
    team_stats['avg_dragons'] = team_stats['avg_dragons'].round(2)
    team_stats['avg_barons'] = team_stats['avg_barons'].round(2)
    
    team_stats = team_stats.sort_values('winrate', ascending=False)
    
    return {"count": len(team_stats), "teams": team_stats.to_dict(orient='records')}


@app.get("/api/leagues/{league_name}/players")
async def get_league_players(league_name: str, position: Optional[str] = None):
    """Get player statistics for a specific league"""
    global player_df
    if player_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = player_df[player_df['league'].str.upper() == league_name.upper()].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="League not found")
    
    if position:
        df = df[df['position'].str.lower() == position.lower()]
    
    player_stats = df.groupby(['playername', 'teamname', 'position']).agg({
        'gameid': 'count',
        'result': 'mean',
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
        'dpm': 'mean',
        'cspm': 'mean',
        'visionscore': 'mean',
    }).reset_index()
    
    player_stats['kda'] = (player_stats['kills'] + player_stats['assists']) / player_stats['deaths'].replace(0, 1)
    
    player_stats.columns = [
        'player', 'team', 'position', 'games', 'winrate',
        'avg_kills', 'avg_deaths', 'avg_assists',
        'avg_dpm', 'avg_cspm', 'avg_vision', 'kda'
    ]
    
    player_stats['winrate'] = (player_stats['winrate'] * 100).round(1)
    player_stats['kda'] = player_stats['kda'].round(2)
    player_stats['avg_kills'] = player_stats['avg_kills'].round(1)
    player_stats['avg_deaths'] = player_stats['avg_deaths'].round(1)
    player_stats['avg_assists'] = player_stats['avg_assists'].round(1)
    player_stats['avg_dpm'] = player_stats['avg_dpm'].round(0)
    
    player_stats = player_stats.sort_values('kda', ascending=False)
    
    return {"count": len(player_stats), "players": player_stats.to_dict(orient='records')}


@app.get("/api/leagues/{league_name}/champions")
async def get_league_champions(league_name: str):
    """Get champion statistics for a specific league"""
    global player_df
    if player_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = player_df[player_df['league'].str.upper() == league_name.upper()].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="League not found")
    
    total_games = df['gameid'].nunique()
    
    champ_stats = df.groupby('champion').agg({
        'gameid': 'count',
        'result': 'mean',
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
    }).reset_index()
    
    champ_stats['kda'] = (champ_stats['kills'] + champ_stats['assists']) / champ_stats['deaths'].replace(0, 1)
    champ_stats['pickrate'] = (champ_stats['gameid'] / total_games * 100).round(1)
    
    champ_stats.columns = [
        'champion', 'games', 'winrate', 'avg_kills', 'avg_deaths', 'avg_assists', 'kda', 'pickrate'
    ]
    
    champ_stats['winrate'] = (champ_stats['winrate'] * 100).round(1)
    champ_stats['kda'] = champ_stats['kda'].round(2)
    
    champ_stats = champ_stats.sort_values('games', ascending=False)
    
    return {"count": len(champ_stats), "champions": champ_stats.to_dict(orient='records')}


@app.get("/api/regions/{region_name}/champions")
async def get_region_champions(region_name: str):
    """Get champion statistics for a specific region"""
    global player_df
    if player_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Map region names using league_details
    region_league_map = {}
    if league_details is not None:
        for _, row in league_details.iterrows():
            region_league_map[row['league']] = row['region']
    
    df = player_df.copy()
    df['region'] = df['league'].map(region_league_map).fillna('Other')
    
    df = player_df[player_df['region'].str.lower() == region_name.lower()].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="Region not found")
    
    total_games = df['gameid'].nunique()
    
    champ_stats = df.groupby('champion').agg({
        'gameid': 'count',
        'result': 'mean',
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
    }).reset_index()
    
    champ_stats['kda'] = (champ_stats['kills'] + champ_stats['assists']) / champ_stats['deaths'].replace(0, 1)
    champ_stats['pickrate'] = (champ_stats['gameid'] / total_games * 100).round(1)
    
    champ_stats.columns = [
        'champion', 'games', 'winrate', 'avg_kills', 'avg_deaths', 'avg_assists', 'kda', 'pickrate'
    ]
    
    champ_stats['winrate'] = (champ_stats['winrate'] * 100).round(1)
    champ_stats['kda'] = champ_stats['kda'].round(2)
    
    champ_stats = champ_stats.sort_values('games', ascending=False)
    
    return {"count": len(champ_stats), "champions": champ_stats.to_dict(orient='records')}


@app.get("/api/leagues/timeline")
async def get_leagues_timeline():
    """Get timeline of all leagues (start and end dates)"""
    global team_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = team_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    timeline = df.groupby('league').agg({
        'date': ['min', 'max'],
        'gameid': 'nunique',
    }).reset_index()
    
    timeline.columns = ['league', 'start_date', 'end_date', 'games']
    
    # Add duration in days
    timeline['duration_days'] = (timeline['end_date'] - timeline['start_date']).dt.days
    
    # Format dates
    timeline['start_date'] = timeline['start_date'].dt.strftime('%Y-%m-%d')
    timeline['end_date'] = timeline['end_date'].dt.strftime('%Y-%m-%d')
    
    # Add tier info from league_details
    if league_details is not None:
        tier_map = dict(zip(league_details['league'], league_details['tier']))
        region_map = dict(zip(league_details['league'], league_details['region']))
        timeline['tier'] = timeline['league'].map(tier_map).fillna('C')
        timeline['region'] = timeline['league'].map(region_map).fillna('Other')
    else:
        timeline['tier'] = 'C'
        timeline['region'] = 'Other'
    
    # Sort by start date
    timeline = timeline.sort_values('start_date')
    
    return {
        "count": len(timeline),
        "timeline": timeline.to_dict(orient='records')
    }


@app.get("/api/leagues/timeline/detailed")
async def get_leagues_timeline_detailed():
    """Get detailed timeline with activity periods and breaks for each league"""
    global team_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = team_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    df['year_week'] = df['date'].dt.strftime('%Y-W%W')
    
    result = []
    
    for league in df['league'].unique():
        league_df = df[df['league'] == league].copy()
        
        # Get games per week
        weekly = league_df.groupby('year_week').agg({
            'date': ['min', 'max'],
            'gameid': 'nunique',
        }).reset_index()
        weekly.columns = ['year_week', 'week_start', 'week_end', 'games']
        weekly = weekly.sort_values('week_start')
        
        # Find activity periods (consecutive weeks with games)
        periods = []
        current_period = None
        
        for i, row in weekly.iterrows():
            if current_period is None:
                current_period = {
                    'start': row['week_start'],
                    'end': row['week_end'],
                    'games': row['games'],
                    'weeks': 1
                }
            else:
                # Check if this week is within 14 days of previous
                days_gap = (row['week_start'] - current_period['end']).days
                
                if days_gap <= 14:  # Continue same period
                    current_period['end'] = row['week_end']
                    current_period['games'] += row['games']
                    current_period['weeks'] += 1
                else:  # New period
                    periods.append(current_period)
                    current_period = {
                        'start': row['week_start'],
                        'end': row['week_end'],
                        'games': row['games'],
                        'weeks': 1
                    }
        
        if current_period:
            periods.append(current_period)
        
        # Get tier and region
        tier = 'C'
        region = 'Other'
        if league_details is not None:
            league_info = league_details[league_details['league'] == league]
            if len(league_info) > 0:
                tier = league_info.iloc[0]['tier']
                region = league_info.iloc[0]['region']
        
        # Format periods
        formatted_periods = []
        for p in periods:
            formatted_periods.append({
                'start_date': p['start'].strftime('%Y-%m-%d'),
                'end_date': p['end'].strftime('%Y-%m-%d'),
                'games': int(p['games']),
                'weeks': int(p['weeks']),
                'duration_days': (p['end'] - p['start']).days + 1,
            })
        
        # Calculate breaks between periods
        breaks = []
        for i in range(len(periods) - 1):
            break_start = periods[i]['end']
            break_end = periods[i + 1]['start']
            break_days = (break_end - break_start).days
            if break_days > 7:  # Only show significant breaks
                breaks.append({
                    'start_date': break_start.strftime('%Y-%m-%d'),
                    'end_date': break_end.strftime('%Y-%m-%d'),
                    'days': break_days,
                })
        
        result.append({
            'league': league,
            'tier': tier,
            'region': region,
            'total_games': int(league_df['gameid'].nunique()),
            'periods': formatted_periods,
            'breaks': breaks,
            'overall_start': league_df['date'].min().strftime('%Y-%m-%d'),
            'overall_end': league_df['date'].max().strftime('%Y-%m-%d'),
        })
    
    # Sort by tier then by start date
    tier_order = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}
    result = sorted(result, key=lambda x: (tier_order.get(x['tier'], 5), x['overall_start']))
    
    return {
        "count": len(result),
        "timeline": result
    }


@app.get("/api/activity/weekly")
async def get_weekly_activity():
    """Get weekly game activity across all leagues"""
    global team_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = team_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
    df['week_start'] = df['week_start'].dt.strftime('%Y-%m-%d')
    
    # Games per week
    weekly = df.groupby('week_start').agg({
        'gameid': 'nunique',
        'league': lambda x: list(x.unique()),
    }).reset_index()
    
    weekly.columns = ['week', 'games', 'leagues']
    weekly['league_count'] = weekly['leagues'].apply(len)
    weekly = weekly.sort_values('week')
    
    return {
        "count": len(weekly),
        "weeks": weekly.to_dict(orient='records')
    }


# ==================== CHATBOT ENDPOINT ====================

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    data: Optional[dict] = None


def search_champion(query: str):
    """Search for champion info"""
    if champion_stats is None:
        return None
    query_lower = query.lower()
    matches = champion_stats[champion_stats['champion'].str.lower().str.contains(query_lower, na=False)]
    if len(matches) > 0:
        champ = matches.iloc[0]
        return {
            'champion': champ['champion'],
            'games': int(champ['games']),
            'winrate': round(champ['winrate'] * 100, 1),
            'kda': round(champ['kda'], 2),
        }
    return None


def search_team(query: str):
    """Search for team info"""
    if team_stats is None:
        return None
    query_lower = query.lower()
    matches = team_stats[team_stats['team'].str.lower().str.contains(query_lower, na=False)]
    if len(matches) > 0:
        team = matches.iloc[0]
        return {
            'team': team['team'],
            'games': int(team['games']),
            'winrate': round(team['winrate'] * 100, 1),
        }
    return None


def search_player(query: str):
    """Search for player info"""
    if player_stats is None:
        return None
    query_lower = query.lower()
    matches = player_stats[player_stats['player'].str.lower().str.contains(query_lower, na=False)]
    if len(matches) > 0:
        player = matches.iloc[0]
        return {
            'player': player['player'],
            'team': player['team'],
            'position': player['position'],
            'games': int(player['games']),
            'winrate': round(player['winrate'] * 100, 1),
            'kda': round(player['kda'], 2),
        }
    return None


def get_top_champions(n: int = 5):
    """Get top champions by games"""
    if champion_stats is None:
        return []
    top = champion_stats.nlargest(n, 'games')
    return [{'champion': r['champion'], 'games': int(r['games']), 'winrate': round(r['winrate'] * 100, 1)} 
            for _, r in top.iterrows()]


def get_top_teams(n: int = 5):
    """Get top teams by winrate (min 20 games)"""
    if team_stats is None:
        return []
    filtered = team_stats[team_stats['games'] >= 20]
    top = filtered.nlargest(n, 'winrate')
    return [{'team': r['team'], 'games': int(r['games']), 'winrate': round(r['winrate'] * 100, 1)} 
            for _, r in top.iterrows()]


def get_top_players(n: int = 5, position: str = None):
    """Get top players by KDA"""
    if player_stats is None:
        return []
    filtered = player_stats[player_stats['games'] >= 10]
    if position:
        filtered = filtered[filtered['position'].str.lower() == position.lower()]
    top = filtered.nlargest(n, 'kda')
    return [{'player': r['player'], 'team': r['team'], 'position': r['position'], 
             'kda': round(r['kda'], 2), 'winrate': round(r['winrate'] * 100, 1)} 
            for _, r in top.iterrows()]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """AI Chatbot for LoL Esports queries"""
    message = request.message.lower().strip()
    
    # Greetings
    if any(word in message for word in ['hi', 'hello', 'xin chào', 'chào', 'hey']):
        return ChatResponse(
            response="Xin chào! 👋 Tôi là LoL Analytics Bot. Bạn có thể hỏi tôi về:\n"
                    "• Thông tin tướng (VD: 'Zed có winrate bao nhiêu?')\n"
                    "• Thông tin đội (VD: 'T1 thắng bao nhiêu trận?')\n"
                    "• Thông tin tuyển thủ (VD: 'Faker có KDA bao nhiêu?')\n"
                    "• Top tướng/đội/tuyển thủ\n"
                    "• Thông tin giải đấu"
        )
    
    # Help
    if any(word in message for word in ['help', 'giúp', 'hướng dẫn', '?']):
        return ChatResponse(
            response="📖 Hướng dẫn sử dụng:\n\n"
                    "🎮 **Tướng**: 'Yasuo stats', 'winrate Zed', 'top tướng'\n"
                    "🏆 **Đội**: 'T1 stats', 'Gen.G winrate', 'top đội'\n"
                    "👤 **Tuyển thủ**: 'Faker stats', 'top mid', 'top adc'\n"
                    "📊 **Tổng quan**: 'tổng số trận', 'có bao nhiêu giải'\n\n"
                    "Hãy thử hỏi tôi điều gì đó!"
        )
    
    # Champion queries
    if any(word in message for word in ['tướng', 'champion', 'champ']):
        if 'top' in message or 'best' in message or 'tốt nhất' in message:
            top_champs = get_top_champions(5)
            if top_champs:
                champ_list = "\n".join([f"• {c['champion']}: {c['winrate']}% WR ({c['games']} games)" 
                                       for c in top_champs])
                return ChatResponse(
                    response=f"🏆 Top 5 tướng được chơi nhiều nhất:\n{champ_list}",
                    data={"champions": top_champs}
                )
    
    # Search specific champion
    champ_keywords = ['yasuo', 'zed', 'ahri', 'jinx', 'thresh', 'lee sin', 'lux', 'ezreal', 
                      'vayne', 'kai\'sa', 'akali', 'sylas', 'yone', 'viego', 'gwen', 'aurora',
                      'rumble', 'xin zhao', 'ambessa', 'wukong', 'alistar', 'rell', 'taliyah',
                      'azir', 'rakan', 'orianna', 'corki', 'varus', 'ashe', 'jhin', 'aphelios']
    
    for champ in champ_keywords:
        if champ in message:
            result = search_champion(champ)
            if result:
                return ChatResponse(
                    response=f"🎮 **{result['champion']}**\n"
                            f"• Games: {result['games']}\n"
                            f"• Win Rate: {result['winrate']}%\n"
                            f"• KDA: {result['kda']}",
                    data={"champion": result}
                )
    
    # Team queries
    if any(word in message for word in ['đội', 'team', 'đội tuyển']):
        if 'top' in message or 'best' in message or 'tốt nhất' in message:
            top_teams = get_top_teams(5)
            if top_teams:
                team_list = "\n".join([f"• {t['team']}: {t['winrate']}% WR ({t['games']} games)" 
                                      for t in top_teams])
                return ChatResponse(
                    response=f"🏆 Top 5 đội có tỷ lệ thắng cao nhất:\n{team_list}",
                    data={"teams": top_teams}
                )
    
    # Search specific team
    team_keywords = ['t1', 'gen.g', 'geng', 'hle', 'hanwha', 'drx', 'kt', 'dplus', 'dk', 'damwon',
                     'jdg', 'blg', 'weibo', 'lng', 'edg', 'tes', 'fpx', 'rng', 'ig',
                     'g2', 'fnatic', 'fnc', 'mad', 'vitality', 'kc', 'karmine',
                     'c9', 'cloud9', '100t', 'tl', 'team liquid', 'eg', 'flyquest']
    
    for team_name in team_keywords:
        if team_name in message:
            result = search_team(team_name)
            if result:
                return ChatResponse(
                    response=f"🏆 **{result['team']}**\n"
                            f"• Games: {result['games']}\n"
                            f"• Win Rate: {result['winrate']}%",
                    data={"team": result}
                )
    
    # Player queries
    if any(word in message for word in ['tuyển thủ', 'player', 'người chơi', 'pro']):
        position = None
        if any(pos in message for pos in ['top', 'tốp']):
            if 'mid' in message or 'giữa' in message:
                position = 'mid'
            elif 'adc' in message or 'bot' in message or 'xạ thủ' in message:
                position = 'bot'
            elif 'sup' in message or 'hỗ trợ' in message:
                position = 'sup'
            elif 'jungle' in message or 'jng' in message or 'rừng' in message:
                position = 'jng'
            else:
                position = 'top'
        
        top_players = get_top_players(5, position)
        if top_players:
            pos_text = f" vị trí {position.upper()}" if position else ""
            player_list = "\n".join([f"• {p['player']} ({p['team']}): KDA {p['kda']}, {p['winrate']}% WR" 
                                    for p in top_players])
            return ChatResponse(
                response=f"👤 Top 5 tuyển thủ{pos_text}:\n{player_list}",
                data={"players": top_players}
            )
    
    # Search specific player
    player_keywords = ['faker', 'chovy', 'showmaker', 'ruler', 'keria', 'canyon', 'zeus', 'gumayusi',
                       'oner', 'deft', 'viper', 'peyz', 'lehends', 'peanut', 'caps', 'jankos',
                       'rekkles', 'hans sama', 'upset', 'humanoid', 'elyoya', 'odoamne']
    
    for player_name in player_keywords:
        if player_name in message:
            result = search_player(player_name)
            if result:
                return ChatResponse(
                    response=f"👤 **{result['player']}** ({result['team']})\n"
                            f"• Position: {result['position'].upper()}\n"
                            f"• Games: {result['games']}\n"
                            f"• Win Rate: {result['winrate']}%\n"
                            f"• KDA: {result['kda']}",
                    data={"player": result}
                )
    
    # Stats queries
    if any(word in message for word in ['tổng', 'total', 'bao nhiêu', 'số lượng']):
        total_games = len(team_df) // 2 if team_df is not None else 0
        total_players = len(player_stats) if player_stats is not None else 0
        total_teams = len(team_stats) if team_stats is not None else 0
        total_champs = len(champion_stats) if champion_stats is not None else 0
        total_leagues = len(league_stats) if league_stats is not None else 0
        
        return ChatResponse(
            response=f"📊 **Thống kê tổng quan**\n"
                    f"• Tổng số trận: {total_games:,}\n"
                    f"• Số tuyển thủ: {total_players:,}\n"
                    f"• Số đội: {total_teams}\n"
                    f"• Số tướng: {total_champs}\n"
                    f"• Số giải đấu: {total_leagues}",
            data={"stats": {
                "games": total_games,
                "players": total_players,
                "teams": total_teams,
                "champions": total_champs,
                "leagues": total_leagues
            }}
        )
    
    # Model/AI queries
    if any(word in message for word in ['model', 'ai', 'predict', 'dự đoán', 'accuracy']):
        if metadata:
            return ChatResponse(
                response=f"🤖 **AI Model Info**\n"
                        f"• Model: XGBoost Classifier\n"
                        f"• Accuracy: {metadata['metrics']['accuracy']*100:.1f}%\n"
                        f"• AUC: {metadata['metrics']['auc']:.4f}\n"
                        f"• Training samples: {metadata['metrics']['train_size']:,}\n\n"
                        f"Bạn có thể thử dự đoán tại trang AI Predict!"
            )
    
    # Default response
    return ChatResponse(
        response="🤔 Tôi không hiểu câu hỏi của bạn. Hãy thử hỏi về:\n"
                "• Thông tin tướng: 'Yasuo stats'\n"
                "• Thông tin đội: 'T1 winrate'\n"
                "• Top tuyển thủ: 'top mid'\n"
                "• Hoặc gõ 'help' để xem hướng dẫn"
    )


# ==================== DETAIL MATCH ENDPOINTS ====================

@app.get("/api/series/games")
async def get_series_games(date: str, team1: str, team2: str):
    """Get individual games in a series"""
    global team_df, player_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Parse date
    target_date = pd.to_datetime(date).date()
    
    # Filter by date and teams
    df = team_df.copy()
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    
    # Find games with these two teams on this date
    df = df[df['date_only'] == target_date]
    df = df[df['teamname'].str.upper().isin([team1.upper(), team2.upper()])]
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="Series not found")
    
    # Get unique games
    games = df.groupby('gameid').agg({
        'date': 'first',
        'teamname': lambda x: list(x),
        'result': lambda x: list(x),
        'teamkills': lambda x: list(x),
        'teamdeaths': lambda x: list(x),
        'dragons': lambda x: list(x),
        'barons': lambda x: list(x),
        'towers': lambda x: list(x),
        'gamelength': 'first',
        'playoffs': 'first',
        'game': 'first',
        'league': 'first',
    }).reset_index()
    
    game_records = []
    for _, row in games.iterrows():
        teams = row['teamname']
        if len(teams) != 2:
            continue
            
        results = list(row['result'])
        kills = list(row['teamkills'])
        deaths = list(row['teamdeaths'])
        dragons = list(row['dragons'])
        barons = list(row['barons'])
        towers = list(row['towers'])
        
        winner_idx = results.index(1) if 1 in results else 0
        
        game_records.append({
            'gameid': row['gameid'],
            'game_number': int(row['game']) if row['game'] else 1,
            'date': str(row['date'])[:10],
            'league': row['league'],
            'team1': teams[0],
            'team2': teams[1],
            'team1_kills': int(kills[0]),
            'team2_kills': int(kills[1]),
            'team1_deaths': int(deaths[0]),
            'team2_deaths': int(deaths[1]),
            'team1_dragons': int(dragons[0]),
            'team2_dragons': int(dragons[1]),
            'team1_barons': int(barons[0]),
            'team2_barons': int(barons[1]),
            'team1_towers': int(towers[0]),
            'team2_towers': int(towers[1]),
            'winner': teams[winner_idx],
            'gamelength': round(row['gamelength'] / 60, 1),
            'playoffs': int(row['playoffs']),
        })
    
    # Sort by game number
    game_records.sort(key=lambda x: x['game_number'])
    
    return {
        "series": {
            "date": date,
            "team1": team1,
            "team2": team2,
            "total_games": len(game_records),
        },
        "games": game_records
    }


@app.get("/api/leagues/{league_name}/teams/{team_name}/matches")
async def get_team_matches_in_league(league_name: str, team_name: str, limit: int = 50):
    """Get matches for a specific team in a league"""
    global team_df
    if team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = team_df[
        (team_df['league'].str.upper() == league_name.upper()) & 
        (team_df['teamname'].str.upper() == team_name.upper())
    ].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="Team not found in league")
    
    # Get unique games
    games = df.groupby('gameid').agg({
        'date': 'first',
        'gamelength': 'first',
        'result': 'first',
        'teamkills': 'first',
        'teamdeaths': 'first',
        'dragons': 'first',
        'barons': 'first',
        'towers': 'first',
        'playoffs': 'first',
        'game': 'first',
    }).reset_index()
    
    # Get opponent info
    match_records = []
    for _, row in games.iterrows():
        game_id = row['gameid']
        game_data = team_df[team_df['gameid'] == game_id]
        
        opponent = game_data[game_data['teamname'].str.upper() != team_name.upper()]['teamname'].iloc[0] if len(game_data) > 1 else 'Unknown'
        opponent_kills = game_data[game_data['teamname'].str.upper() != team_name.upper()]['teamkills'].iloc[0] if len(game_data) > 1 else 0
        
        match_records.append({
            'gameid': game_id,
            'date': str(row['date'])[:10],
            'opponent': opponent,
            'result': 'Win' if row['result'] == 1 else 'Loss',
            'score': f"{int(row['teamkills'])}-{int(opponent_kills)}",
            'kills': int(row['teamkills']),
            'deaths': int(row['teamdeaths']),
            'dragons': int(row['dragons']),
            'barons': int(row['barons']),
            'towers': int(row['towers']),
            'gamelength': round(row['gamelength'] / 60, 1),
            'playoffs': int(row['playoffs']),
        })
    
    match_records.sort(key=lambda x: x['date'], reverse=True)
    
    return {"count": len(match_records), "matches": match_records[:limit]}


@app.get("/api/leagues/{league_name}/players/{player_name}/matches")
async def get_player_matches_in_league(league_name: str, player_name: str, limit: int = 50):
    """Get matches for a specific player in a league"""
    global player_df
    if player_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = player_df[
        (player_df['league'].str.upper() == league_name.upper()) & 
        (player_df['playername'].str.upper() == player_name.upper())
    ].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="Player not found in league")
    
    match_records = []
    for _, row in df.iterrows():
        match_records.append({
            'gameid': row['gameid'],
            'date': str(row['date'])[:10],
            'team': row['teamname'],
            'champion': row['champion'],
            'result': 'Win' if row['result'] == 1 else 'Loss',
            'kills': int(row['kills']),
            'deaths': int(row['deaths']),
            'assists': int(row['assists']),
            'kda': round((row['kills'] + row['assists']) / max(row['deaths'], 1), 2),
            'cs': int(row.get('total_cs', 0)),
            'dpm': round(row['dpm'], 0),
            'gamelength': round(row['gamelength'] / 60, 1),
            'playoffs': int(row.get('playoffs', 0)),
        })
    
    match_records.sort(key=lambda x: x['date'], reverse=True)
    
    return {"count": len(match_records), "matches": match_records[:limit]}


@app.get("/api/matches/{game_id}")
async def get_match_details(game_id: str):
    """Get detailed information for a specific match with full player stats"""
    global player_df, team_df
    if player_df is None or team_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get team data
    team_data = team_df[team_df['gameid'] == game_id]
    if len(team_data) == 0:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Get player data
    player_data = player_df[player_df['gameid'] == game_id]
    
    # Helper to safely get value
    def safe_int(val, default=0):
        try:
            if pd.isna(val):
                return default
            return int(val)
        except:
            return default
    
    def safe_float(val, decimals=1, default=0.0):
        try:
            if pd.isna(val):
                return default
            return round(float(val), decimals)
        except:
            return default
    
    # Build match info
    teams = []
    gold_timeline = {'timestamps': [10, 15, 20, 25], 'team1': [], 'team2': []}
    team_names = []
    
    for idx, (_, team_row) in enumerate(team_data.iterrows()):
        team_name = team_row['teamname']
        team_names.append(team_name)
        team_players = player_data[player_data['teamname'] == team_name]
        
        # Calculate team totals for gold timeline from players
        team_gold_10 = team_players['goldat10'].sum() if 'goldat10' in team_players else 0
        team_gold_15 = team_players['goldat15'].sum() if 'goldat15' in team_players else 0
        team_gold_20 = team_players['goldat20'].sum() if 'goldat20' in team_players else 0
        team_gold_25 = team_players['goldat25'].sum() if 'goldat25' in team_players else 0
        
        if idx == 0:
            gold_timeline['team1'] = [safe_int(team_gold_10), safe_int(team_gold_15), safe_int(team_gold_20), safe_int(team_gold_25)]
        else:
            gold_timeline['team2'] = [safe_int(team_gold_10), safe_int(team_gold_15), safe_int(team_gold_20), safe_int(team_gold_25)]
        
        players = []
        key_player = None
        max_damage_share = 0
        
        for _, p in team_players.iterrows():
            # Full player stats
            player_stats = {
                'name': p['playername'],
                'champion': p['champion'],
                'position': p['position'],
                'side': p.get('side', ''),
                # Combat
                'kills': safe_int(p['kills']),
                'deaths': safe_int(p['deaths']),
                'assists': safe_int(p['assists']),
                'kda': safe_float((p['kills'] + p['assists']) / max(p['deaths'], 1), 2),
                # Multi-kills
                'double_kills': safe_int(p.get('doublekills', 0)),
                'triple_kills': safe_int(p.get('triplekills', 0)),
                'quadra_kills': safe_int(p.get('quadrakills', 0)),
                'penta_kills': safe_int(p.get('pentakills', 0)),
                # First blood
                'first_blood_kill': safe_int(p.get('firstbloodkill', 0)),
                'first_blood_assist': safe_int(p.get('firstbloodassist', 0)),
                # CS
                'cs': safe_int(p.get('total cs', 0)),
                'minion_kills': safe_int(p.get('minionkills', 0)),
                'monster_kills': safe_int(p.get('monsterkills', 0)),
                'cspm': safe_float(p.get('cspm', 0)),
                # Damage
                'damage_to_champions': safe_int(p.get('damagetochampions', 0)),
                'dpm': safe_float(p.get('dpm', 0)),
                'damage_share': safe_float(p.get('damageshare', 0) * 100, 1),
                'damage_taken_pm': safe_float(p.get('damagetakenperminute', 0)),
                # Gold
                'total_gold': safe_int(p.get('totalgold', 0)),
                'earned_gold': safe_int(p.get('earnedgold', 0)),
                'gold_pm': safe_float(p.get('earned gpm', 0)),
                'gold_share': safe_float(p.get('earnedgoldshare', 0) * 100 if p.get('earnedgoldshare') else 0, 1),
                # Vision
                'wards_placed': safe_int(p.get('wardsplaced', 0)),
                'wards_killed': safe_int(p.get('wardskilled', 0)),
                'control_wards': safe_int(p.get('controlwardsbought', 0)),
                'vision_score': safe_int(p.get('visionscore', 0)),
                'vspm': safe_float(p.get('vspm', 0)),
                # Gold at timestamps
                'gold_at_10': safe_int(p.get('goldat10', 0)),
                'gold_at_15': safe_int(p.get('goldat15', 0)),
                'gold_at_20': safe_int(p.get('goldat20', 0)),
                'gold_at_25': safe_int(p.get('goldat25', 0)),
                # Gold diff at timestamps
                'gold_diff_10': safe_int(p.get('golddiffat10', 0)),
                'gold_diff_15': safe_int(p.get('golddiffat15', 0)),
                'gold_diff_20': safe_int(p.get('golddiffat20', 0)),
                'gold_diff_25': safe_int(p.get('golddiffat25', 0)),
                # CS at timestamps
                'cs_at_10': safe_int(p.get('csat10', 0)),
                'cs_at_15': safe_int(p.get('csat15', 0)),
            }
            
            players.append(player_stats)
            
            # Track key player (highest damage share)
            dmg_share = safe_float(p.get('damageshare', 0))
            if dmg_share > max_damage_share:
                max_damage_share = dmg_share
                key_player = {
                    'name': p['playername'],
                    'champion': p['champion'],
                    'position': p['position'],
                    'kda': player_stats['kda'],
                    'damage_share': player_stats['damage_share'],
                    'kills': player_stats['kills'],
                    'deaths': player_stats['deaths'],
                    'assists': player_stats['assists'],
                }
        
        # Sort players by position order
        position_order = {'top': 0, 'jng': 1, 'mid': 2, 'bot': 3, 'sup': 4}
        players.sort(key=lambda x: position_order.get(x['position'].lower(), 5))
        
        # Calculate team totals
        total_damage = sum(p['damage_to_champions'] for p in players)
        total_gold = sum(p['total_gold'] for p in players)
        total_vision = sum(p['vision_score'] for p in players)
        
        # Get bans (from first player row)
        bans = []
        if len(team_players) > 0:
            first_player = team_players.iloc[0]
            for i in range(1, 6):
                ban = first_player.get(f'ban{i}', '')
                if ban and not pd.isna(ban):
                    bans.append(ban)
        
        # Get picks
        picks = []
        if len(team_players) > 0:
            first_player = team_players.iloc[0]
            for i in range(1, 6):
                pick = first_player.get(f'pick{i}', '')
                if pick and not pd.isna(pick):
                    picks.append(pick)
        
        # Dragon types
        dragon_types = {
            'infernals': safe_int(team_row.get('infernals', 0)),
            'mountains': safe_int(team_row.get('mountains', 0)),
            'clouds': safe_int(team_row.get('clouds', 0)),
            'oceans': safe_int(team_row.get('oceans', 0)),
            'chemtechs': safe_int(team_row.get('chemtechs', 0)),
            'hextechs': safe_int(team_row.get('hextechs', 0)),
            'elders': safe_int(team_row.get('elders', 0)),
        }
        
        teams.append({
            'name': team_name,
            'side': team_players.iloc[0]['side'] if len(team_players) > 0 else '',
            'result': 'Win' if team_row['result'] == 1 else 'Loss',
            'kills': safe_int(team_row['teamkills']),
            'deaths': safe_int(team_row['teamdeaths']),
            'dragons': safe_int(team_row['dragons']),
            'dragon_types': dragon_types,
            'barons': safe_int(team_row['barons']),
            'heralds': safe_int(team_row.get('heralds', 0)),
            'void_grubs': safe_int(team_row.get('void_grubs', 0)),
            'towers': safe_int(team_row['towers']),
            'inhibitors': safe_int(team_row.get('inhibitors', 0)),
            'first_blood': safe_int(team_row.get('firstblood', 0)),
            'first_dragon': safe_int(team_row.get('firstdragon', 0)),
            'first_baron': safe_int(team_row.get('firstbaron', 0)),
            'first_tower': safe_int(team_row.get('firsttower', 0)),
            'turret_plates': safe_int(team_row.get('turretplates', 0)),
            'total_gold': total_gold,
            'total_damage': total_damage,
            'total_vision': total_vision,
            'bans': bans,
            'picks': picks,
            'key_player': key_player,
            'players': players
        })
    
    # Calculate gold diff timeline
    gold_diff_timeline = []
    for i in range(4):
        diff = gold_timeline['team1'][i] - gold_timeline['team2'][i] if gold_timeline['team1'] and gold_timeline['team2'] else 0
        gold_diff_timeline.append(diff)
    
    match_info = {
        'gameid': game_id,
        'date': str(team_data.iloc[0]['date'])[:10],
        'league': team_data.iloc[0]['league'],
        'patch': team_data.iloc[0].get('patch', ''),
        'gamelength': safe_float(team_data.iloc[0]['gamelength'] / 60, 1),
        'playoffs': safe_int(team_data.iloc[0].get('playoffs', 0)),
        'gold_timeline': gold_timeline,
        'gold_diff_timeline': gold_diff_timeline,
        'teams': teams
    }
    
    return match_info


@app.get("/api/players/{player_name}/leagues")
async def get_player_leagues(player_name: str):
    """Get all leagues a player has played in"""
    global player_df, league_details
    if player_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = player_df[player_df['playername'].str.upper() == player_name.upper()]
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="Player not found")
    
    leagues = df.groupby('league').agg({
        'gameid': 'count',
        'result': 'mean',
        'kills': 'mean',
        'deaths': 'mean',
        'assists': 'mean',
    }).reset_index()
    
    leagues['kda'] = (leagues['kills'] + leagues['assists']) / leagues['deaths'].replace(0, 1)
    leagues['winrate'] = (leagues['result'] * 100).round(1)
    leagues['kda'] = leagues['kda'].round(2)
    
    leagues.columns = ['league', 'games', 'winrate_raw', 'avg_kills', 'avg_deaths', 'avg_assists', 'kda', 'winrate']
    leagues = leagues[['league', 'games', 'winrate', 'kda', 'avg_kills', 'avg_deaths', 'avg_assists']]
    
    # Add tier info
    if league_details is not None:
        tier_map = dict(zip(league_details['league'], league_details['tier']))
        region_map = dict(zip(league_details['league'], league_details['region']))
        leagues['tier'] = leagues['league'].map(tier_map).fillna('C')
        leagues['region'] = leagues['league'].map(region_map).fillna('Unknown')
    else:
        leagues['tier'] = 'C'
        leagues['region'] = 'Unknown'
    
    leagues = leagues.sort_values('games', ascending=False)
    
    return {
        "player": player_name,
        "total_leagues": len(leagues),
        "leagues": leagues.to_dict(orient='records')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

