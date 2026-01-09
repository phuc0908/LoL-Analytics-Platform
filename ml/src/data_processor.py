"""
Data Processor for LoL Esports Match Data
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class LoLDataProcessor:
    """Process LoL esports match data for ML training"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.team_df = None
        self.player_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        print(f"📂 Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        print(f"✅ Loaded {len(self.df):,} rows, {len(self.df.columns)} columns")
        return self.df
    
    def separate_team_player_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Separate team-level and player-level data"""
        # Team data has position == 'team'
        self.team_df = self.df[self.df['position'] == 'team'].copy()
        self.player_df = self.df[self.df['position'] != 'team'].copy()
        
        print(f"📊 Team rows: {len(self.team_df):,}")
        print(f"👤 Player rows: {len(self.player_df):,}")
        
        return self.team_df, self.player_df
    
    def create_match_features(self) -> pd.DataFrame:
        """Create features for win prediction model"""
        if self.team_df is None:
            self.separate_team_player_data()
        
        # Get Blue and Red side data for each game
        blue_df = self.team_df[self.team_df['side'] == 'Blue'].copy()
        red_df = self.team_df[self.team_df['side'] == 'Red'].copy()
        
        # Merge on gameid
        match_df = blue_df.merge(
            red_df, 
            on='gameid', 
            suffixes=('_blue', '_red')
        )
        
        print(f"🎮 Total matches: {len(match_df):,}")
        
        # Create features
        features = pd.DataFrame()
        features['gameid'] = match_df['gameid']
        features['blue_team'] = match_df['teamname_blue']
        features['red_team'] = match_df['teamname_red']
        features['league'] = match_df['league_blue']
        features['patch'] = match_df['patch_blue']
        features['gamelength'] = match_df['gamelength_blue']
        
        # Target: Blue team wins (1) or loses (0)
        features['blue_wins'] = match_df['result_blue'].astype(int)
        
        # === DRAFT FEATURES ===
        # We'll encode champions as categorical later
        for i in range(1, 6):
            features[f'blue_pick{i}'] = match_df[f'pick{i}_blue']
            features[f'red_pick{i}'] = match_df[f'pick{i}_red']
            features[f'blue_ban{i}'] = match_df[f'ban{i}_blue']
            features[f'red_ban{i}'] = match_df[f'ban{i}_red']
        
        # === OBJECTIVE FEATURES ===
        obj_cols = ['dragons', 'heralds', 'barons', 'towers', 'inhibitors', 
                    'void_grubs', 'turretplates', 'elders']
        
        for col in obj_cols:
            if f'{col}_blue' in match_df.columns:
                features[f'blue_{col}'] = match_df[f'{col}_blue'].fillna(0)
                features[f'red_{col}'] = match_df[f'{col}_red'].fillna(0)
        
        # === FIRST OBJECTIVES (Pre-game predictors when available) ===
        first_cols = ['firstblood', 'firstdragon', 'firstherald', 'firstbaron', 'firsttower']
        for col in first_cols:
            if f'{col}_blue' in match_df.columns:
                features[f'blue_{col}'] = match_df[f'{col}_blue'].fillna(0)
        
        # === COMBAT STATS ===
        combat_cols = ['teamkills', 'teamdeaths']
        for col in combat_cols:
            if f'{col}_blue' in match_df.columns:
                features[f'blue_{col}'] = match_df[f'{col}_blue'].fillna(0)
                features[f'red_{col}'] = match_df[f'{col}_red'].fillna(0)
        
        # === GOLD/XP DIFF AT DIFFERENT TIMES ===
        time_points = [10, 15, 20, 25]
        for t in time_points:
            if f'golddiffat{t}_blue' in match_df.columns:
                features[f'golddiff_at{t}'] = match_df[f'golddiffat{t}_blue'].fillna(0)
                features[f'xpdiff_at{t}'] = match_df[f'xpdiffat{t}_blue'].fillna(0)
                features[f'csdiff_at{t}'] = match_df[f'csdiffat{t}_blue'].fillna(0)
        
        self.match_features = features
        return features
    
    def prepare_ml_data(self, use_early_game: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for ML training
        
        Args:
            use_early_game: If True, include early game stats (gold diff at 10/15 min)
                           This is for live prediction. If False, only use draft.
        """
        if not hasattr(self, 'match_features'):
            self.create_match_features()
        
        df = self.match_features.copy()
        
        # Features for prediction
        feature_cols = []
        
        # Numeric features
        numeric_cols = [
            'blue_dragons', 'red_dragons',
            'blue_heralds', 'red_heralds', 
            'blue_barons', 'red_barons',
            'blue_towers', 'red_towers',
            'blue_teamkills', 'red_teamkills',
            'blue_teamdeaths', 'red_teamdeaths',
        ]
        
        if use_early_game:
            numeric_cols.extend([
                'golddiff_at10', 'xpdiff_at10', 'csdiff_at10',
                'golddiff_at15', 'xpdiff_at15', 'csdiff_at15',
            ])
        
        # First objectives
        first_obj_cols = ['blue_firstblood', 'blue_firstdragon', 
                         'blue_firstherald', 'blue_firsttower']
        
        # Add available columns
        for col in numeric_cols + first_obj_cols:
            if col in df.columns:
                feature_cols.append(col)
        
        # Handle missing values
        X = df[feature_cols].fillna(0)
        y = df['blue_wins']
        
        print(f"✅ Prepared {len(X):,} samples with {len(feature_cols)} features")
        print(f"📊 Blue win rate: {y.mean()*100:.1f}%")
        
        return X, y
    
    def get_champion_stats(self) -> pd.DataFrame:
        """Calculate champion win rates and pick rates"""
        if self.player_df is None:
            self.separate_team_player_data()
        
        champ_stats = self.player_df.groupby('champion').agg({
            'gameid': 'count',
            'result': 'mean',
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
        }).reset_index()
        
        champ_stats.columns = ['champion', 'games', 'winrate', 'avg_kills', 'avg_deaths', 'avg_assists']
        champ_stats['kda'] = (champ_stats['avg_kills'] + champ_stats['avg_assists']) / champ_stats['avg_deaths'].replace(0, 1)
        champ_stats = champ_stats.sort_values('games', ascending=False)
        
        return champ_stats
    
    def get_team_stats(self) -> pd.DataFrame:
        """Calculate team statistics"""
        if self.team_df is None:
            self.separate_team_player_data()
        
        team_stats = self.team_df.groupby('teamname').agg({
            'gameid': 'count',
            'result': 'mean',
            'teamkills': 'mean',
            'teamdeaths': 'mean',
            'dragons': 'mean',
            'barons': 'mean',
            'towers': 'mean',
            'gamelength': 'mean',
        }).reset_index()
        
        team_stats.columns = ['team', 'games', 'winrate', 'avg_kills', 'avg_deaths', 
                             'avg_dragons', 'avg_barons', 'avg_towers', 'avg_gamelength']
        team_stats = team_stats.sort_values('winrate', ascending=False)
        
        return team_stats
    
    def get_player_stats(self) -> pd.DataFrame:
        """Calculate player statistics"""
        if self.player_df is None:
            self.separate_team_player_data()
        
        player_stats = self.player_df.groupby(['playername', 'teamname', 'position']).agg({
            'gameid': 'count',
            'result': 'mean',
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
            'damagetochampions': 'mean',
            'dpm': 'mean',
            'cspm': 'mean',
            'visionscore': 'mean',
        }).reset_index()
        
        player_stats.columns = ['player', 'team', 'position', 'games', 'winrate',
                               'avg_kills', 'avg_deaths', 'avg_assists', 
                               'avg_damage', 'avg_dpm', 'avg_cspm', 'avg_vision']
        player_stats['kda'] = (player_stats['avg_kills'] + player_stats['avg_assists']) / player_stats['avg_deaths'].replace(0, 1)
        player_stats = player_stats.sort_values('games', ascending=False)
        
        return player_stats
    
    def get_league_stats(self) -> pd.DataFrame:
        """Get statistics by league"""
        if self.team_df is None:
            self.separate_team_player_data()
            
        league_stats = self.team_df.groupby('league').agg({
            'gameid': 'nunique',
            'gamelength': 'mean',
            'teamkills': 'mean',
        }).reset_index()
        
        league_stats.columns = ['league', 'games', 'avg_gamelength', 'avg_kills']
        league_stats = league_stats.sort_values('games', ascending=False)
        
        return league_stats


if __name__ == "__main__":
    # Test the processor
    processor = LoLDataProcessor("../2025_LoL_esports_match_data_from_OraclesElixir.csv")
    processor.load_data()
    processor.separate_team_player_data()
    
    # Create features
    features = processor.create_match_features()
    print("\n📋 Match features sample:")
    print(features.head())
    
    # Prepare ML data
    X, y = processor.prepare_ml_data()
    print(f"\n🎯 X shape: {X.shape}")
    print(f"🎯 y shape: {y.shape}")
    
    # Get stats
    print("\n🏆 Top 10 Champions by games:")
    print(processor.get_champion_stats().head(10))
    
    print("\n🏆 Top 10 Teams by winrate (min 20 games):")
    team_stats = processor.get_team_stats()
    print(team_stats[team_stats['games'] >= 20].head(10))

