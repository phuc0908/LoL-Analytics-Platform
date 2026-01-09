"""
Advanced Statistics with League Weighting and Regional Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from league_config import get_league_weight, get_league_tier, get_region, LEAGUE_TIERS, REGIONS


class AdvancedStatsProcessor:
    """Process advanced statistics with league weighting"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.team_df = None
        self.player_df = None
        
    def load_data(self):
        """Load and prepare data"""
        print(f"📂 Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        
        # Add league metadata
        self.df['league_weight'] = self.df['league'].apply(get_league_weight)
        self.df['league_tier'] = self.df['league'].apply(get_league_tier)
        self.df['region'] = self.df['league'].apply(get_region)
        
        # Separate team and player data
        self.team_df = self.df[self.df['position'] == 'team'].copy()
        self.player_df = self.df[self.df['position'] != 'team'].copy()
        
        print(f"✅ Loaded {len(self.df):,} rows")
        print(f"📊 Team rows: {len(self.team_df):,}")
        print(f"👤 Player rows: {len(self.player_df):,}")
        
        return self
    
    def get_league_details(self) -> pd.DataFrame:
        """Get detailed league statistics"""
        league_stats = self.team_df.groupby('league').agg({
            'gameid': 'nunique',
            'result': 'count',
            'playoffs': 'mean',
            'gamelength': 'mean',
            'teamkills': 'mean',
            'dragons': 'mean',
            'barons': 'mean',
            'league_weight': 'first',
            'league_tier': 'first',
            'region': 'first',
        }).reset_index()
        
        league_stats.columns = [
            'league', 'games', 'total_rows', 'playoff_rate', 
            'avg_gamelength', 'avg_kills', 'avg_dragons', 'avg_barons',
            'weight', 'tier', 'region'
        ]
        
        # Calculate teams per league
        teams_per_league = self.team_df.groupby('league')['teamname'].nunique().reset_index()
        teams_per_league.columns = ['league', 'teams']
        league_stats = league_stats.merge(teams_per_league, on='league')
        
        # Calculate players per league
        players_per_league = self.player_df.groupby('league')['playername'].nunique().reset_index()
        players_per_league.columns = ['league', 'players']
        league_stats = league_stats.merge(players_per_league, on='league')
        
        # Format
        league_stats['playoff_rate'] = (league_stats['playoff_rate'] * 100).round(1)
        league_stats['avg_gamelength'] = (league_stats['avg_gamelength'] / 60).round(1)
        league_stats['avg_kills'] = league_stats['avg_kills'].round(1)
        league_stats['avg_dragons'] = league_stats['avg_dragons'].round(2)
        league_stats['avg_barons'] = league_stats['avg_barons'].round(2)
        
        return league_stats.sort_values('games', ascending=False)
    
    def get_region_stats(self) -> pd.DataFrame:
        """Get statistics by region"""
        region_stats = self.team_df.groupby('region').agg({
            'gameid': 'nunique',
            'gamelength': 'mean',
            'teamkills': 'mean',
            'dragons': 'mean',
            'barons': 'mean',
        }).reset_index()
        
        region_stats.columns = [
            'region', 'games', 'avg_gamelength', 
            'avg_kills', 'avg_dragons', 'avg_barons'
        ]
        
        region_stats['flag'] = region_stats['region'].map(
            lambda r: REGIONS.get(r, {}).get('flag', '🌐')
        )
        
        region_stats['avg_gamelength'] = (region_stats['avg_gamelength'] / 60).round(1)
        region_stats['avg_kills'] = region_stats['avg_kills'].round(1)
        
        return region_stats.sort_values('games', ascending=False)
    
    def get_matches_by_league(self, league: str, limit: int = 50) -> pd.DataFrame:
        """Get recent matches for a specific league"""
        df = self.team_df[self.team_df['league'] == league].copy()
        
        # Get unique games
        games = df.groupby('gameid').agg({
            'date': 'first',
            'teamname': lambda x: list(x),
            'result': lambda x: list(x),
            'teamkills': lambda x: list(x),
            'gamelength': 'first',
            'playoffs': 'first',
            'game': 'first',
        }).reset_index()
        
        # Format match data
        matches = []
        for _, row in games.iterrows():
            teams = row['teamname']
            results = row['result']
            kills = row['teamkills']
            
            if len(teams) == 2:
                winner_idx = results.index(1) if 1 in results else 0
                loser_idx = 1 - winner_idx
                
                matches.append({
                    'gameid': row['gameid'],
                    'date': row['date'],
                    'team1': teams[0],
                    'team2': teams[1],
                    'winner': teams[winner_idx],
                    'score': f"{int(kills[0])}-{int(kills[1])}",
                    'gamelength': round(row['gamelength'] / 60, 1),
                    'playoffs': int(row['playoffs']),
                    'game_number': int(row['game']),
                })
        
        matches_df = pd.DataFrame(matches)
        if len(matches_df) > 0:
            matches_df = matches_df.sort_values('date', ascending=False).head(limit)
        
        return matches_df
    
    def get_players_by_league(self, league: str) -> pd.DataFrame:
        """Get player statistics for a specific league"""
        df = self.player_df[self.player_df['league'] == league].copy()
        
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
        player_stats['avg_cspm'] = player_stats['avg_cspm'].round(1)
        
        return player_stats.sort_values('kda', ascending=False)
    
    def get_champions_by_league(self, league: str) -> pd.DataFrame:
        """Get champion statistics for a specific league"""
        df = self.player_df[self.player_df['league'] == league].copy()
        
        champ_stats = df.groupby('champion').agg({
            'gameid': 'count',
            'result': 'mean',
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
        }).reset_index()
        
        champ_stats['kda'] = (champ_stats['kills'] + champ_stats['assists']) / champ_stats['deaths'].replace(0, 1)
        
        # Calculate pick rate
        total_games = df['gameid'].nunique()
        champ_stats['pickrate'] = (champ_stats['gameid'] / total_games * 100).round(1)
        
        champ_stats.columns = [
            'champion', 'games', 'winrate', 'avg_kills', 'avg_deaths', 'avg_assists', 'kda', 'pickrate'
        ]
        
        champ_stats['winrate'] = (champ_stats['winrate'] * 100).round(1)
        champ_stats['kda'] = champ_stats['kda'].round(2)
        
        return champ_stats.sort_values('games', ascending=False)
    
    def get_teams_by_league(self, league: str) -> pd.DataFrame:
        """Get team statistics for a specific league"""
        df = self.team_df[self.team_df['league'] == league].copy()
        
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
        
        return team_stats.sort_values('winrate', ascending=False)
    
    def get_champions_by_region(self, region: str) -> pd.DataFrame:
        """Get champion statistics for a specific region"""
        df = self.player_df[self.player_df['region'] == region].copy()
        
        champ_stats = df.groupby('champion').agg({
            'gameid': 'count',
            'result': 'mean',
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
        }).reset_index()
        
        champ_stats['kda'] = (champ_stats['kills'] + champ_stats['assists']) / champ_stats['deaths'].replace(0, 1)
        
        total_games = df['gameid'].nunique()
        champ_stats['pickrate'] = (champ_stats['gameid'] / total_games * 100).round(1)
        
        champ_stats.columns = [
            'champion', 'games', 'winrate', 'avg_kills', 'avg_deaths', 'avg_assists', 'kda', 'pickrate'
        ]
        
        champ_stats['winrate'] = (champ_stats['winrate'] * 100).round(1)
        champ_stats['kda'] = champ_stats['kda'].round(2)
        
        return champ_stats.sort_values('games', ascending=False)
    
    def export_all_stats(self, output_dir: str = "../data"):
        """Export all statistics to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n📊 Exporting advanced statistics...")
        
        # League details
        leagues = self.get_league_details()
        leagues.to_csv(output_path / "league_details.csv", index=False)
        print(f"✅ league_details.csv ({len(leagues)} leagues)")
        
        # Region stats
        regions = self.get_region_stats()
        regions.to_csv(output_path / "region_stats.csv", index=False)
        print(f"✅ region_stats.csv ({len(regions)} regions)")
        
        print("\n✅ All statistics exported!")


def main():
    """Generate advanced statistics"""
    data_path = Path(__file__).parent.parent.parent / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
    
    processor = AdvancedStatsProcessor(str(data_path))
    processor.load_data()
    processor.export_all_stats()
    
    # Test league details
    print("\n" + "=" * 60)
    print("🎮 LCK MATCHES (Sample)")
    print("=" * 60)
    print(processor.get_matches_by_league('LCK', 5).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("👤 LCK TOP PLAYERS")
    print("=" * 60)
    print(processor.get_players_by_league('LCK').head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🏆 LCK TOP CHAMPIONS")
    print("=" * 60)
    print(processor.get_champions_by_league('LCK').head(10).to_string(index=False))


if __name__ == "__main__":
    main()

