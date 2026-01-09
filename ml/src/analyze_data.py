"""
Analyze LoL data structure for series and league weighting
"""
import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent.parent / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
df = pd.read_csv(data_path, low_memory=False)

print("=" * 60)
print("📊 DATA STRUCTURE ANALYSIS")
print("=" * 60)

print(f"\n📈 Total rows: {len(df):,}")
print(f"🎮 Unique gameids: {df['gameid'].nunique():,}")
print(f"📝 Rows per game: {len(df) / df['gameid'].nunique():.1f}")

# Check playoffs column (indicates BO3/BO5)
print("\n" + "=" * 60)
print("🏆 PLAYOFFS (BO Series indicator)")
print("=" * 60)
print(df['playoffs'].value_counts())

# Check game column (game number in series)
print("\n" + "=" * 60)
print("🎯 GAME NUMBER IN SERIES")
print("=" * 60)
print(df['game'].value_counts().sort_index())

# Top leagues with game counts
print("\n" + "=" * 60)
print("🌍 LEAGUES BY GAMES")
print("=" * 60)
league_stats = df[df['position'] == 'team'].groupby('league').agg({
    'gameid': 'nunique',
    'playoffs': 'mean'  # % of playoff games
}).reset_index()
league_stats.columns = ['league', 'games', 'playoff_rate']
league_stats = league_stats.sort_values('games', ascending=False)
print(league_stats.head(20).to_string(index=False))

# Check for international tournaments
print("\n" + "=" * 60)
print("🌐 INTERNATIONAL TOURNAMENTS")
print("=" * 60)
intl_keywords = ['Worlds', 'MSI', 'International', 'WLDs', 'All-Star']
for kw in intl_keywords:
    matches = df[df['league'].str.contains(kw, case=False, na=False)]
    if len(matches) > 0:
        print(f"{kw}: {matches['gameid'].nunique()} games")

# Analyze series structure (BO1 vs BO3 vs BO5)
print("\n" + "=" * 60)
print("📋 SERIES STRUCTURE (BO1/BO3/BO5)")
print("=" * 60)

# Group by match (assuming gameid format contains match identifier)
# Check max game number per league
max_games = df.groupby('league')['game'].max().sort_values(ascending=False)
print("\nMax games in series by league:")
print(max_games.head(20))

# Identify series by looking at consecutive games
print("\n" + "=" * 60)
print("🏅 LEAGUE IMPORTANCE TIERS")
print("=" * 60)
print("""
Tier S (Weight 3.0): Worlds, MSI
Tier A (Weight 2.0): LCK, LPL, LEC, LCS
Tier B (Weight 1.5): PCS, VCS, LJL, CBLOL, LLA
Tier C (Weight 1.0): Regional leagues (LFL, PRM, etc.)
Tier D (Weight 0.8): Academy/Challenger leagues
""")

# Sample gameid to understand format
print("\n" + "=" * 60)
print("🔍 SAMPLE GAME IDs")
print("=" * 60)
print(df['gameid'].head(10).tolist())

