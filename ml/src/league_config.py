"""
League Configuration with Importance Weights
Used for weighted statistics and analysis
"""

# League Tiers with weights
LEAGUE_TIERS = {
    # Tier S - International (Weight 3.0)
    "S": {
        "weight": 3.0,
        "name": "International",
        "leagues": ["WLDs", "MSI", "Worlds"],
        "color": "#FFD700"  # Gold
    },
    
    # Tier A - Major Regions (Weight 2.0)
    "A": {
        "weight": 2.0,
        "name": "Major Regions",
        "leagues": ["LCK", "LPL", "LEC", "LCS", "LTA N", "LTA S"],
        "color": "#C0C0C0"  # Silver
    },
    
    # Tier B - Minor Regions (Weight 1.5)
    "B": {
        "weight": 1.5,
        "name": "Minor Regions",
        "leagues": ["PCS", "VCS", "LJL", "CBLOL", "LLA", "LCO", "TCL", "LCL"],
        "color": "#CD7F32"  # Bronze
    },
    
    # Tier C - Regional Leagues (Weight 1.0)
    "C": {
        "weight": 1.0,
        "name": "Regional Leagues",
        "leagues": [
            "LFL", "PRM", "NLC", "LVP SL", "LIT", "TCL", "EBL", "HLL", "UL",
            "EM", "LAS", "CD", "AL", "LCP", "RL", "HC", "HM", "LRS", "LRN",
            "ESLOL", "GLL", "LCK CL", "LDL", "LFL2", "ROL"
        ],
        "color": "#4A90D9"  # Blue
    },
    
    # Tier D - Academy/Challenger (Weight 0.8)
    "D": {
        "weight": 0.8,
        "name": "Academy/Challenger",
        "leagues": ["LCKC", "NACL", "LDL", "EUM", "LLA CL", "CBLOLA"],
        "color": "#808080"  # Gray
    }
}

# Region mapping
REGIONS = {
    "Korea": {
        "leagues": ["LCK", "LCKC", "KeSPA"],
        "flag": "🇰🇷",
        "color": "#0047AB"
    },
    "China": {
        "leagues": ["LPL", "LDL", "DCup"],
        "flag": "🇨🇳",
        "color": "#DE2910"
    },
    "Europe": {
        "leagues": ["LEC", "LFL", "LFL2", "PRM", "NLC", "LVP SL", "LIT", "EBL", 
                   "UL", "HLL", "EM", "GLL", "ESLOL", "AL", "HM", "TCL"],
        "flag": "🇪🇺",
        "color": "#003399"
    },
    "North America": {
        "leagues": ["LCS", "NACL", "LTA N"],
        "flag": "🇺🇸",
        "color": "#B22234"
    },
    "Latin America": {
        "leagues": ["LLA", "LTA S", "CD", "LAS", "LLA CL"],
        "flag": "🌎",
        "color": "#009B3A"
    },
    "Southeast Asia": {
        "leagues": ["PCS", "VCS", "LCP"],
        "flag": "🌏",
        "color": "#FF6600"
    },
    "Japan": {
        "leagues": ["LJL"],
        "flag": "🇯🇵",
        "color": "#BC002D"
    },
    "Brazil": {
        "leagues": ["CBLOL", "CBLOLA"],
        "flag": "🇧🇷",
        "color": "#009C3B"
    },
    "Oceania": {
        "leagues": ["LCO"],
        "flag": "🇦🇺",
        "color": "#00008B"
    },
    "International": {
        "leagues": ["WLDs", "MSI", "Worlds", "EWC", "All-Star"],
        "flag": "🌍",
        "color": "#FFD700"
    }
}


def get_league_weight(league: str) -> float:
    """Get weight for a league"""
    for tier_data in LEAGUE_TIERS.values():
        if league in tier_data["leagues"]:
            return tier_data["weight"]
    return 1.0  # Default weight


def get_league_tier(league: str) -> str:
    """Get tier for a league"""
    for tier, tier_data in LEAGUE_TIERS.items():
        if league in tier_data["leagues"]:
            return tier
    return "C"  # Default tier


def get_region(league: str) -> str:
    """Get region for a league"""
    for region, region_data in REGIONS.items():
        if league in region_data["leagues"]:
            return region
    return "Other"


def get_all_leagues_with_info():
    """Get all leagues with their tier, weight, and region info"""
    result = {}
    for tier, tier_data in LEAGUE_TIERS.items():
        for league in tier_data["leagues"]:
            result[league] = {
                "tier": tier,
                "tier_name": tier_data["name"],
                "weight": tier_data["weight"],
                "color": tier_data["color"],
                "region": get_region(league)
            }
    return result


if __name__ == "__main__":
    print("League Configuration Test")
    print("=" * 50)
    
    test_leagues = ["LCK", "LPL", "WLDs", "LFL", "LCKC", "PCS"]
    for league in test_leagues:
        print(f"{league}: Tier {get_league_tier(league)}, Weight {get_league_weight(league)}, Region: {get_region(league)}")

