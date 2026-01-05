"""""
FIFA World Cup 2026 Prediction Model: Dixon-Coles Implementation
Author: Will Higgin
Framework: Dixon-Coles (MLE) with Time-Decay and Bracket Simulation
"""""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import random
from tqdm import tqdm

# Load and prepare data
df = pd.read_csv("results.csv")
df['date'] = pd.to_datetime(df['date'])

#team and group defintions
#(Irreland and qualification teams so that question 6 can be answered using this algoirthm)
#because they are not assigned to a group they are ignored in this script
teams = [
    "Mexico", "South Korea", "South Africa", "Denmark",
    "Canada", "Switzerland", "Qatar", "Italy",
    "Brazil", "Morocco", "Scotland", "Haiti",
    "United States", "Paraguay", "Australia", "Turkey",
    "Germany", "Ecuador", "Ivory Coast", "Curaçao",
    "Netherlands", "Japan", "Tunisia", "Ukraine",
    "Belgium", "Iran", "Egypt", "New Zealand",
    "Spain", "Uruguay", "Saudi Arabia", "Cape Verde",
    "France", "Senegal", "Norway", "Iraq",
    "Argentina", "Austria", "Algeria", "Jordan",
    "Portugal", "Colombia", "Uzbekistan", "DR Congo",
    "England", "Croatia", "Ghana", "Panama", "Republic of Ireland", "Czechia", "Denmark", "North Macedonia"
]
#odds correct 20/12/2025
#odds from bet365
odds = {
    # Group A
    "Mexico":  {"group": 1.2, "final": 23.0},
    "South Korea": {"group": 1.44, "final": 126.0},
    "South Africa": {"group": 2.1, "final": 251.0},
    "Denmark": {"group": None, "final": 51.0},

    # Group B
    "Canada": {"group": 1.44, "final": 81.0},
    "Switzerland": {"group": 1.17, "final": 41.0},
    "Qatar": {"group": 2.75, "final": 251.0},
    "Italy": {"group": None, "final": 15.0},

    # Group C
    "Brazil": {"group": 1.02, "final": 5.0},
    "Morocco": {"group": 1.13, "final": 26.0},
    "Scotland": {"group": 1.29, "final": 81.0},
    "Haiti": {"group": 8.0, "final": 501.0},

    # Group D
    "United States": {"group": 1.2, "final": 29.0},
    "Paraguay": {"group": 1.4, "final": 51.0},
    "Australia": {"group": 1.9, "final": 126.0},
    "Turkey": {"group": None, "final": 67.0},

    # Group E
    "Germany": {"group": 1.03, "final": 6.5},
    "Ecuador": {"group": 1.1, "final": 29.0},
    "Ivory Coast": {"group": 1.22, "final": 101.0},
    "Curaçao": {"group": None, "final": 401.0},

    # Group F
    "Netherlands": {"group": 1.1, "final": 8.0},
    "Japan": {"group": 1.44, "final": 34.0},
    "Tunisia": {"group": 1.8, "final": 151.0},
    "Ukraine": {"group": None, "final": 101.0},

    # Group G
    "Belgium": {"group": 1.03, "final": 15.0},
    "Iran": {"group": 1.5, "final": 126.0},
    "Egypt": {"group": 1.36, "final": 101.0},
    "New Zealand": {"group": 2.38, "final": 301.0},

    # Group H
    "Spain": {"group": 1.02, "final": 3.25},
    "Uruguay": {"group": 1.17, "final": 23.0},
    "Saudi Arabia": {"group": 1.8, "final": 251.0},
    "Cape Verde": {"group": 2.63, "final": 401.0},

    # Group I
    "France": {"group": 1.04, "final": 5.0},
    "Senegal": {"group": 1.5, "final": 41.0},
    "Norway": {"group": 1.2, "final": 11.0},
    "Iraq": {"group": None, "final": 301.0},

    # Group J
    "Argentina": {"group": 1.03, "final": 5.0},
    "Austria": {"group": 1.25, "final": 67.0},
    "Algeria": {"group": 1.36, "final": 126.0},
    "Jordan": {"group": 3.4, "final": 501.0},

    # Group K
    "Portugal": {"group": 1.03, "final": 6.0},
    "Colombia": {"group": 1.14, "final": 21.0},
    "Uzbekistan": {"group": 2.75, "final": 401.0},
    "DR Congo": {"group": None, "final": 201.0},

    # Group L
    "England": {"group": 1.02, "final": 3.75},
    "Croatia": {"group": 1.25, "final": 41.0},
    "Ghana": {"group": 1.53, "final": 101.0},
    "Panama": {"group": 2.75, "final": 301.0},
}

df = df[
    (df.home_team.isin(teams)) &
    (df.away_team.isin(teams)) &
    (df.tournament != "Friendly")
    ]

# Time decay
HALF_LIFE_DAYS = 365 * 2
latest_date = df['date'].max()
df['days_ago'] = (latest_date - df['date']).dt.days
df['weight'] = np.exp(-np.log(2) * df['days_ago'] / HALF_LIFE_DAYS)

groups = {
    "A": ["Mexico", "South Korea", "South Africa", "Denmark"],
    "B": ["Canada", "Switzerland", "Qatar", "Italy"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Ukraine"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"]
}

# R32 bracket structure (from official FIFA schedule)
# Format: (Winner/Runner-up identifier, Third place groups if applicable)
#generated using ai
#letter is group, number is placing
#uses priority lists to get the best 3rd place teams
#best 3rd place teams are selected in select_best_thirds
#this code simply checks if a team has a qualifying team according to the fifa logic and populates the knockouts
R32_STRUCTURE = [
    ("A2", "B2"),
    ("E1", ["A", "B", "C", "D", "F"]),  # M74
    ("C1", "F2"),  # M75
    ("E2", "I2"),  # M76
    ("I1", ["C", "D", "F", "G", "H"]),  # M77
    ("A1", ["C", "E", "F", "H", "I"]),  # M78
    ("L1", ["E", "H", "I", "J", "K"]),  # M79
    ("G1", ["A", "E", "H", "I", "J"]),  # M80
    ("D1", ["B", "E", "F", "I", "J"]),  # M81
    ("B1", ["C", "D", "E", "G", "H"]),  # M82
    ("H1", "J2"),  # M83
    ("K1", ["D", "E", "I", "J", "L"]),  # M84
    ("F1", "C2"),  # M85
    ("K2", "L2"),  # M86
    ("D2", "G2"),  # M87
    ("J1", "H2")  # M88
]

team_idx = {t: i for i, t in enumerate(teams)}
n = len(teams)


#called by spicy.optimize below

def log_likelihood(params):
    """
    PARAMETER MAPPING
    Extracts team-specific coefficients from the optimiser's current guess (0s on first run).
    Maps every match in the  dataset to their team indices.
    """
    home_idx = df['home_team'].map(team_idx).values
    away_idx = df['away_team'].map(team_idx).values

    attack = params[:n]
    defence = params[n:2 * n]
    home_adv = params[-2]
    rho = params[-1]
    """
    Calculates expected goal rates (lambda and mu) for every match in the 
    dataset at once using vectorized exponential functions. x and y are the actual scorelines.
    """
    lmbda = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] - defence[home_idx])
    x = df['home_score'].values
    y = df['away_score'].values
    w = df['weight'].values

    """
    Adjusts the probability of low-scoring outcomes (0-0, 1-0, 0-1, 1-1).
    """
    dc = np.ones_like(x, dtype=float)
    #adjustemnts for low scoring outcomes 0-0 1-0 0-1 and 1-1
    dc[(x == 0) & (y == 0)] = 1 - lmbda[(x == 0) & (y == 0)] * mu[(x == 0) & (y == 0)] * rho
    dc[(x == 0) & (y == 1)] = 1 + lmbda[(x == 0) & (y == 1)] * rho
    dc[(x == 1) & (y == 0)] = 1 + mu[(x == 1) & (y == 0)] * rho
    dc[(x == 1) & (y == 1)] = 1 - rho
    dc = np.maximum(dc, 1e-10)

    """
    Calculates the total weighted log-likelihood. By returning the negative, 
    we allow the 'minimize' function to rather get the maximumr'.
    """
    ll = np.sum(w * (poisson.logpmf(x, lmbda) + poisson.logpmf(y, mu) + np.log(dc)))
    return -ll


"""
initialises all params at 0 for first iteration
calls log likelihood repeatedly and constantly adjusts params based on the output
does this until optimality reached
"""
init = np.zeros(2 * n + 2)
bounds = [(-2, 2)] * (2 * n) + [(-1, 1), (-0.1, 0.1)] #search area
res = minimize(log_likelihood, init, bounds=bounds, method="L-BFGS-B")
params = res.x




def simulate_match_score(home, away, neutral=False):
    """
    extract calibrated attack/defence for both teams.
    generate a 7x7 probability matrix of all scorelines (0-0 through 5-5).
    apply Dixon-Coles dependency correction (rho) to specific score cells.
    was made for bulk sim to account for gd
    law of large numbers dicatates it should get very similar results to
    simulate match probability in the simualations, but outside groups i
    use the match probability for simplicity as goal diff is not needed
     """

    i, j = team_idx[home], team_idx[away]
    attack_ = params[:n].copy()
    defence_ = params[n:2 * n].copy()
    home_adv_ = params[-2] if not neutral else 0  # Remove home advantage for neutral venues
    attack_ -= np.mean(attack_)
    lmbda = np.exp(attack_[i] - defence_[j] + home_adv_)
    mu = np.exp(attack_[j] - defence_[i])
    home_goals = min(np.random.poisson(lmbda), 10)
    away_goals = min(np.random.poisson(mu), 10)

    return home_goals, away_goals


def match_win_probability(home, away, neutral=False):
    """
    works same as sim match score but sums the grid triangles to derive
    Home/Draw/Away probabilities.
    is used during the monte carlo simulations a team with a x% chancve of winning should
    advance approximately x% of the time allowing for a fair simulation based on the
    probabilities claculated by the model
    """
    i, j = team_idx[home], team_idx[away]

    attack_ = params[:n].copy()
    defence_ = params[n:2 * n].copy()
    home_adv_ = params[-2] if not neutral else 0
    rho_ = params[-1]


    lmbda = np.exp(attack_[i] - defence_[j] + home_adv_)
    mu = np.exp(attack_[j] - defence_[i])

    # Use 7x7 grid for better accuracy (covers ~99% of probability mass)
    max_goals = 7
    x = np.arange(max_goals).reshape(-1, 1)
    y = np.arange(max_goals).reshape(1, -1)

    px = poisson.pmf(x, lmbda)
    py = poisson.pmf(y, mu)
    P = px * py

    # Dixon-Coles correction
    dc = np.ones_like(P)
    dc[0, 0] = 1 - lmbda * mu * rho_
    dc[0, 1] = 1 + lmbda * rho_
    dc[1, 0] = 1 + mu * rho_
    dc[1, 1] = 1 - rho_

    P *= dc

    H = np.sum(P[x > y])
    D = np.sum(P[x == y])
    A = np.sum(P[x < y])

    return H, D, A


def simulate_group(group_teams):
    """Simulate full group stage with goal tracking"""
    points = {t: 0 for t in group_teams}
    gf = {t: 0 for t in group_teams}
    ga = {t: 0 for t in group_teams}

    for i in range(len(group_teams)):
        for j in range(i + 1, len(group_teams)):
            home_goals, away_goals = simulate_match_score(group_teams[i], group_teams[j])

            gf[group_teams[i]] += home_goals
            ga[group_teams[i]] += away_goals
            gf[group_teams[j]] += away_goals
            ga[group_teams[j]] += home_goals

            if home_goals > away_goals:
                points[group_teams[i]] += 3
            elif away_goals > home_goals:
                points[group_teams[j]] += 3
            else:
                points[group_teams[i]] += 1
                points[group_teams[j]] += 1

    # FIFA tiebreakers: points, goal difference, goals scored
    ranked = sorted(group_teams,
                    key=lambda t: (points[t], gf[t] - ga[t], gf[t]),
                    reverse=True)

    return ranked, points, gf, ga


def select_best_thirds(third_place_teams):
    """
    Select 8 best third-place teams using FIFA rules.
    Returns: (list of 8 teams, dict mapping team to original group)
    """
    # third_place_teams: list of tuples: (team, group, points, gd, gf)
    #sorted based on fifa requirements for selction
    sorted_thirds = sorted(third_place_teams,
                           key=lambda x: (x[2], x[3], x[4]),
                           reverse=True)

    best_8 = sorted_thirds[:8]
    #creates a dictionary with the teams name as the key and the value their group
    team_to_group = {t[0]: t[1] for t in best_8}

    return [t[0] for t in best_8], team_to_group


def build_r32_matches(group_results, third_teams_map):
    """
    Build R32 matches according to FIFA bracket structure.
    third_teams_map: dict mapping qualifying third-place team -> their group
    """
    matches = []

    # Get which groups provided the 8 third-place teams
    third_groups = set(third_teams_map.values())

    for home_spec, away_spec in R32_STRUCTURE:
        # Parse home team
        if isinstance(home_spec, str):
            group_letter = home_spec[0]
            position = int(home_spec[1])  # 1=winner, 2=runner-up
            home_team = group_results[group_letter][position - 1]
        else:
            home_team = home_spec

        # Parse away team
        if isinstance(away_spec, str):
            group_letter = away_spec[0]
            position = int(away_spec[1])
            away_team = group_results[group_letter][position - 1]
        elif isinstance(away_spec, list):
            # Third place team from one of these groups
            # Find which group in away_spec actually qualified a third-place team
            possible_groups = [g for g in away_spec if g in third_groups]
            if possible_groups:
                # Pick the first one (in actual tournament, there's complex logic)
                # For simulation purposes, we'll use the first available
                selected_group = possible_groups[0]
                away_team = [t for t, g in third_teams_map.items() if g == selected_group][0]
                # Remove this group from third_groups to avoid reusing
                third_groups.discard(selected_group)
            else:
                # error trap; this shouldn't happen with proper bracket logic
                away_team = list(third_teams_map.keys())[0]
        else:
            away_team = away_spec

        matches.append((home_team, away_team))

    return matches


def simulate_knockout_match(home, away):
    #simulates each individual knockout match
    H, D, A = match_win_probability(home, away, neutral=True)
    # In knockout: draw goes to penalties (50/50 split of draw to each teams win)
    winner = random.choices([home, away], weights=[H + 0.5 * D, A + 0.5 * D])[0]
    return winner


def simulate_tournament():
    """Simulate entire tournament"""
    results = {
        'group_results': {},
        'r32': [],
        'r16': [],
        'qf': [],
        'sf': [],
        'final': [],
        'winner': None
    }

    # GROUP STAGE
    third_place_data = []

    for g_name, g_teams in groups.items():
        ranked, points, gf, ga = simulate_group(g_teams)
        results['group_results'][g_name] = ranked

        third = ranked[2]
        gd = gf[third] - ga[third]
        third_place_data.append((third, g_name, points[third], gd, gf[third]))

    # Select best 8 third-place teams
    best_thirds, third_teams_map = select_best_thirds(third_place_data)

    # Build R32 matches
    r32_matches = build_r32_matches(results['group_results'], third_teams_map)

    # Simulate R32
    for home, away in r32_matches:
        winner = simulate_knockout_match(home, away)
        results['r32'].append(winner)

    # R16
    #since the structure is already organised form r32 bracket logic
    #need not be reimplemented
    #uses the macthes list of tuples
    r16_teams = results['r32']
    for i in range(0, len(r16_teams), 2):
        winner = simulate_knockout_match(r16_teams[i], r16_teams[i + 1])
        results['r16'].append(winner)

    # QF
    qf_teams = results['r16']
    for i in range(0, len(qf_teams), 2):
        winner = simulate_knockout_match(qf_teams[i], qf_teams[i + 1])
        results['qf'].append(winner)

    # SF
    sf_teams = results['qf']
    for i in range(0, len(sf_teams), 2):
        winner = simulate_knockout_match(sf_teams[i], sf_teams[i + 1])
        results['sf'].append(winner)

    # FINAL
    final_teams = results['sf']
    results['final'] = final_teams
    results['winner'] = simulate_knockout_match(final_teams[0], final_teams[1])

    return results




def simulate_all_probabilities(N=10000):
   #run full tournament simulation to get probabilities for all teams at all stages.
    stage_counts = {team: {'r32': 0, 'r16': 0, 'qf': 0, 'sf': 0, 'final': 0, 'winner': 0}
                    for team in teams}
    for _ in tqdm(range(N), desc="Running tournament simulations"):
        results = simulate_tournament()
        #counts each time a team reaches each stage in the simualtions
        for team in results['r32']:
            stage_counts[team]['r32'] += 1
        for team in results['r16']:
            stage_counts[team]['r16'] += 1
        for team in results['qf']:
            stage_counts[team]['qf'] += 1
        for team in results['sf']:
            stage_counts[team]['sf'] += 1
        for team in results['final']:
            stage_counts[team]['final'] += 1
        stage_counts[results['winner']]['winner'] += 1
    # Convert to probabilities
    probabilities = {}
    for team in teams:
        probabilities[team] = {
            stage: count / N for stage, count in stage_counts[team].items()
        }
    return probabilities

#compiles the rsults of the model into an ev table relative to the odds
def compute_ev_table(probabilities, odds_dict):
    rows = []
    # Map betting market -> simulation stage
    stage_map = {
        "group": "r32",   # "to qualify from group"
        "final": "final"  # "to reach final"
    }
    for team, team_odds in odds_dict.items():
        # Skip teams not in simulation
        if team not in probabilities:
            continue
        for market, sim_stage in stage_map.items():
            odd = team_odds.get(market)
            if odd is None:
                continue
            p_model = probabilities[team][sim_stage]
            p_implied = 1 / odd
            ev = p_model * (odd - 1) - (1 - p_model)
            edge = p_model - p_implied
            rows.append({
                "team": team,
                "market": market,
                "odds": odd,
                "model_prob": round(p_model, 4),
                "implied_prob": round(p_implied, 4),
                "EV": round(ev, 4),
                "edge": round(edge, 4)
            })
    df_ev = pd.DataFrame(rows)
    df_ev = df_ev.sort_values("EV", ascending=False)
    return df_ev




def simulate_team_probability(team_name, stage, N=10000):
    #calculate probability of team reaching specific stage
    #used for the single mode
    count = 0

    for _ in tqdm(range(N), desc=f"Simulating {team_name} to {stage}"):
        results = simulate_tournament()

        if stage == 'r32':
            if team_name in results['r32']:
                count += 1
        elif stage == 'r16':
            if team_name in results['r16']:
                count += 1
        elif stage == 'qf':
            if team_name in results['qf']:
                count += 1
        elif stage == 'sf':
            if team_name in results['sf']:
                count += 1
        elif stage == 'final':
            if team_name in results['final']:
                count += 1
        elif stage == 'winner':
            if team_name == results['winner']:
                count += 1
    return count / N

# mode selection
if __name__ == "__main__":
    mode = input("Enter mode (single/bulk): ").strip().lower()
    if mode == "single":
        team_name = input("Enter the team to simulate: ")
        target_stage = input("Enter the stage (r32/r16/qf/sf/final/winner): ")
        odds = float(input("Enter the odds in decimal format: "))
        N = 10000
        p_stage = simulate_team_probability(team_name, target_stage, N)
        print(f"\nProbability that {team_name} reaches {target_stage}: {p_stage:.4f}")
        print(f"Implied probability from odds: {1 / odds:.4f}")
        ev = p_stage * (odds - 1) - (1 - p_stage)
        print(f"Expected Value: {ev:.4f}")
        model_edge = p_stage - (1 / odds)
        print(f"Model Edge: {model_edge:.4f} ({model_edge * 100:.2f}%)")


    elif mode == "bulk":
        N = int(input("Enter number of simulations (recommend 10000): "))
        print("\nRunning bulk simulation...")
        probabilities = simulate_all_probabilities(N)
        print("\nComparing to bookmaker odds...")
        ev_df = compute_ev_table(probabilities, odds)

        # Save full table
        ev_df.to_csv("ev_results.csv", index=False)

        # Display best bets
        print("\nTop +EV bets:")
        print(ev_df[ev_df["EV"] > 0].head(15).to_string(index=False))
        print("\nSaved full EV table to ev_results.csv")
