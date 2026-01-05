import random
from tqdm import tqdm
import main


def knockout_winner(home, away, neutral=False):
    """
    Uses already-fitted Dixon–Coles model from main.py
    """
    H, D, A = main.match_win_probability(home, away, neutral=neutral)
    return random.choices(
        [home, away],
        weights=[H + 0.5 * D, A + 0.5 * D]
    )[0]


def ireland_qualify_prob_unknown_final(N=10000):
    """
    Q: What % chance do you give Ireland to qualify,
       BEFORE knowing final opponent?
    """
    qualify = 0

    for _ in tqdm(range(N), desc="Ireland qualify (opponent unknown)"):
        # Semi-final: Czechia home
        if knockout_winner("Czechia", "Republic of Ireland") != "Republic of Ireland":
            continue

        # Other semi-final
        finalist = knockout_winner("Denmark", "North Macedonia")

        # Final at Aviva (Ireland home)
        if knockout_winner("Republic of Ireland", finalist) == "Republic of Ireland":
            qualify += 1

    return qualify / N


def ireland_qualify_prob_vs_macedonia(N=10000):
    """
    Q: How does this change if we know final opponent = North Macedonia?
    """
    qualify = 0

    for _ in tqdm(range(N), desc="Ireland qualify (vs Macedonia)"):
        if knockout_winner("Czechia", "Republic of Ireland") != "Republic of Ireland":
            continue

        if knockout_winner("Republic of Ireland", "North Macedonia") == "Republic of Ireland":
            qualify += 1

    return qualify / N


if __name__ == "__main__":
    N = 10000

    p1 = ireland_qualify_prob_unknown_final(N)
    p2 = ireland_qualify_prob_vs_macedonia(N)

    print(f"\nIreland qualification probability (opponent unknown): {p1:.2%}")
    print(f"Ireland qualification probability (final vs North Macedonia): {p2:.2%}")
