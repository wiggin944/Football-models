import pandas as pd
from sqlalchemy import create_engine, event

from .settings import AGG_STATS, SOURCE_DB


def make_engine(db_path=SOURCE_DB):
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _sqlite_settings(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")
        dbapi_connection.execute("PRAGMA busy_timeout=30000")

    return engine


def read_panel(engine):
    cols = AGG_STATS + ["tkl"]
    stat_select = ", ".join(f"ss.{c}" for c in cols)

    query = f"""
        SELECT st.team_id, st.competition_id, st.season_id,
               t.name AS squad, c.name AS comp, se.label AS season,
               st.position AS pos, st.n90, {stat_select}
        FROM stints st
        JOIN teams t         ON st.team_id = t.team_id
        JOIN competitions c  ON st.competition_id = c.competition_id
        JOIN seasons se      ON st.season_id = se.season_id
        JOIN stint_stats ss  ON st.stint_id = ss.stint_id
    """

    return pd.read_sql(query, engine)


def read_team_match_metrics(engine):
    query = """
        SELECT tms.team_id, m.competition_id, m.season_id,
               AVG(tms.ppda) AS ppda,
               AVG(tms.ppda_allowed) AS ppda_allowed,
               AVG(tms.deep) AS deep,
               AVG(tms.deep_allowed) AS deep_allowed,
               AVG(tms.xg) AS xg_per_match
        FROM team_match_stats tms
        JOIN matches m ON tms.match_id = m.match_id
        WHERE tms.team_id IS NOT NULL
          AND m.competition_id IS NOT NULL
        GROUP BY tms.team_id, m.competition_id, m.season_id
    """

    return pd.read_sql(query, engine)


def read_setpiece(engine):
    query = """
        SELECT team_id, season_id,
               setpiece_xg_share,
               transition_threat_proxy
        FROM team_setpiece_stats
        WHERE team_id IS NOT NULL
          AND season_id IS NOT NULL
    """

    return pd.read_sql(query, engine)
