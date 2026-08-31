"""
Connection privilege check.

sql_guard.py stops a write from being generated. That is application code: any
new endpoint, refactor, or direct call that skips it removes the protection.
The durable control is the database login itself — an account that only holds
SELECT cannot delete data no matter what SQL reaches it.

This module reports what the connected account can actually do, so the service
can refuse to run (or at minimum say so loudly) when it is connected with a
privileged login such as `sa`.

Set REQUIRE_READONLY_DB=1 to make an over-privileged connection a hard failure.
Default is to warn, so enabling the check cannot take a running system down
before the read-only login has been provisioned.
"""

import logging
import os
from functools import lru_cache

logger = logging.getLogger("db_privileges")

STRICT = os.environ.get("REQUIRE_READONLY_DB", "").strip().lower() in ("1", "true", "yes")

# Roles/permissions that would let the account modify data or schema.
_PRIVILEGE_SQL = """
SELECT
    SUSER_SNAME()                                    AS login_name,
    DB_NAME()                                        AS database_name,
    CONVERT(INT, IS_SRVROLEMEMBER('sysadmin'))       AS is_sysadmin,
    CONVERT(INT, ISNULL(IS_ROLEMEMBER('db_owner'), 0))      AS is_db_owner,
    CONVERT(INT, ISNULL(IS_ROLEMEMBER('db_datawriter'), 0)) AS is_data_writer,
    CONVERT(INT, ISNULL(IS_ROLEMEMBER('db_ddladmin'), 0))   AS is_ddl_admin,
    CONVERT(INT, HAS_PERMS_BY_NAME(NULL, NULL, 'CONTROL SERVER')) AS has_control_server
"""

_WRITE_FLAGS = [
    ("is_sysadmin", "sysadmin (server role)"),
    ("has_control_server", "CONTROL SERVER"),
    ("is_db_owner", "db_owner"),
    ("is_data_writer", "db_datawriter"),
    ("is_ddl_admin", "db_ddladmin"),
]


class OverPrivilegedConnection(RuntimeError):
    """Raised in strict mode when the DB account can modify data."""


def check_privileges(conn):
    """
    Return a report describing what the connected account can do.

    Never raises — a server that will not answer the question yields
    {"known": False} so the caller can proceed and log the uncertainty.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(_PRIVILEGE_SQL)
        row = cursor.fetchone()
        columns = [c[0] for c in cursor.description]
    except Exception as exc:
        logger.warning("Could not determine connection privileges: %s", exc)
        return {"known": False, "write_capable": None, "grants": []}
    finally:
        try:
            cursor.close()
        except Exception:
            pass

    report = dict(zip(columns, row))
    grants = [label for key, label in _WRITE_FLAGS if report.get(key)]

    return {
        "known": True,
        "login_name": report.get("login_name"),
        "database_name": report.get("database_name"),
        "write_capable": bool(grants),
        "grants": grants,
    }


def describe(report):
    """One-line human summary of a privilege report."""
    if not report.get("known"):
        return "Database privileges could not be determined."
    if not report.get("write_capable"):
        return (
            f"Login '{report['login_name']}' is read-only on "
            f"'{report['database_name']}'."
        )
    return (
        f"Login '{report['login_name']}' can MODIFY data on "
        f"'{report['database_name']}' (holds: {', '.join(report['grants'])}). "
        "Provision a SELECT-only login — see scripts/create_readonly_login.sql."
    )


@lru_cache(maxsize=32)
def _cached_verdict(server, database, username):
    """Cache key only. The value is filled in by enforce_read_only."""
    return {}


def enforce_read_only(conn, server="", database="", username="", strict=None):
    """
    Check the connection and act on the result.

    Returns the report. In strict mode a write-capable connection raises
    OverPrivilegedConnection; otherwise it is logged as a warning once per
    (server, database, login) rather than on every request.
    """
    strict = STRICT if strict is None else strict

    cache = _cached_verdict(server, database, username)
    if "report" in cache:
        report = cache["report"]
    else:
        report = check_privileges(conn)
        cache["report"] = report
        # Log once per distinct connection identity, not per question.
        if report.get("write_capable"):
            logger.warning("SECURITY: %s", describe(report))
        else:
            logger.info("%s", describe(report))

    if strict and report.get("write_capable"):
        raise OverPrivilegedConnection(describe(report))

    return report
