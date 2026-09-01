from sql_guard import validate_read_only

# Row cap so a runaway SELECT cannot exhaust memory on the API host.
MAX_ROWS = 5000


def validate_tsql(sql):
    """Return (is_safe, reason). Allows one read-only SELECT / WITH...SELECT."""
    return validate_read_only(sql)


def execute_tsql(conn, sql, max_rows=MAX_ROWS):
    """
    Execute a read-only query and return (columns, rows).

    Re-validates before executing: callers should already have validated, but
    this is the last point before the query reaches the database, so it must
    not depend on a caller remembering to check.
    """
    is_safe, reason = validate_tsql(sql)
    if not is_safe:
        raise ValueError(f"Refusing to execute non-read-only query: {reason}")

    cursor = conn.cursor()
    cursor.execute(sql)

    columns = []
    if cursor.description:
        columns = [col[0] for col in cursor.description]

    # fetchmany rather than fetchall so an unbounded result set cannot be
    # pulled into memory in full.
    rows = [tuple(row) for row in cursor.fetchmany(max_rows)]

    cursor.close()
    return columns, rows
