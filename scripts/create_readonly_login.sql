/*
================================================================================
 SatBot read-only database login
================================================================================

 WHY
 ---
 The service currently connects as 'sa' (sysadmin, db_owner, db_datawriter,
 CONTROL SERVER). The only thing preventing an LLM-generated DELETE or DROP from
 executing is application code in sql_guard.py. That code is one refactor away
 from being bypassed.

 This script creates a login that CANNOT write, so a destructive statement fails
 at the database even if it somehow reaches it.

 HOW TO RUN
 ----------
 1. Change @Password below to a real secret.
 2. Review @DatabaseName — run the script once per database SatBot may read.
 3. Execute in SSMS as an administrator.
 4. Point the service at the new login (see "AFTER RUNNING" at the bottom).

 SAFE TO RE-RUN: every step is guarded by an existence check.
================================================================================
*/

USE [master];
GO

DECLARE @LoginName  SYSNAME       = N'satbot_reader';
DECLARE @Password   NVARCHAR(128) = N'CHANGE_ME_BEFORE_RUNNING';   -- <<< SET THIS

IF @Password = N'CHANGE_ME_BEFORE_RUNNING'
BEGIN
    RAISERROR('Set @Password to a real secret before running this script.', 16, 1);
    RETURN;
END

-- ---------------------------------------------------------------------------
-- 1. Server login. No server roles are granted, so it is a plain login with
--    no rights anywhere until explicitly given them per database below.
-- ---------------------------------------------------------------------------
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = @LoginName)
BEGIN
    DECLARE @sql NVARCHAR(MAX) = N'
        CREATE LOGIN ' + QUOTENAME(@LoginName) + N'
        WITH PASSWORD = ' + QUOTENAME(@Password, '''') + N',
             CHECK_POLICY = ON,
             DEFAULT_DATABASE = [master];';
    EXEC sys.sp_executesql @sql;
    PRINT 'Created login: ' + @LoginName;
END
ELSE
    PRINT 'Login already exists: ' + @LoginName;
GO


/*
================================================================================
 2. PER-DATABASE GRANTS
    Change the USE statement and re-run this block for each database SatBot
    is allowed to read (e.g. Hims_Zrams).
================================================================================
*/
USE [Hims_Zrams];          -- <<< one database per run
GO

DECLARE @LoginName SYSNAME = N'satbot_reader';

-- Database user mapped to the login
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = @LoginName)
BEGIN
    DECLARE @sql NVARCHAR(MAX) =
        N'CREATE USER ' + QUOTENAME(@LoginName) +
        N' FOR LOGIN ' + QUOTENAME(@LoginName) + N';';
    EXEC sys.sp_executesql @sql;
    PRINT 'Created user in ' + DB_NAME();
END

-- Read: db_datareader grants SELECT on every table and view, including tables
-- added later, so newly ticked tables work without re-running this script.
ALTER ROLE [db_datareader] ADD MEMBER [satbot_reader];

-- Write: DENY overrides any GRANT, including one added by mistake later.
-- This is the control that actually stops DELETE / TRUNCATE / DROP / UPDATE.
DENY INSERT, UPDATE, DELETE                 TO [satbot_reader];
DENY ALTER, CONTROL, CREATE TABLE, CREATE VIEW,
     CREATE PROCEDURE, CREATE FUNCTION      TO [satbot_reader];
DENY EXECUTE                                TO [satbot_reader];
DENY BACKUP DATABASE, BACKUP LOG            TO [satbot_reader];

-- Explicitly keep it out of every writable fixed role.
IF IS_ROLEMEMBER('db_owner',      'satbot_reader') = 1
    ALTER ROLE [db_owner]      DROP MEMBER [satbot_reader];
IF IS_ROLEMEMBER('db_datawriter','satbot_reader') = 1
    ALTER ROLE [db_datawriter] DROP MEMBER [satbot_reader];
IF IS_ROLEMEMBER('db_ddladmin',  'satbot_reader') = 1
    ALTER ROLE [db_ddladmin]   DROP MEMBER [satbot_reader];

PRINT 'Granted read-only access on ' + DB_NAME() + ' to satbot_reader';
GO


/*
================================================================================
 3. VERIFY — run this while connected AS satbot_reader.
    Expected: is_sysadmin/is_db_owner/is_data_writer/is_ddl_admin all 0.
================================================================================

    SELECT
        SUSER_SNAME()                                    AS login_name,
        DB_NAME()                                        AS database_name,
        IS_SRVROLEMEMBER('sysadmin')                     AS is_sysadmin,
        ISNULL(IS_ROLEMEMBER('db_owner'), 0)             AS is_db_owner,
        ISNULL(IS_ROLEMEMBER('db_datawriter'), 0)        AS is_data_writer,
        ISNULL(IS_ROLEMEMBER('db_ddladmin'), 0)          AS is_ddl_admin,
        HAS_PERMS_BY_NAME(NULL, NULL, 'CONTROL SERVER')  AS has_control_server;

    -- These must all fail with "permission denied":
    --   DELETE FROM dbo.RoadMaster WHERE 1 = 0;
    --   UPDATE dbo.RoadMaster SET Length = Length WHERE 1 = 0;
    --   SELECT TOP 1 * INTO dbo.__perm_test FROM dbo.RoadMaster;
    --   DROP TABLE dbo.__perm_test;

    -- This must succeed:
    --   SELECT COUNT(*) FROM dbo.RoadMaster;


================================================================================
 AFTER RUNNING
================================================================================
 1. Repoint the service at the new login. In the SatBot admin screen, reconnect
    the environment with username 'satbot_reader' and the password you set.

 2. Turn on enforcement so the service refuses a privileged connection:

        set REQUIRE_READONLY_DB=1
        uvicorn fastapi_app:app --host 0.0.0.0 --port 8001

    With this set, connecting as a write-capable login raises
    OverPrivilegedConnection instead of silently working (see db_privileges.py).

 3. Confirm: GET /admin/check-db-status now reports read_only = true.
================================================================================
*/
