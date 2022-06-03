-- This query creates a view of OUD-no procedure records.
-- Note, because the resulted tables are too large we create views and then use the MS SQL Server to export these views into csv files.

--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_no_icd_based_procedures];

-- ALTER
USE usr_sfouladvand;
GO

CREATE VIEW oud_no_icd_presc_based_procedures_view
AS
SELECT DISTINCT B.[ENROLID]
      ,B.[SVCDATE]
      ,B.[PROCCD]
      --,B.[PROCTYP]
--INTO  [usr_sfouladvand].[dbo].[oud_no_icd_based_procedures]
from [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join [TRVNORM].[dbo].[ENCOUNTER_PROCEDURE] B
on A.ENROLID  = B.ENROLID 
;

