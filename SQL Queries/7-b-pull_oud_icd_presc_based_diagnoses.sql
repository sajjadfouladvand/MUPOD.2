-- This query extracts diagnosis records for the OUD cohort. 
-- Note, ENCOUNTER_DIAGNOSIS table contain diagnosis records. 
-- Note, since the result table is too large we used views and didn't actually generated the table. Then we use the MS SQL Server IDE to export the view in a csv file.



--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_diagnoses];
USE usr_sfouladvand;
GO
--ALTER
CREATE VIEW oud_yes_icd_presc_based_diagnoses_view
AS
select distinct 
	B.ENROLID,
	B.SVCDATE, 
	B.DIAG_CD, 
	B.CCS_CATGRY 
--INTO  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_diagnoses]
from [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join [TRVNORM].[dbo].[ENCOUNTER_DIAGNOSIS] B
on A.ENROLID = B.ENROLID 
;

