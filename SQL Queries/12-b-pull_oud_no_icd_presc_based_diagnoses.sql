-- This query creates a view of OUD-no diagnosis records.
-- Note, because the resulted tables are too large we create views and then use the MS SQL Server to export these views into csv files.


--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_diagnoses];

-- ALTER
CREATE VIEW oud_no_icd_presc_based_diagnoses_view
AS
SELECT DISTINCT 
	B.ENROLID,
	B.SVCDATE, 
	B.DIAG_CD, 
	B.CCS_CATGRY 
--INTO  [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_diagnoses]
FROM [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join [TRVNORM].[dbo].[ENCOUNTER_DIAGNOSIS] B
ON A.ENROLID = B.ENROLID 
;

