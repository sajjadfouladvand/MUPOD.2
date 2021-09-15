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

