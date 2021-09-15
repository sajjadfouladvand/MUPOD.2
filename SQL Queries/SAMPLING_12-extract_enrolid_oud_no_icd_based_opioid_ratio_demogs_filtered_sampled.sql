
DROP TABLE IF EXISTS oud_no_icd_presc_based_opioid_ratio_demogs_filtered3_sampled;

SELECT TOP 100000 [ENROLID]

INTO oud_no_icd_presc_based_opioid_ratio_demogs_filtered3_sampled
FROM [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_opioid_ratio_demogs_filtered3] 
ORDER BY NEWID()
