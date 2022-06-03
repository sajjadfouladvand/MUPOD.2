-- This query just sample the data and I have used it for debugging purpose as the actual data was too big. 
-- Note, the results in our final models and publications are using the entire data.

USE usr_sfouladvand;

DROP TABLE IF EXISTS oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3_sampled;

SELECT TOP 10000 [ENROLID]
      ,[num_pos]
      ,[num_negs]
      ,[SEX]
      ,[decade_dob]
      ,[opioid_ratio]
INTO oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3_sampled
FROM [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3] 
ORDER BY NEWID()
