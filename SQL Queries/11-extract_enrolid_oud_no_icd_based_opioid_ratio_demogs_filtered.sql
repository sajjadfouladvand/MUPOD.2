/****** Script for SelectTopNRows command from SSMS  ******/
DROP TABLE IF EXISTS oud_no_icd_presc_based_opioid_ratio_demogs_filtered3
SELECT [ENROLID]
      ,[num_pos]
      ,[num_negs]
      ,[SEX]
      ,[decade_dob]
      ,[opioid_ratio]
      ,[random_idx]
  INTO oud_no_icd_presc_based_opioid_ratio_demogs_filtered3
  FROM [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_opioid_ratio_demogs] A
  WHERE A.num_pos >=3
