/****** Script for SelectTopNRows command from SSMS  ******/
USE usr_sfouladvand;

DROP TABLE IF EXISTS oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3;

SELECT  [ENROLID]
      ,[num_pos]
      ,[num_negs]
      ,[SEX]
      ,[decade_dob]
      ,[opioid_ratio]
  INTO oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3
  FROM [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_opioid_ratio_demogs] A
  WHERE A.num_pos >=3