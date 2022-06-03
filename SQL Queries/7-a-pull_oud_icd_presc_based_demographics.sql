-- This query extract demographic features for OUD-no cohort

--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_demographics];
USE usr_sfouladvand;

GO
--ALTER
CREATE VIEW oud_yes_icd_presc_based_demographics_view
AS
SELECT DISTINCT B.[ENROLID]
      ,B.[DOBYR]
      ,B.[SEX]
--INTO  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_demographics]
FROM [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join [TRVNORM].[dbo].[ENROLID_DETAIL] B
ON A.ENROLID = B.ENROLID 

;

