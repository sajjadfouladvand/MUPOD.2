-- This query creates a view of OUD-no demographic records.

--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_demographics];

-- ALTER
USE usr_sfouladvand;
GO

CREATE VIEW oud_no_icd_presc_based_demographics_view
AS
select distinct B.[ENROLID]
      ,B.[DOBYR]
      ,B.[SEX]
--into  [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_demographics]
from [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join [TRVNORM].[dbo].[ENROLID_DETAIL] B
on A.ENROLID = B.ENROLID 
;

