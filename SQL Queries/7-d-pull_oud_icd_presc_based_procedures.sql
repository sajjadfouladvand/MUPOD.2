--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_procedures];
USE usr_sfouladvand;
GO
--ALTER
CREATE VIEW oud_yes_icd_presc_based_procedures_view
AS
select distinct B.[ENROLID]
      ,B.[SVCDATE]
      ,B.[PROCCD]
      --,B.[PROCTYP]
--into  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_procedures]
from [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join [TRVNORM].[dbo].[ENCOUNTER_PROCEDURE] B
on A.ENROLID = B.ENROLID 
;

