-- This query extracts medication (prescription) records for the OUD cohort. 
-- Note, since the result table is too large we used views and didn't actually generated the table. Then we use the MS SQL Server IDE to export the view in a csv file.

--DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_prescriptions];
USE usr_sfouladvand;
GO
--ALTER
USE usr_sfouladvand;
GO
CREATE VIEW oud_yes_icd_presc_based_prescriptions_view
AS
select distinct B.ENROLID 
				,B.FILLDATE 
				,B.TCGPI_ID 
				--,B.NDCNUM
				--,B.ROOT_CLASSIFICATION
				--,B.SECONDARY_CLASSIFICATION
				--,B.TCGPI_NAME
				--,B.DRUG_NAME
--into  [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_prescriptions]
from [usr_sfouladvand].[dbo].[oud_yes_icd_presc_based_opioid_ratio_demogs_filtered3] A
left join  [TRVNORM].[dbo].[PRESCRIPTION] B
on A.ENROLID = B.ENROLID 
;
