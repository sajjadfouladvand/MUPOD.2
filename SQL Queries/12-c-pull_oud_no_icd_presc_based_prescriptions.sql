-- This query creates a view of OUD-no medication (prescription) records.
-- Note, because the resulted tables are too large we create views and then use the MS SQL Server to export these views into csv files.


--DROP VIEW IF EXISTS oud_no_icd_presc_based_prescriptions_view;

--ALTER
CREATE VIEW oud_no_icd_presc_based_prescriptions_view
AS
SELECT  B.ENROLID 
				,B.FILLDATE 
				,B.TCGPI_ID
				--,B.ROOT_CLASSIFICATION
				--,B.SECONDARY_CLASSIFICATION
				--,B.TCGPI_NAME
				--,B.DRUG_NAME
FROM [usr_sfouladvand].[dbo].[oud_no_icd_presc_based_opioid_ratio_demogs_filtered3] A
LEFT JOIN  [TRVNORM].[dbo].[PRESCRIPTION] B
ON A.ENROLID = B.ENROLID 
;
