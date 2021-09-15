DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD_and_presc];


SELECT C.ENROLID, MIN(C.diagnoses_date) AS diagnoses_date 
INTO [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD_and_presc]
FROM
(SELECT A.ENROLID, A.diagnoses_date
FROM [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD] A
UNION
SELECT B.ENROLID, B.diagnoses_date
FROM [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_prescriptions] B
) C
GROUP BY C.ENROLID

