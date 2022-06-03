-- This query finds all patients who have at least one ICD diagnosis of OUD and their first diagnosis date. 
-- The result is a table with two columns one representing OUD patients IDs and the other their OUD diagnosis date.

DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD];

SELECT ED.ENROLID, MIN(ED.SVCDATE) AS diagnoses_date
INTO [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD]
FROM [TRVNORM].[dbo].[ENCOUNTER_DIAGNOSIS] ED
--  JOIN TRVNORM..ENCOUNTER E
--    ON ED.ENCOUNTERID = E.ENCOUNTERID
WHERE (ED.DIAG_CD like 'F11%'
   OR ED.DIAG_CD like '3040%'
   OR ED.DIAG_CD like '3055%')
--   AND E.IS_ELIG_1YR_PRIOR = 1
GROUP BY ED.ENROLID
