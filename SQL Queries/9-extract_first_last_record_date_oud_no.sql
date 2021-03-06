-- This query extracts some metadata for OUD-no cohort for matching purposes.
-- This query extracts first and last record dates for all OUD-no patients.
-- It also exclude patinets who have data availability for less than or equal to 356 days

USE usr_sfouladvand;

DROP TABLE IF EXISTS oud_no_enrolid_and_date_range;

SELECT B.ENROLID
		, B.MIN_DATE
		, B.MAX_DATE
INTO oud_no_enrolid_and_date_range
FROM
(SELECT A.ENROLID
		, MIN(A.FIRST_RECORD_DATE) AS MIN_DATE
		, MAX(A.LAST_RECORD_DATE) AS MAX_DATE
FROM
(SELECT  CTR1.ENROLID
		, MIN(DG.SVCDATE) AS FIRST_RECORD_DATE
		, MAX(DG.SVCDATE) AS LAST_RECORD_DATE
FROM oud_no_icd_presc_based_enrolids_all CTR1
INNER JOIN [TRVNORM].[dbo].[ENCOUNTER_DIAGNOSIS] DG
ON CTR1.ENROLID = DG.ENROLID
GROUP BY CTR1.ENROLID
UNION
SELECT  CTR2.ENROLID
		, MIN(PRS.FILLDATE) AS FIRST_RECORD_DATE
		, MAX(PRS.FILLDATE) AS LAST_RECORD_DATE
FROM oud_no_icd_presc_based_enrolids_all CTR2
INNER JOIN [TRVNORM].[dbo].[PRESCRIPTION] PRS
ON CTR2.ENROLID = PRS.ENROLID
GROUP BY CTR2.ENROLID
UNION
SELECT CTR3.ENROLID
		, MIN(PCD.SVCDATE) AS FIRST_RECORD_DATE
		, MAX(PCD.SVCDATE) AS LAST_RECORD_DATE
FROM oud_no_icd_presc_based_enrolids_all CTR3
INNER JOIN [TRVNORM].[dbo].[ENCOUNTER_PROCEDURE] PCD
ON CTR3.ENROLID = PCD.ENROLID
GROUP BY CTR3.ENROLID ) A
GROUP BY A.ENROLID ) B
WHERE DATEDIFF(DAY, B.MIN_DATE, B.MAX_DATE) > 365
