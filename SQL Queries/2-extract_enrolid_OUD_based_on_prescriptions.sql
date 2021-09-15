-- THIS QUERY FINDS OUD PATINETS BY LOOKING INTO THEIR PRESCRIPTIONS. 
USE usr_sfouladvand;

DROP TABLE IF EXISTS  [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_prescriptions];

SELECT B.ENROLID, B.diagnoses_date
INTO oud_yes_enrolids_based_on_prescriptions
FROM
(SELECT  A.ENROLID, MIN(A.FILLDATE) AS diagnoses_date
FROM [TRVNORM].[dbo].[PRESCRIPTION] A
WHERE A.TCGPI_ID like '65200010%' 
OR A.TCGPI_ID like '65100050%'  
OR A.TCGPI_ID like '96448248%' 
GROUP BY A.ENROLID) B

CREATE INDEX oud_patients_based_on_prescriptions_enrolids_idx on [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_prescriptions] (ENROLID);





/*
SELECT B.ENROLID, B.diagnoses_date
INTO oud_yes_enrolids_based_on_prescriptions
FROM
(SELECT  A.ENROLID, MIN(A.FILLDATE) AS diagnoses_date
FROM [TRVNORM].[dbo].[PRESCRIPTION] A
WHERE A.TCGPI_ID like '65200010%' 
OR A.TCGPI_ID like '65100050%'  
OR A.TCGPI_ID like '96448248%' 
GROUP BY A.ENROLID) B
inner join ENROLID_DETAIL_VIEW C
on B.ENROLID = C.ENROLID
WHERE DATEDIFF(DAY, C.ELIG_SPAN_BEGIN, B.diagnoses_date) > 365

CREATE INDEX oud_patients_based_on_prescriptions_enrolids_idx on [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_prescriptions] (ENROLID);
*/