-- This query extracts patients who:
-- have been prescribed with Opioid medications at least once
-- have never been diagnosed with any OUD ICD diagnoses codes
-- Have never been prescribed with Bup. or Met.

use usr_sfouladvand;
-- 145216721
DROP TABLE IF EXISTS oud_no_icd_presc_based_enrolids_all;

SELECT P1.ENROLID
INTO oud_no_icd_presc_based_enrolids_all
FROM [TRVNORM].[dbo].[PRESCRIPTION] P1
WHERE substring(P1.TCGPI_ID,1,2) = '65' 
EXCEPT
(SELECT D.ENROLID
FROM [TRVNORM].[dbo].[ENCOUNTER_DIAGNOSIS] D
WHERE D.DIAG_CD like 'F11%'
    OR D.DIAG_CD like '3040%'
    OR D.DIAG_CD like '3055%'
UNION 
SELECT P2.ENROLID
FROM [TRVNORM].[dbo].[PRESCRIPTION] P2
WHERE P2.TCGPI_ID like '65200010%' 
	OR P2.TCGPI_ID like '65100050%'  
	OR P2.TCGPI_ID like '96448248%' ) 
