-- This script extract some demographic information for the OUD cohort. 
-- The result of this query will be later used to match cases and controls.
-- The result shows patients ID, number of months in which they had Opioid prescriptions (other than BUprenorphine) 
-- and number of month with no Opioid prescriptions. 
-- and the ratio of the previous two metrics 
-- Patient's sex and the decade that they were born in (for matching purpose).
-- Note, TCGPI with the first two digits of 65 is an Opioid medication. And 65200010 and 65100050 are excluded because they are Buprenorphine.
-- Note, it only consider Opioid prescriptions outside of a prediction window of 180 days.

USE usr_sfouladvand;

DROP TABLE IF EXISTS oud_yes_icd_presc_based_opioid_ratio_demogs;

SELECT B.ENROLID, 
	   B.num_pos, 
	   D.num_negs, 
	   F.SEX,
	   F.decade_dob,
	   ( CAST(B.num_pos AS FLOAT)/CAST((B.num_pos+D.num_negs) AS FLOAT) ) as opioid_ratio
INTO oud_yes_icd_presc_based_opioid_ratio_demogs
FROM 
(SELECT DISTINCT  A.ENROLID,
	   count( DISTINCT SUBSTRING(CONVERT(nvarchar, A.FILLDATE, 112), 1, 6)) AS num_pos
	   FROM [TRVNORM].[dbo].[PRESCRIPTION] A
	   inner join [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD_and_presc] B1
	   on B1.ENROLID = A.ENROLID
	   WHERE (substring(A.TCGPI_ID,1,2) = '65' and substring(A.TCGPI_ID,1,8) != '65200010' and substring(A.TCGPI_ID,1,8) != '65100050' and substring(A.TCGPI_ID,1,8) != '96448248') 
	   AND    DATEDIFF(day, A.FILLDATE, B1.diagnoses_date) > 180 -- This 180 is used to make sure we only consider the opiiod prescriptions till 6 month before the OUD diagnoses date. The reason is that we try to predict OUD 6 month before
	   GROUP BY A.ENROLID
		) B
inner join (SELECT DISTINCT C.ENROLID,
	   count( DISTINCT SUBSTRING(CONVERT(nvarchar, C.FILLDATE, 112), 1, 6)) AS num_negs
	   FROM [TRVNORM].[dbo].[PRESCRIPTION] C
	   inner join [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD_and_presc] D1
	   on D1.ENROLID = C.ENROLID
	   WHERE not(substring(C.TCGPI_ID,1,2) = '65')-- Although these two are not Opioid but they are not non-opioid either: and substring(C.TCGPI_ID,1,8) != '65200010' and substring(C.TCGPI_ID,1,8) != '65100050')  
	   AND    DATEDIFF(day, C.FILLDATE, D1.diagnoses_date) > 180
	   GROUP BY C.ENROLID	
		) D
on B.ENROLID = D.ENROLID
inner join (SELECT DISTINCT E.ENROLID, E.SEX, ((2018 - E.DOBYR)/10) as decade_dob
	   FROM [TRVNORM].[dbo].[ENROLID_DETAIL] E
		) F
		on B.ENROLID = F.ENROLID
inner join [usr_sfouladvand].[dbo].[oud_yes_enrolids_based_on_ICD_and_presc] G
		on G.ENROLID = F.ENROLID



