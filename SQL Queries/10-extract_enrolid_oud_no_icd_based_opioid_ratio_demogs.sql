
-- This query extract some demographic and metadata information for OUD-no patients for case control matching and inclusion exclusion criteria.
-- The result include:
-- patient's ID
-- NUmber of months in which they have been prescribed with an Opioid (other than Buprenorphine).
-- Note, it only consider Opioid prescriptions outside of a prediction window of 180 days.
-- Number of months in which they have been not prescribed with any Opioid.
-- Patient's sex
-- The decade that the patient were born in.



USE usr_sfouladvand;

DROP TABLE IF EXISTS oud_no_icd_presc_based_opioid_ratio_demogs;

SELECT B.ENROLID, 
	   B.num_pos, 
	   D.num_negs, 
	   F.SEX,
	   F.decade_dob,
	   ( CAST(B.num_pos AS FLOAT)/CAST((B.num_pos+D.num_negs) AS FLOAT) ) as opioid_ratio,
	   abs(checksum(newid()) % 10000) as random_idx
INTO oud_no_icd_presc_based_opioid_ratio_demogs
FROM 
(SELECT DISTINCT  A.ENROLID,
	   count( DISTINCT SUBSTRING(CONVERT(nvarchar, A.FILLDATE, 112), 1, 6)) AS num_pos
	   FROM [TRVNORM].[dbo].[PRESCRIPTION] A
	   INNER JOIN [oud_no_enrolid_and_date_range] A1
	   ON A1.ENROLID = A.ENROLID
	   WHERE (substring(A.TCGPI_ID,1,2) = '65')-- BECAUSE we already excluded patients with Bup. and Met. prescription out of negative cohort we dont need this: and substring(A.TCGPI_ID,1,8) != '65200010' and substring(A.TCGPI_ID,1,8) != '65100050' and substring(A.TCGPI_ID,1,8) != '96448248')  
	   AND DATEDIFF(DAY, A.FILLDATE, A1.MAX_DATE) > 180
	   GROUP BY A.ENROLID
		) B
inner join (SELECT DISTINCT C.ENROLID,
	   count( DISTINCT SUBSTRING(CONVERT(nvarchar, C.FILLDATE, 112), 1, 6)) AS num_negs
	   FROM [TRVNORM].[dbo].[PRESCRIPTION] C
	   INNER JOIN [oud_no_enrolid_and_date_range] C1
	   ON C1.ENROLID = C.ENROLID	   
	   WHERE not(substring(C.TCGPI_ID,1,2) = '65')-- and substring(C.TCGPI_ID,1,8) != '65200010' and substring(C.TCGPI_ID,1,8) != '65100050' and substring(A.TCGPI_ID,1,8) != '96448248')  
	   AND DATEDIFF(DAY, C.FILLDATE, C1.MAX_DATE) > 180
	   GROUP BY C.ENROLID	
		) D
on B.ENROLID = D.ENROLID
inner join (SELECT DISTINCT E.ENROLID, E.SEX, ((2018 - E.DOBYR)/10) as decade_dob
	   FROM [TRVNORM].[dbo].[ENROLID_DETAIL] E
		) F
		on B.ENROLID = F.ENROLID
inner join [oud_no_enrolid_and_date_range] G
		on G.ENROLID = F.ENROLID



