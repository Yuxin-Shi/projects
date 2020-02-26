SET search_path TO parlgov;

CREATE VIEW parliamentary_cabinet_sequence as
(SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, election_id, e_date, e_type
 FROM election JOIN cabinet ON election_id = election.id
 ORDER BY election_id
);

CREATE VIEW european_parliament_sequence as
(SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, t1.id as election_id, after_date as e_date, t1.e_type
 FROM  cabinet JOIN 
            (SELECT e2.id, e2.country_id, e1.e_date as before_date, e2.e_date as after_date, e2.e_type
             FROM election e1 JOIN election e2 ON e1.previous_ep_election_id = e2.id
             WHERE e1.e_type = 'European Parliament' and e2.e_type = 'European Parliament'
             ) as t1 ON cabinet.country_id = t1.country_id
 WHERE cabinet.start_date >= after_date and cabinet.start_date <= before_date
 ) UNION ALL
 (SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, t2.id, e_date, t2.e_type
  FROM cabinet JOIN 
            (SELECT id, country_id, e_date, e_type FROM election
             WHERE id NOT IN (SELECT previous_ep_election_id FROM election 
     		           WHERE e_type = 'European Parliament' and previous_ep_election_id is NOT NULL) 
              and e_type = 'European Parliament'
             ) as t2 ON cabinet.country_id = t2.country_id
  WHERE cabinet.start_date >= t2.e_date
 );

CREATE TABLE election_Sequence as
(SELECT cabinet_id, election_id 
 FROM (SELECT * FROM parliamentary_cabinet_sequence  UNION ALL SELECT * FROM european_parliament_sequence) as a1
 ORDER BY e_date DESC, cabinet_id, CASE e_type WHEN 'Parliamentary election' THEN 1 WHEN 'European Parliament' THEN 2 END
);






SELECT cabinet_id, election_id
FROM(
SELECT cabinet_id, a1.election_id, country.name as country_name
FROM ((SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, election_id, e_date, e_type
           FROM election JOIN cabinet ON election_id = election.id
           ORDER BY election_id) UNION ALL (SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, t1.id as election_id, after_date as e_date, t1.e_type
                                                                  FROM  cabinet JOIN 
                                                                              (SELECT e2.id, e2.country_id, e1.e_date as before_date, e2.e_date as after_date, e2.e_type
                                                                               FROM election e1 JOIN election e2 ON e1.previous_ep_election_id = e2.id
                                                                               WHERE e1.e_type = 'European Parliament' and e2.e_type = 'European Parliament'
                                                                               ) as t1 ON cabinet.country_id = t1.country_id
                                                                  WHERE cabinet.start_date >= after_date and cabinet.start_date <= before_date
                                                                  ) UNION ALL
                                                                 (SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, t2.id, e_date, t2.e_type
                                                                  FROM cabinet JOIN 
                                                                             (SELECT id, country_id, e_date, e_type FROM election
                                                                              WHERE id NOT IN (SELECT previous_ep_election_id FROM election 
     		                                                                            WHERE e_type = 'European Parliament' and previous_ep_election_id is NOT NULL) 
                                                                              and e_type = 'European Parliament'
                                                                              ) as t2 ON cabinet.country_id = t2.country_id
                                                                  WHERE cabinet.start_date >= t2.e_date)) as a1 LEFT JOIN cabinet ON a1.cabinet_id = cabinet.id 
								          LEFT JOIN  country ON country.id = cabinet.country_id
ORDER BY e_date DESC, cabinet_id, CASE e_type WHEN 'Parliamentary election' THEN 1 WHEN 'European Parliament' THEN 2 END
) as a2
WHERE country_name = 'France';




