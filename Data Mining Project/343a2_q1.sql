SET search_path TO parlgov;

CREATE VIEW notwin_party as
(SELECT e1.election_id, e1.party_id
 FROM  election_result e1, election_result e2
 WHERE e1.election_id = e2.election_id and e1.votes < e2.votes
);

CREATE VIEW win_party as
((SELECT election_id , party_id FROM election_result WHERE votes is NOT NULL) 
 EXCEPT
 (SELECT * FROM notwin_party)
);

CREATE VIEW party_win_times as
(SELECT party_id, country_id, count(election_id) as win_times
 FROM win_party JOIN party ON win_party.party_id = party.id
 GROUP BY party_id, country_id
);

CREATE VIEW default_win_times as
(SELECT id, country_id, 0 as default_wintime  
 FROM 
 (SELECT distinct party.id, country_id 
 FROM election_result JOIN party ON election_result.party_id = party.id
 ) as p
);

CREATE VIEW all_party_win_times as
(SELECT * 
 FROM (SELECT * FROM party_win_times
 UNION
 SELECT * FROM default_win_times WHERE id NOT IN
                                      (SELECT party_id FROM party_win_times))
 as a
);

CREATE VIEW find_winners as
(SELECT * 
 FROM ((SELECT * FROM all_party_win_times) as a
 NATURAL LEFT JOIN 
 (SELECT country_id, avg(win_times) as avg_country FROM all_party_win_times GROUP BY country_id) as b 
 ) as p
);

CREATE VIEW winners_p1 as
(SELECT  country.name as countryName, party.name as partyName, 
 party_family.family as partyFamily, win_times as wonElections, find_winners.party_id
 FROM find_winners 
            LEFT JOIN party ON find_winners.party_id = party.id 
            LEFT JOIN party_family ON find_winners.party_id = party_family.party_id 
            LEFT JOIN country ON find_winners.country_id = country.id 
 WHERE win_times > 3 * avg_country
);

CREATE VIEW winners_p2 as
(SELECT countryName, partyName, partyFamily, wonElections, winners_p1.party_id, election.id, e_date
 FROM winners_p1 
            JOIN win_party ON winners_p1.party_id = win_party.party_id
            LEFT JOIN election ON election_id = election.id
);

CREATE VIEW old_win as
(SELECT w1.partyFamily, w1.party_id, w1.id, w1.e_date
 FROM winners_p2 w1, winners_p2 w2
 WHERE (w1.partyFamily = w2.partyFamily or w1.partyFamily is NULL) and w1.party_id = w2.party_id and w1.e_date < w2.e_date
);

CREATE VIEW recent_win as
((SELECT partyFamily, party_id, id, e_date FROM winners_p2)
 EXCEPT
 (SELECT * FROM old_win)
);

CREATE TABLE winners as
(SELECT countryName, partyName, winners_p2.partyFamily, wonElections, 
             recent_win.id as mostRecentlyWonElectionId, recent_win.e_date as mostRecentlyWonElectionYear
 FROM recent_win, winners_p2
 WHERE (recent_win.partyFamily = winners_p2.partyFamily or recent_win.partyFamily is NULL) 
              and recent_win.party_id = winners_p2.party_id and recent_win.id = winners_p2.id
);


