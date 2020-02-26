SET search_path TO parlgov;

CREATE VIEW pair_times as
(SELECT e1.party_id as alliedPartyId1, e2.party_id as alliedPartyId2, count(e1.election_id) as num_times
 FROM election_result e1, election_result e2
 WHERE e1.election_id = e2.election_id and e1.id = e2.alliance_id
 GROUP BY e1.party_id, e2.party_id
);

CREATE VIEW num_eleciton_country as
(SELECT country_id, count(id) as num_election
 FROM election
 GROUP BY country_id
);

CREATE TABLE Election_Alliances as
(SELECT alliedPartyId1, alliedPartyId2, num_eleciton_country.country_id as countryId
 FROM pair_times LEFT JOIN party ON pair_times.alliedPartyId1 = party.id
                            LEFT JOIN num_eleciton_country ON party.country_id = num_eleciton_country.country_id
 WHERE num_times >= 0.3 * num_election
);


