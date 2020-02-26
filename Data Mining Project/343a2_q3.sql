SET search_path TO parlgov;

CREATE VIEW yearly_country_cabinet as
(SELECT id, country_id, EXTRACT(YEAR FROM start_date) as start_year
 FROM cabinet
 WHERE EXTRACT(YEAR FROM start_date) >= 1999 and EXTRACT(YEAR FROM start_date) < 2019
);

CREATE VIEW cabinet_with_party as
(SELECT cabinet_party.cabinet_id, country_id, start_year, party_id
 FROM  yearly_country_cabinet JOIN cabinet_party ON  yearly_country_cabinet.id = cabinet_party.cabinet_id
);

CREATE VIEW num_cabinet_country as
(SELECT country_id, count(DISTINCT cabinet_id) as num_cabinet1 
 FROM cabinet_with_party
 GROUP BY country_id
);

CREATE VIEW num_cabinet_party as
(SELECT party_id, count(DISTINCT cabinet_id) as num_cabinet2 
 FROM cabinet_with_party
 GROUP BY party_id
);

CREATE VIEW check_every_cabinet as
(SELECT cabinet_id, cabinet_with_party.country_id, start_year, cabinet_with_party.party_id
 FROM cabinet_with_party LEFT JOIN num_cabinet_country ON cabinet_with_party.country_id = num_cabinet_country.country_id
                                         LEFT JOIN num_cabinet_party ON cabinet_with_party.party_id = num_cabinet_party.party_id
 WHERE num_cabinet2 = num_cabinet1
);

CREATE TABLE Commited_parties as
(SELECT  country.name as countryName, party.name as partyName, family as partyFamily, 
 state_market as stateMarket
 FROM check_every_cabinet LEFT JOIN country ON check_every_cabinet.country_id = country.id
                                           LEFT JOIN party_family ON check_every_cabinet.party_id = party_family.party_id
                                           LEFT JOIN party ON check_every_cabinet.party_id = party.id
                                           LEFT JOIN party_position ON check_every_cabinet.party_id = party_position.party_id
);

