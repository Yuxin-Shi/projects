SET search_path TO parlgov;

CREATE VIEW cabinet_sequence as
(SELECT c1.id, c1.country_id, c1.start_date, c2.start_date as end_date
 FROM cabinet c1 LEFT JOIN cabinet c2 ON c1.id = c2.previous_cabinet_id
 ORDER BY c1.id
);

CREATE VIEW party_sequence as
(SELECT * FROM cabinet_sequence LEFT JOIN 
                          (SELECT party_id, cabinet_id, pm FROM cabinet_party WHERE pm = 't') as c1 
	          ON cabinet_sequence.id = c1.cabinet_id
 ORDER BY cabinet_sequence.id
);

CREATE TABLE Sequences_of_Cabinets as
(SELECT country.name as countryName, party_sequence.id as cabinetId,
 start_date as startDate, end_date as endDate, party.name as pmParty
 FROM party_sequence LEFT JOIN party ON party_sequence.id = party.id
                                    LEFT JOIN country ON party_sequence.country_id = country.id
);



