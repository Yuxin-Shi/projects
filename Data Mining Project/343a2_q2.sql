SET search_path TO parlgov;

CREATE VIEW participation_rate as
(SELECT e_date, country_id, votes_cast * 1.0 / electorate as p_rate
 FROM election
 WHERE votes_cast is NOT NULL
);

CREATE VIEW rate_country as
(SELECT country_id, EXTRACT(YEAR FROM e_date) as e_year, avg(p_rate) as avg_rate
 FROM participation_rate
 GROUP BY country_id, EXTRACT(YEAR FROM e_date)
 HAVING EXTRACT(YEAR FROM e_date) >=2001 and EXTRACT(YEAR FROM e_date) <= 2016
);

CREATE VIEW have_decreasing as
(SELECT r1.country_id
 FROM rate_country r1, rate_country r2
 WHERE r1.country_id = r2.country_id and r1.e_year < r2.e_year and r1. avg_rate > r2.avg_rate
);

CREATE VIEW participate_p1 as
(SELECT country_id
 FROM ((SELECT country_id FROM rate_country)
             EXCEPT
             (SELECT country_id FROM have_decreasing)) as t
);

CREATE TABLE participate as
(SELECT name as countryName, e_year as year, avg_rate as participationRatio
 FROM rate_country 
            RIGHT JOIN participate_p1 ON rate_country.country_id = participate_p1.country_id
            LEFT JOIN country ON id = participate_p1.country_id
);


