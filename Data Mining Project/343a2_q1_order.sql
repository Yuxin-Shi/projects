SET search_path TO parlgov;

SELECT * FROM winners 
ORDER BY countryName, wonElections, partyName DESC;
