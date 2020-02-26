SET search_path TO parlgov;

SELECT * FROM Election_Alliances ORDER BY countryId DESC, alliedPartyId1 DESC, alliedPartyId2 DESC;
