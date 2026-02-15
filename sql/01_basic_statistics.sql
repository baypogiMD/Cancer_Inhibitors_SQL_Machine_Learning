SELECT
    COUNT(*) AS total_compounds,
    AVG(pIC50) AS mean_pIC50,
    MIN(pIC50) AS min_pIC50,
    MAX(pIC50) AS max_pIC50
FROM inhibitors;
