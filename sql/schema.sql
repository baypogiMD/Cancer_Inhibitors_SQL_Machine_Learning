CREATE TABLE inhibitors (
    compound_id TEXT PRIMARY KEY,
    smiles TEXT,
    ic50 REAL,
    pIC50 REAL,
    molecular_weight REAL,
    logp REAL,
    hbd INTEGER,
    hba INTEGER,
    tpsa REAL,
    activity_label INTEGER
);
