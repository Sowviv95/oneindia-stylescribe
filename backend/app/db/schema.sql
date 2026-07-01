CREATE TABLE IF NOT EXISTS authors (
    author_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    language TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS author_articles (
    article_id TEXT PRIMARY KEY,
    author_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    heading TEXT,
    url TEXT,
    category TEXT,
    tags TEXT,
    keywords TEXT,
    meta_description TEXT,
    added_date TEXT,
    content_from_metadata TEXT,
    extracted_text TEXT NOT NULL,
    text_char_count INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(author_id) REFERENCES authors(author_id)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    author_id TEXT NOT NULL,
    status TEXT NOT NULL,
    articles_seen INTEGER NOT NULL,
    articles_ingested INTEGER NOT NULL,
    articles_failed INTEGER NOT NULL,
    metadata_rows_seen INTEGER NOT NULL,
    warnings_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS author_style_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    author_id TEXT NOT NULL,
    article_count INTEGER NOT NULL,
    language TEXT NOT NULL,
    status TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    excerpt_pack_json TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(author_id) REFERENCES authors(author_id)
);
