-- infrastructure/postgres/init.sql
-- Bootstrap schema for RAG Platform
-- Runs automatically when postgres container first starts

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for text search

-- ── Documents metadata ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    use_case_id     VARCHAR(64)  NOT NULL,
    user_id         VARCHAR(64)  NOT NULL,
    filename        TEXT         NOT NULL,
    content_type    VARCHAR(128) NOT NULL,
    size_bytes      BIGINT       NOT NULL,
    status          VARCHAR(32)  NOT NULL DEFAULT 'pending',
    chunk_count     INTEGER      NOT NULL DEFAULT 0,
    metadata        JSONB        NOT NULL DEFAULT '{}',
    error_message   TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    indexed_at      TIMESTAMPTZ
);

CREATE INDEX idx_documents_use_case ON documents(use_case_id);
CREATE INDEX idx_documents_user     ON documents(user_id);
CREATE INDEX idx_documents_status   ON documents(status);

-- ── Query audit log ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         VARCHAR(64)  NOT NULL,
    use_case_id     VARCHAR(64)  NOT NULL,
    query           TEXT         NOT NULL,
    answer          TEXT,
    confidence      FLOAT,
    escalated       BOOLEAN      NOT NULL DEFAULT FALSE,
    latency_ms      INTEGER,
    token_usage     JSONB        NOT NULL DEFAULT '{}',
    session_id      VARCHAR(64),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_query_logs_user       ON query_logs(user_id);
CREATE INDEX idx_query_logs_use_case   ON query_logs(use_case_id);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at DESC);
CREATE INDEX idx_query_logs_escalated  ON query_logs(escalated) WHERE escalated = TRUE;

-- ── Evaluation results ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    use_case_id     VARCHAR(64)  NOT NULL,
    faithfulness    FLOAT        NOT NULL,
    answer_relevancy FLOAT       NOT NULL,
    context_recall  FLOAT        NOT NULL,
    passed          BOOLEAN      NOT NULL,
    sample_size     INTEGER      NOT NULL DEFAULT 0,
    evaluated_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eval_use_case ON eval_results(use_case_id);

-- ── Token usage per team (for budget tracking) ──────────────────
CREATE TABLE IF NOT EXISTS token_usage (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    use_case_id     VARCHAR(64)  NOT NULL,
    user_id         VARCHAR(64)  NOT NULL,
    input_tokens    INTEGER      NOT NULL DEFAULT 0,
    output_tokens   INTEGER      NOT NULL DEFAULT 0,
    cost_usd        FLOAT        NOT NULL DEFAULT 0.0,
    recorded_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_token_use_case ON token_usage(use_case_id);
CREATE INDEX idx_token_recorded ON token_usage(recorded_at DESC);

-- ── View: daily cost per use case ───────────────────────────────
CREATE OR REPLACE VIEW v_daily_cost AS
SELECT
    use_case_id,
    DATE(recorded_at) AS day,
    SUM(input_tokens)  AS total_input_tokens,
    SUM(output_tokens) AS total_output_tokens,
    SUM(cost_usd)      AS total_cost_usd
FROM token_usage
GROUP BY use_case_id, DATE(recorded_at)
ORDER BY day DESC, total_cost_usd DESC;

COMMENT ON TABLE documents    IS 'Document metadata — actual vectors stored in Qdrant';
COMMENT ON TABLE query_logs   IS 'Full audit trail of every query (GDPR: review retention policy)';
COMMENT ON TABLE eval_results IS 'Ragas evaluation runs per use case';
COMMENT ON TABLE token_usage  IS 'Per-request token tracking for budget management';
