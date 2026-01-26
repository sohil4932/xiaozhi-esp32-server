-- ============================================================================
-- Add Realtime API Support to Agent Configuration
-- ============================================================================
-- This migration adds realtime_model_id column to support OpenAI Realtime API
-- Date: 2026-01-24 23:00
-- ============================================================================

START TRANSACTION;

-- Add realtime_model_id column to ai_agent table
ALTER TABLE ai_agent
ADD COLUMN realtime_model_id VARCHAR(32) COMMENT 'Realtime API' AFTER intent_model_id;

-- Add realtime_model_id column to ai_agent_template table
ALTER TABLE ai_agent_template
ADD COLUMN realtime_model_id VARCHAR(32) COMMENT 'Realtime API' AFTER intent_model_id;

-- Add index for faster lookups
CREATE INDEX idx_ai_agent_realtime_model ON ai_agent(realtime_model_id);

COMMIT;
