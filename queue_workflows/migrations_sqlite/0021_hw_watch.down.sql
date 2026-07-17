-- Revert 0021: drop the hardware flight-recorder table (indexes go with it).
DROP TABLE IF EXISTS hw_watch_samples;
