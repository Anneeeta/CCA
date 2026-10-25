# Anonymous paired human-evaluation results

This directory contains the 40 paired sessions used for the manuscript's human
evaluation: ten topics, two comparisons, and two sessions per topic–comparison
cell. There are 20 CCA–RAG robot sessions and 20 CCA–Co-STORM sessions.

- `paired_sessions.json`: one record per paired session, with the two systems'
  original nine-metric ratings.
- `ratings.csv`: the same ratings in long form, with 720 system–metric values.
- `topics.json`: the ten predefined study topics.

Only session IDs S001–S040, predefined topics, comparison systems and ratings are
published. Session IDs do not identify people. Names, submitted participant
identifiers, P codes, cross-session participant linkage, original filenames,
timestamps and free-text dialogue are not included. No identity lookup is provided.

Collected ratings take values 2, 4, 6, 8 and 10. The `reported_score` column divides
the collected value by two for the manuscript's 1–5 presentation. The auxiliary
Overall field is outside the nine-metric analysis and is not included.

The 2026-10-26 update changes the public metadata schema to session-level results;
the selected 40 sessions and every rating value remain unchanged. The offline
analysis remains in `../analysis/reproduce_human.py`.
