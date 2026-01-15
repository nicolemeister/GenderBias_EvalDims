#!/bin/bash
#SBATCH --job-name=generate_outlines
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --array=0-69                   # REQUIRED: Run indices 0 through 69 (70 total jobs)

#SBATCH --output=/nlp/scr/nmeist/EvalDims/scripts/output_logs/launch_%A_%a.log
#SBATCH --error=/nlp/scr/nmeist/EvalDims/scripts/error_logs/launch_%A_%a.err

#SBATCH --cpus-per-task=2              # Matches your successful interactive session
#SBATCH --mem=10G                      # 10G is safe (your interactive session used 8G)
#SBATCH --time=2-00:00:00              # 2 Days (D-HH:MM:SS). Adjust if your jobs need longer.
#SBATCH --open-mode=append             # Append to logs rather than overwrite

cd ../

commands=(
  "python main.py --name armstrong --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name armstrong --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name armstrong --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name armstrong --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name armstrong --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name rozado --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name rozado --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name rozado --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name rozado --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name rozado --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name rozado --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name rozado --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name wang --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name wang --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name wang --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name wang --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name wang --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name wang --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name wang --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name wen --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name wen --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name wen --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name wen --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name wen --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name wen --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name wen --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name karvonen --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name karvonen --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name karvonen --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name karvonen --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name karvonen --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name karvonen --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name karvonen --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name zollo --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name zollo --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name zollo --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name zollo --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name zollo --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name zollo --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name zollo --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name yin --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name yin --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name yin --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name yin --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name yin --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name yin --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name yin --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name gaeb --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name gaeb --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name gaeb --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name gaeb --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name gaeb --job karvonen --model gpt-5-nano-2025-08-07"
  "python main.py --name gaeb --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name gaeb --job yin --model gpt-5-nano-2025-08-07"

  "python main.py --name lippens --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name lippens --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name lippens --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name lippens --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name lippens --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name lippens --job yin --model gpt-5-nano-2025-08-07"
  "python main.py --name lippens --job karvonen --model gpt-5-nano-2025-08-07"

  "python main.py --name seshadri --job armstrong --model gpt-5-nano-2025-08-07"
  "python main.py --name seshadri --job wang --model gpt-5-nano-2025-08-07"
  "python main.py --name seshadri --job wen --model gpt-5-nano-2025-08-07"
  "python main.py --name seshadri --job rozado --model gpt-5-nano-2025-08-07"
  "python main.py --name seshadri --job zollo --model gpt-5-nano-2025-08-07"
  "python main.py --name seshadri --job yin --model gpt-5-nano-2025-08-07"
  "python main.py --name seshadri --job karvonen --model gpt-5-nano-2025-08-07"
)

cmd="${commands[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: ${cmd}"
exec bash -lc "${cmd}"