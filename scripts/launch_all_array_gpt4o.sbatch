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
  "python main.py --name armstrong --job armstrong"
  "python main.py --name armstrong --job rozado"
  "python main.py --name armstrong --job wang"
  "python main.py --name armstrong --job wen"
  "python main.py --name armstrong --job karvonen"
  "python main.py --name armstrong --job zollo"
  "python main.py --name armstrong --job yin"

  "python main.py --name rozado --job armstrong"
  "python main.py --name rozado --job rozado"
  "python main.py --name rozado --job wang"
  "python main.py --name rozado --job wen"
  "python main.py --name rozado --job karvonen"
  "python main.py --name rozado --job zollo"
  "python main.py --name rozado --job yin"

  "python main.py --name wang --job armstrong"
  "python main.py --name wang --job rozado"
  "python main.py --name wang --job wang"
  "python main.py --name wang --job wen"
  "python main.py --name wang --job karvonen"
  "python main.py --name wang --job zollo"
  "python main.py --name wang --job yin"

  "python main.py --name wen --job armstrong"
  "python main.py --name wen --job rozado"
  "python main.py --name wen --job wang"
  "python main.py --name wen --job wen"
  "python main.py --name wen --job karvonen"
  "python main.py --name wen --job zollo"
  "python main.py --name wen --job yin"

  "python main.py --name karvonen --job armstrong"
  "python main.py --name karvonen --job rozado"
  "python main.py --name karvonen --job wang"
  "python main.py --name karvonen --job wen"
  "python main.py --name karvonen --job karvonen"
  "python main.py --name karvonen --job zollo"
  "python main.py --name karvonen --job yin"

  "python main.py --name zollo --job armstrong"
  "python main.py --name zollo --job rozado"
  "python main.py --name zollo --job wang"
  "python main.py --name zollo --job wen"
  "python main.py --name zollo --job karvonen"
  "python main.py --name zollo --job zollo"
  "python main.py --name zollo --job yin"

  "python main.py --name yin --job armstrong"
  "python main.py --name yin --job rozado"
  "python main.py --name yin --job wang"
  "python main.py --name yin --job wen"
  "python main.py --name yin --job karvonen"
  "python main.py --name yin --job zollo"
  "python main.py --name yin --job yin"

  "python main.py --name gaeb --job armstrong"
  "python main.py --name gaeb --job rozado"
  "python main.py --name gaeb --job wang"
  "python main.py --name gaeb --job wen"
  "python main.py --name gaeb --job karvonen"
  "python main.py --name gaeb --job zollo"
  "python main.py --name gaeb --job yin"

  "python main.py --name lippens --job armstrong"
  "python main.py --name lippens --job wang"
  "python main.py --name lippens --job wen"
  "python main.py --name lippens --job rozado"
  "python main.py --name lippens --job zollo"
  "python main.py --name lippens --job yin"
  "python main.py --name lippens --job karvonen"

  "python main.py --name seshadri --job armstrong"
  "python main.py --name seshadri --job wang"
  "python main.py --name seshadri --job wen"
  "python main.py --name seshadri --job rozado"
  "python main.py --name seshadri --job zollo"
  "python main.py --name seshadri --job yin"
  "python main.py --name seshadri --job karvonen"
)

cmd="${commands[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: ${cmd}"
exec bash -lc "${cmd}"