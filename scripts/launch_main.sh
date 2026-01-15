#!/bin/bash

SESSION="main_run"
GROUP_SIZE=1  # Now interpreted as the number of tmux windows to create

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

total=${#commands[@]}

if (( GROUP_SIZE < 1 )); then
  echo "GROUP_SIZE must be at least 1"
  exit 1
fi

# Calculate number of commands per window
per_window=$(( (total + GROUP_SIZE - 1) / GROUP_SIZE ))

tmux new-session -d -s $SESSION

for ((window=0; window<GROUP_SIZE; window++)); do
    start=$((window * per_window))
    end=$((start + per_window))
    if (( end > total )); then
        end=$total
    fi

    if (( start >= total )); then
        break
    fi

    group=("${commands[@]:start:end-start}")

    # Join with semicolons
    longcmd=$(printf "%s; " "${group[@]}")
    longcmd="${longcmd%; }"  # remove trailing semicolon+space

    # Prefix with cd into EvalDims first
    fullcmd="cd EvalDims; $longcmd"

    echo "$fullcmd"
    if [ $window -eq 0 ]; then
        tmux rename-window -t $SESSION:0 "win_$window"
    else
        tmux new-window -t $SESSION -n "win_$window"
    fi

    tmux send-keys -t $SESSION:win_$window "$fullcmd" C-m
done

echo "Launched $total commands across $GROUP_SIZE tmux windows (approx $per_window per window)."
echo "Attach with: tmux attach -t $SESSION"
