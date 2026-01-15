#!/bin/bash                                                                                                                                                                                                                                                              

commands=(

"python main.py --name armstrong --job armstrong --model gpt-5-nano-2025-08-07"
"python main.py --name armstrong --job rozado --model gpt-5-nano-2025-08-07"
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