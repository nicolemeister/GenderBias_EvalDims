#!/bin/bash

SESSION="multi_run_semicolons"
GROUP_SIZE=5

# All commands
commands=(

"python analyze.py --name armstrong --job karvonen"
"python analyze.py --name armstrong --job zollo"
"python analyze.py --name armstrong --job yin"

"python analyze.py --name rozado --job armstrong"
"python analyze.py --name rozado --job rozado"
"python analyze.py --name rozado --job wang"
"python analyze.py --name rozado --job wen"

"python analyze.py --name wang --job armstrong"

"python analyze.py --name wen --job armstrong"
"python analyze.py --name wen --job rozado"
"python analyze.py --name wen --job wen"
"python analyze.py --name wen --job karvonen"
"python analyze.py --name wen --job zollo"
"python analyze.py --name wen --job yin"

"python analyze.py --name karvonen --job armstrong" 
"python analyze.py --name karvonen --job rozado"
"python analyze.py --name karvonen --job wang"
"python analyze.py --name karvonen --job wen"
"python analyze.py --name karvonen --job karvonen"

"python analyze.py --name zollo --job rozado"
"python analyze.py --name zollo --job wang"
"python analyze.py --name zollo --job wen"
"python analyze.py --name zollo --job karvonen"
"python analyze.py --name zollo --job zollo"
"python analyze.py --name zollo --job yin"

"python analyze.py --name yin --job armstrong"  
"python analyze.py --name yin --job karvonen"
"python analyze.py --name yin --job zollo"
"python analyze.py --name yin --job yin"

)

total=${#commands[@]}
window=0

# Create tmux session
tmux new-session -d -s $SESSION

# Process commands in groups of $GROUP_SIZE
for ((i=0; i<total; i+=GROUP_SIZE)); do
    group=("${commands[@]:i:GROUP_SIZE}")

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
    ((window++))
done

echo "Launched $total commands across $window tmux windows."
echo "Attach with: tmux attach -t $SESSION"





# "python analyze.py --name armstrong --job armstrong"
# "python analyze.py --name armstrong --job rozado"
# "python analyze.py --name armstrong --job wang"
# "python analyze.py --name armstrong --job wen"
# "python analyze.py --name armstrong --job karvonen"
# "python analyze.py --name armstrong --job zollo"
# "python analyze.py --name armstrong --job yin"

# "python analyze.py --name rozado --job armstrong"
# "python analyze.py --name rozado --job rozado"
# "python analyze.py --name rozado --job wang"
# "python analyze.py --name rozado --job wen"
# "python analyze.py --name rozado --job karvonen"
# "python analyze.py --name rozado --job zollo"
# "python analyze.py --name rozado --job yin"

# "python analyze.py --name wang --job armstrong"
# "python analyze.py --name wang --job rozado"
# "python analyze.py --name wang --job wang"
# "python analyze.py --name wang --job wen"
# "python analyze.py --name wang --job karvonen"
# "python analyze.py --name wang --job zollo"
# "python analyze.py --name wang --job yin"

# "python analyze.py --name wen --job armstrong"
# "python analyze.py --name wen --job rozado"
# "python analyze.py --name wen --job wang"
# "python analyze.py --name wen --job wen"
# "python analyze.py --name wen --job karvonen"
# "python analyze.py --name wen --job zollo"
# "python analyze.py --name wen --job yin"

# "python analyze.py --name karvonen --job armstrong"
# "python analyze.py --name karvonen --job rozado"
# "python analyze.py --name karvonen --job wang"
# "python analyze.py --name karvonen --job wen"
# "python analyze.py --name karvonen --job karvonen"
# "python analyze.py --name karvonen --job zollo"
# "python analyze.py --name karvonen --job yin"

# "python analyze.py --name zollo --job armstrong"
# "python analyze.py --name zollo --job rozado"
# "python analyze.py --name zollo --job wang"
# "python analyze.py --name zollo --job wen"
# "python analyze.py --name zollo --job karvonen"
# "python analyze.py --name zollo --job zollo"
# "python analyze.py --name zollo --job yin"

# "python analyze.py --name yin --job armstrong"
# "python analyze.py --name yin --job rozado"
# "python analyze.py --name yin --job wang"
# "python analyze.py --name yin --job wen"
# "python analyze.py --name yin --job karvonen"
# "python analyze.py --name yin --job zollo"
# "python analyze.py --name yin --job yin"

# "python analyze.py --name gaeb --job armstrong"
# "python analyze.py --name gaeb --job rozado"
# "python analyze.py --name gaeb --job wang"
# "python analyze.py --name gaeb --job wen"
# "python analyze.py --name gaeb --job karvonen"
# "python analyze.py --name gaeb --job zollo"
# "python analyze.py --name gaeb --job yin"

# "python analyze.py --name lippens --job armstrong"
# "python analyze.py --name lippens --job wang"
# "python analyze.py --name lippens --job wen"
# "python analyze.py --name lippens --job rozado"
# "python analyze.py --name lippens --job zollo"
# "python analyze.py --name lippens --job yin"
# "python analyze.py --name lippens --job karvonen"

# "python analyze.py --name seshadri --job armstrong"
# "python analyze.py --name seshadri --job wang"
# "python analyze.py --name seshadri --job wen"
# "python analyze.py --name seshadri --job rozado"
# "python analyze.py --name seshadri --job zollo"
# "python analyze.py --name seshadri --job yin"
# "python analyze.py --name seshadri --job karvonen"
