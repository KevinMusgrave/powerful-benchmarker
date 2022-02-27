conda deactivate && conda activate "$6"
while :
do
    cd "$5"
    echo "Getting progress"
    python summary.py --root_experiment_folder "$1" --save_to_file summary.txt
    summary_fail=$?
    python print_progress.py --root_experiment_folder "$1" --save_to_file progress.txt
    progress_fail=$?
    if [ "$summary_fail" = "1" ] && [ "$progress_fail" = "1" ];
        then
            echo "fail!"
        else
            echo "Moving summary.txt"
            mv summary.txt "$1"    
            echo "Moving progress.txt"
            mv progress.txt "$1"
            cd "$1"
            echo "Removing existing zip"
            rm all_logs.zip csvs.zip
            echo "Zipping files"
            find -maxdepth 4 -name *.err -newermt '5 hours ago' -o -name *.out -newermt '5 hours ago' | zip -qr all_logs -@
            find -maxdepth 4 -name trials.csv -o -name score_vs_test_accuracy.csv -o -name reproduction_score_vs_test_accuracy.csv -o -name study.pkl | zip -qr csvs -@
            zip -qu csvs.zip summary.txt progress.txt
            echo "Deleting existing files"
            gdrive list -q "'$3' in parents" --no-header --max 0 | cut -d" " -f1 - | xargs -L 1 gdrive delete
            echo "Starting upload"
            gdrive upload --parent "$3" all_logs.zip
            gdrive upload --parent "$3" csvs.zip
    fi
    echo "Sleeping for $4"
    sleep "$4"
done