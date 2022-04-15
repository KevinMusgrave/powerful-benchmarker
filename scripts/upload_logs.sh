conda deactivate && conda activate "$5"
while :
do
    cd "$4"
    echo "Getting progress"
    python print_progress.py --exp_folder "$1" --save_to_file progress.txt --with_validator_progress
    progress_fail=$?
    if [ "$progress_fail" = "1" ];
        then
            echo "fail!"
        else
            echo "Moving progress.txt"
            mv progress.txt "$1"
            cd "$1"
            echo "Removing existing zip"
            rm all_logs.zip csvs.zip
            echo "Zipping files"
            find -maxdepth 4 -name *.err -o -name *.out | zip -qr all_logs -@
            find -maxdepth 4 -name trials.csv | zip -qr csvs -@
            zip -qu csvs.zip progress.txt
            for w in $6 $7
            do
                if [ -f $w ]; then
                zip -qu csvs.zip $w
                fi
            done
            echo "Deleting existing files"
            gdrive list -q "'$2' in parents" --no-header --max 0 | cut -d" " -f1 - | xargs -L 1 gdrive delete
            echo "Starting upload"
            gdrive upload --parent "$2" all_logs.zip
            gdrive upload --parent "$2" csvs.zip
    fi
    echo "Sleeping for $3"
    sleep "$3"
done