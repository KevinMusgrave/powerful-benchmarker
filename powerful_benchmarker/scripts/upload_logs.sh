conda deactivate && conda activate "$6"
while :
do
    cd "$5"
    echo "Getting progress"
    python -m powerful_benchmarker.print_progress --exp_folder "$1" --save_to_file progress.txt
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
            echo "Deleting existing files"
            gdrive list -q "'$3' in parents" --no-header --max 0 | cut -d" " -f1 - | xargs -L 1 gdrive delete
            echo "Starting upload"
            gdrive upload --parent "$3" all_logs.zip
            gdrive upload --parent "$3" csvs.zip
    fi
    echo "Sleeping for $4"
    sleep "$4"
done