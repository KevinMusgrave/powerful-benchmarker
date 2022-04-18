echo "Deleting existing files from $2"
gdrive list -q "'$2' in parents" --no-header --max 0 | cut -d" " -f1 - | xargs -L 1 gdrive delete
echo "Starting upload"
for w in $1
do
    if [ -f $w ]; then
    echo "Uploading $w"
    gdrive upload --parent "$2" $w
    fi
done
