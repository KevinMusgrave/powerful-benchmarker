cd $1
echo "finding all_dfs.pkl files"
rm all_dfs.zip
find -maxdepth 2 -name all_dfs.pkl | zip -qr all_dfs -@