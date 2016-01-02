for file in "dev" "test" "train"; do 
    python download.py  ../data/tagged_data/whole_text_full_city2/$file.tag $file.extra;
done;
