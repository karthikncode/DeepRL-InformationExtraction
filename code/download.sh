i=0
for extraQuery in "adulterated" "scandal" "countries" "fake"; do
	i=$((i + 1))
	for file in "dev" "test" "train"; do 
    	python download.py  ../data/tagged_data/EMA/$file.tag $file.$i.extra $extraQuery &
	done;
done;

wait
echo "Download complete"
