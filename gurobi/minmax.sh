agent=2
city=50
for num in {1..1000..1}
do
    python entrance.py $city $agent $num 3600
done