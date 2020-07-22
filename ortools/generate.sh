for anum in {5..5..1}
do
    for cnum in {50..50..50}
    do
        for dnum in {1..1000..1}
        do
            python main.py 'test' $anum $cnum $dnum 2
        done
    done
done

#for anum in {5..10..1}
#do
#    for cnum in {50..100..50}
#    do
#        for dnum in {1..1000..1}
#        do
#            python main.py 'test' $anum $cnum $dnum 2
#        done
#    done
#done
#
#for anum in {5..10..1}
#do
#    for cnum in {50..100..50}
#    do
#        for dnum in {1..1000..1}
#        do
#            python main.py 'test' $anum $cnum $dnum 5
#        done
#    done
#done
#
#
#for anum in {5..10..1}
#do
#    for cnum in {200..1000..50}
#    do
#        for dnum in {1..500..1}
#        do
#            python main.py 'test' $anum $cnum $dnum 5
#        done
#    done
#done