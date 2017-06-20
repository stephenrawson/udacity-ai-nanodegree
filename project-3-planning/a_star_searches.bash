#!/bin/bash
echo "Running A* planning searches

Problems
-----------------
    1. Air Cargo Problem 1
    2. Air Cargo Problem 2
    3. Air Cargo Problem 3

Search Algorithms
-----------------
    8. astar_search h_1
    9. astar_search h_ignore_preconditions
    10. astar_search h_pg_levelsum

-----------------
Starting Searches...
"

for problem in 1 2 3
do
    for test in 8 9 10
    do
        echo -n "Problem $problem test $test...."
        python run_search.py -p $problem -s $test >> p${problem}_a_star_search.txt
        echo "Done"
    done
    echo -e "\nResults Can be found in: p${problem}_a_star_search.txt\n"
done
echo "Tests Complete"