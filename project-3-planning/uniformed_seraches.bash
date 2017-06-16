#!/bin/bash
echo "Running uninformed planning seraches

Problems
-----------------
    1. Air Cargo Problem 1
    2. Air Cargo Problem 2
    3. Air Cargo Problem 3

Search Algorithms
-----------------
    1. breadth_first_search 
    2. breadth_first_tree_search 
    3. depth_first_graph_search 
    4. depth_limited_search 
    5. uniform_cost_search 
    6. recursive_best_first_search h_1
    7. greedy_best_first_graph_search h_1
    8. astar_search h_1
    9. astar_search h_ignore_preconditions
    10. astar_search h_pg_levelsum

-----------------
Starting Searches...
"

for problem in 1 2 3
do
    for test in 1 2 3 4 5 6 7
    do
        echo -n "Problem $problem test $test...."
        python run_search.py -p $problem -s $test >> p${problem}_uniformed_search.txt
        echo "Done"
    done
    echo -e "\nResults Can be found in: p${problem}_uniformed_search.txt\n"
done
echo "Tests Complete"